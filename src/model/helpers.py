from collections import namedtuple
import re
import nltk
import math
import time
import numpy as np
import torch
from typing import List
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
nltk.download("punkt")

################################################################################
#########################       Training Helpers       #########################
################################################################################
Hypothesis = namedtuple('Hypothesis', ['value', 'score'])

def batch_iter(data, batch_size, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    :param data: list of tuples containing source and target sentence. ie.
        (list of (src_sent, tgt_sent))
    :type data: List[Tuple[List[str], List[str]]]
    :param batch_size: batch size
    :type batch_size: int
    :param shuffle: whether to randomly shuffle the dataset
    :type shuffle: boolean
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]

        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]

        yield src_sents, tgt_sents

def evaluate_ppl(model, val_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    :param model: NMT Model
    :type model: NMT
    :param dev_data: list of tuples containing source and target sentence.
        i.e. (list of (src_sent, tgt_sent))
    :param val_data: List[Tuple[List[str], List[str]]]
    :param batch_size: size of batches to extract
    :type batch_size: int
    :returns ppl: perplexity on val sentences
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(val_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)
        avg_val_loss = cum_loss / len(val_data)

    if was_training:
        model.train()

    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:
    """ Given decoding results and reference sentences, compute corpus-level BLEU score.
    :param references: a list of gold-standard reference target sentences
    :type references: List[List[str]]
    :param hypotheses: a list of hypotheses, one for each reference
    :type hypotheses: List[Hypothesis]
    :returns bleu_score: corpus-level BLEU score
    """
    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def evaluate_bleu(references, model, source):
    """Generate decoding results and compute BLEU score.
    :param model: NMT Model
    :type model: NMT
    :param references: a list of gold-standard reference target sentences
    :type references: List[List[str]]
    :param source: a list of source sentences
    :type source: List[List[str]]
    :returns bleu_score: corpus-level BLEU score
    """
    with torch.no_grad():
        top_hypotheses = []
        for s in tqdm(source, leave=False):
            hyps = model.beam_search(s, beam_size=16, max_decoding_time_step=(len(s)+10))
            top_hypotheses.append(hyps[0])
    
    s1 = compute_corpus_level_bleu_score(references, top_hypotheses)
    
    return s1

def train_and_evaluate(model, train_data, val_data, optimizer, epochs=10, train_batch_size=32, clip_grad=2, log_every = 100, valid_niter = 500, model_save_path="NMT_model.ckpt"):
    num_trail = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0

    print('Begin Maximum Likelihood training')
    train_time = begin_time = time.time()

    val_data_tgt = [tgt for _, tgt in val_data]
    val_data_src = [src for src, _ in val_data]

    for epoch in tqdm(range(epochs)):
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            
            optimizer.zero_grad()
            
            batch_size = len(src_sents)
            
            example_losses = -model(src_sents, tgt_sents)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size
            loss.backward()
            
            # clip gradient
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()
            
            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val
            
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                        'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                            report_loss / report_examples,
                                                                                            math.exp(report_loss / report_tgt_words),
                                                                                            cum_examples,
                                                                                            report_tgt_words / (time.time() - train_time),
                                                                                            time.time() - begin_time))
                
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.

                

            # perform validation
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                            cum_loss / cum_examples,
                                                                                            np.exp(cum_loss / cum_tgt_words),
                                                                                            cum_examples))
                
                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                print('begin validation ...')

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, val_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl
                
                bleu_score = evaluate_bleu(val_data_tgt, model, val_data_src)*100

                print('validation: iter %d, dev. ppl %f, bleu_score %f' % (train_iter, dev_ppl, bleu_score))

                is_better = len(hist_valid_scores) == 0 or bleu_score > max(hist_valid_scores)
                hist_valid_scores.append(bleu_score)

                if is_better:
                    print('save currently the best model to [%s]' % model_save_path)
                    model.save(model_save_path)


################################################################################
#########################      Prediction Helpers      #########################
################################################################################
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()

def predict(model, input_string):
    """
    Preprocesses an input_string to prepare it for use by the selected model,
    and uses the resulting string to make a prediction (translation).
    """
    nmt_document_preprocessor = lambda x: nltk.word_tokenize(x)
    with torch.no_grad():
        translation = untokenize(model.beam_search(
            nmt_document_preprocessor(input_string),
            beam_size=64,
            max_decoding_time_step=len(input_string)+10
        )[0].value)
    return translation