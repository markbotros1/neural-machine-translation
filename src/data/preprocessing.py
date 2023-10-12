import numpy as np
import gensim
import torch
import nltk

def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
        The paddings should be at the end of each sentence.
    :param sents: list of sentences, where each sentence
                                    is represented as a list of words
    :type sents: list[list[str]]
    :param pad_token: padding token
    :type pad_token: str
    :returns sents_padded: list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentence in the batch now has equal length.
    :rtype: list[list[str]]
    """
    sents_padded = []

    max_len = max([len(sent) for sent in sents])
    sents_padded = [(sent + ([pad_token] * (max_len - len(sent)))) for sent in sents]

    return sents_padded


def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    :param file_path: path to file containing corpus
    :type file_path: str
    :param source: "tgt" or "src" indicating whether text
        is of the source language or target language
    :type source: str
    """
    data = []
    for line in open(file_path):
        sent = nltk.word_tokenize(line)
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>']
        data.append(sent)

    return data


def create_embed_matrix(vocab, sents, vector_size, epochs): ## Mod A
  w2id = vocab.word2id
  emb = np.zeros(shape=(len(w2id), vector_size))
  emb[0] = np.zeros(vector_size)
  emb[1] = np.zeros(vector_size)
  emb[2] = np.zeros(vector_size)
  emb[3] = np.zeros(vector_size)

  model = gensim.models.Word2Vec(sents, vector_size=vector_size, min_count=2, window=5, epochs=epochs)

  for word,id in w2id.items():
    if word in ['<pad>', '<s>', '</s>', '<unk>']:
      continue
    emb[id] = model.wv[word]
  
  return torch.FloatTensor(emb)