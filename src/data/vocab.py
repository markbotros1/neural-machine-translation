import torch
from typing import List
from collections import Counter
from itertools import chain
from .preprocessing import pad_sents

class Vocab(object):
    """ Vocabulary, i.e. structure containing either
    src or tgt language terms.
    """
    def __init__(self, word2id=None):
        """ Init Vocab Instance.
        
        :param word2id: dictionary mapping words 2 indices
        :type word2id: dict[str, int]
        """
        if word2id:
            self.word2id = word2id
        else:
            self.word2id = dict()
            self.word2id['<pad>'] = 0   # Pad Token
            self.word2id['<s>'] = 1     # Start Token
            self.word2id['</s>'] = 2    # End Token
            self.word2id['<unk>'] = 3   # Unknown Token
        self.unk_id = self.word2id['<unk>']
        self.id2word = {v: k for k, v in self.word2id.items()}

    def __getitem__(self, word):
        """ Retrieve word's index. Return the index for the unk
        token if the word is out of vocabulary.
        
        :param word: word to look up
        :type word: str
        :returns: index of word
        :rtype: int
        """
        return self.word2id.get(word, self.unk_id)

    def __contains__(self, word):
        """ Check if word is captured by Vocab.
        
        :param word: word to look up
        :type word: str
        :returns: whether word is in vocab
        :rtype: bool
        """
        return word in self.word2id

    def __setitem__(self, key, value):
        """ Raise error, if one tries to edit the Vocab directly.
        """
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        """ Compute number of words in Vocab.
        
        :returns: number of words in Vocab
        :rtype: int
        """
        return len(self.word2id)

    def __repr__(self):
        """ Representation of Vocab to be used
        when printing the object.
        """
        return 'Vocabulary[size=%d]' % len(self)

    def id2word(self, wid):
        """ Return mapping of index to word.
        
        :param wid: word index
        :type wid: int
        :returns: word corresponding to index
        :rtype: str
        """
        return self.id2word[wid]

    def add(self, word):
        """ Add word to Vocab, if it is previously unseen.
        
        :param word: to add to Vocab
        :type word: str
        :returns: index that the word has been assigned
        :rtype: int
        """
        if word not in self:
            wid = self.word2id[word] = len(self)
            self.id2word[wid] = word
            return wid
        else:
            return self[word]

    def words2indices(self, sents):
        """ Convert list of words or list of sentences of words
        into list or list of list of indices.
        
        :param sents: sentence(s) in words
        :type sents: Union[List[str], List[List[str]]]
        :returns: sentence(s) in indices
        :rtype: Union[List[int], List[List[int]]]
        """
        if type(sents[0]) == list:
            return [[self[w] for w in s] for s in sents]
        else:
            return [self[w] for w in sents]

    def indices2words(self, word_ids):
        """ Convert list of indices into words.
        
        :param word_ids: list of word ids
        :type word_ids: List[int]
        :returns: list of words
        :rtype: List[Str]
        """
        return [self.id2word[w_id] for w_id in word_ids]

    def to_input_tensor(self, sents: List[List[str]], device: torch.device) -> torch.Tensor:
        """ Convert list of sentences (words) into tensor with necessary padding for 
        shorter sentences.
        
        :param sents: list of sentences (words)
        :type sents: List[List[str]]
        :param device: Device on which to load the tensor, ie. CPU or GPU
        :type device: torch.device
        :returns: Sentence tensor of (max_sentence_length, batch_size)
        :rtype: torch.Tensor
        """

        word_ids = self.words2indices(sents)
        sents_t = pad_sents(word_ids, self['<pad>'])
        sents_var = torch.tensor(sents_t, dtype=torch.long, device=device)
        return torch.t(sents_var)

    @staticmethod
    def from_corpus(corpus, size, freq_cutoff=2):
        """ Given a corpus construct a Vocab.
        
        :param corpus: corpus of text produced by read_corpus function
        :type corpus: List[str]
        :param size: # of words in vocabulary
        :type size: int
        :param freq_cutoff: if word occurs n < freq_cutoff times, drop the word
        :type freq_cutoff: int
        :returns: Vocab instance produced from provided corpus
        :rtype: Vocab
        """
        vocab_entry = Vocab()
        word_freq = Counter(chain(*corpus))
        valid_words = [w for w, v in word_freq.items() if v >= freq_cutoff]
        top_k_words = sorted(valid_words, key=lambda w: word_freq[w], reverse=True)[:size]
        for word in top_k_words:
            vocab_entry.add(word)
        return vocab_entry
    
    @staticmethod
    def from_subword_list(subword_list):
        """Given a list of subwords, construct the Vocab.
        
        :param subword_list: list of subwords in corpus
        :type subword_list: List[str]
        :returns: Vocab instance produced from provided list
        :rtype: Vocab
        """
        vocab_entry = Vocab()
        for subword in subword_list:
            vocab_entry.add(subword)
        return vocab_entry