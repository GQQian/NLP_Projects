import os
import random
from nltk import HiddenMarkovModelTagger
from preprocessor import *
from gt_ngram import gt_ngram
from ngram import ngram
from hmm_model import hmm_model
from math import log

class hmm_bw_model(hmm_model):
	def __init__(self):
		self.states = set()
		self.symbols = set()
		self.transitions_f, self.transitions_b = {}
		self.outputs = {} # key: (symbol, state), value: probability of P(symbol, state|state)


	def train(self, tagged_sentences = None):
		# merge all symbols and states
	    symbol_content, state_content = [], []
	    for sent in tagged_sentences:
	        for token in sent:
	            symbol_content.append(token[0])
	            state_content.append(token[2])

	    # use ngram, gt_ngram to get states, symbols set, self.transitions
	    symbol_ngram_f = gt_ngram(" ".join(symbol_content))
		symbol_ngram_b = gt_ngram(" ".join(symbol_content[::-1]))

	    state_ngram = ngram(" ".join(state_content))

	    self.symbols = set(symbol_ngram.ntoken_count(1).keys())
	    self.states = set(state_ngram.ntoken_count(1).keys())

	    self.transitions_f = symbol_ngram_f.generate_ngram(2)
		self.transitions_b = symbol_ngram_b.generate_ngram(2)

	    # compute self.outputs
	    count_dict = {} # key: tuple(symbol, state),  value: count
	    for sent in tagged_sentences:
	        for token in sent:
	            symbol, state = token[0], token[2]
	            _tuple = (symbol, state)
	            count_dict[_tuple] = count_dict.get(_tuple, 0) + 1

	    for key, val in count_dict.items():
	        self.outputs[key] = 1.0 * val / state_ngram.ncounter_dic[1][tuple(key[1])]


	def tag_sentence(self, untagged_sentence, algorithm = ""):
		"""
		parameter:
        tagged_sequences:
            type: list of tuples, structure of tuple(TOKEN, POS)
		rtype: list of TAG
		"""

		tags = ['O'] * len(untagged_sentence)

		def get_scores(untagged_sentence, transitions):
			# tag's score, key: state, value: score
			curr_score = {'O': 0}

			# 1st word
			token = untagged_sentence[0]
			tuples = [tuple([token[0], state[0]]) for state in self.states]

			for _tuple in tuples:
				if _tuple in self.outputs:
					curr_score[_tuple[1]] = self.outputs[_tuple]

			tags[0] = max(curr_score.items(), key=lambda x: x[1])[0]

			# words after 1st one
			for i in xrange(1, len(untagged_sentence)):
				last_score, curr_score = curr_score, {'O': 0}
				token = untagged_sentence[i]
				tuples = [tuple([token[0], state[0]]) for state in self.states]

				for _tuple in tuples:
					if _tuple in self.outputs:
						curr_tag, curr_score[curr_tag] = _tuple[1], 0
						for last_tag in last_score:
							if tuple([last_tag, curr_tag]) in transitions:
								curr_score[curr_tag] += self.outputs[_tuple] * transitions[tuple([last_tag, curr_tag])] * \
														last_score[last_tag]

				tags[i] = max(curr_score.items(), key=lambda x: x[1])[0]

		scores_f = get_scores(untagged_sentence, transitions_f)
		scores_b = get_scores(reversed(untagged_sentence), transitions_b)

		_len = len(scores_f)
		tags = ['O'] * _len
		for i in xrange(_len):
			_max = 0, 'O'
			for tag in scores_f[i]:
				if tag in scores_b[-i] and scores_b[-i][tag] * scores_f[i][tag] > _max:
					_max = scores_b[-i][tag] * scores_f[i][tag]
					tag[i] = tag

		return tags
