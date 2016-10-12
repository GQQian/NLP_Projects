import os
import random
from nltk import HiddenMarkovModelTagger
from preprocessor import *
from gt_ngram import gt_ngram
from ngram import ngram
####################################################################################################
####################################################################################################
######### Use HiddenMarkovModelTagger class in nltk package to build model #########################
######### Reference: http://www.nltk.org/_modules/nltk/tag/hmm.html#HiddenMarkovModelTagger ########
######### Can write a tranform() to optimize the model #############################################
####################################################################################################
####################################################################################################

class hmm_model(object):
	def __init__(self):
		self.states = set()
		self.symbols = set()
		self.transitions = {}
		self.outputs = {} # key: (symbol, state), value: probability of P(symbol, state|state)


	def train(self, tagged_sentences = None):
		# TODO: why cannot comment parameter?
		# """
		# parameter:
        # tagged_sequences:
        #     type: list of tuples, structure of tuple(TOKEN, POS, TAG)
		# """

		# merge all symbols and states
	    symbol_content, state_content = [], []
	    for sent in tagged_sentences:
	        for token in sent:
	            symbol_content.append(token[0])
	            state_content.append(token[2])

	    # use ngram, gt_ngram to get states, symbols set, self.transitions
	    symbol_ngram = gt_ngram(" ".join(symbol_content))
	    state_ngram = ngram(" ".join(state_content))

	    self.symbols = set(symbol_ngram.ntoken_count(1).keys())
	    self.states = set(state_ngram.ntoken_count(1).keys())

	    self.transitions = state_ngram.generate_ngram(2)

	    # compute self.outputs
	    count_dict = {} # key: tuple(symbol, state),  value: count
	    for sent in tagged_sentences:
	        for token in sent:
	            symbol, state = token[0], token[2]
	            _tuple = (symbol, state)
	            count_dict[_tuple] = count_dict.get(_tuple, 0) + 1

	    for key, val in count_dict.items():
	        self.outputs[key] = 1.0 * val / state_ngram.ncounter_dic[1][tuple(key[1])]


	def tag_sentence(self, untagged_sentence, algorithm = "viterbi"):
		"""
		parameter:
        tagged_sequences:
            type: list of tuples, structure of tuple(TOKEN, POS)
		rtype: list of TAG
		"""

		tag = ['O'] * len(untagged_sentence)

		if algorithm == "viterbi":
		    # 1st word
			token = untagged_sentence[0]
			tuples = [tuple([token[0], state[0]]) for state in self.states]
			_max = 0

			for _tuple in tuples:
				if _tuple in self.outputs and self.outputs[_tuple] > _max:
					_max = self.outputs[_tuple]
					tag[0] = _tuple[1]

			# words after 1st one
			for i in xrange(1, len(untagged_sentence)):
				token = untagged_sentence[i]
				tuples = [tuple([token[0], state[0]]) for state in self.states]
				_max = 0

				for _tuple in tuples:
					if _tuple in self.outputs and tuple([tag[0], _tuple[1]]) in self.transitions and \
					   self.outputs[_tuple] * self.transitions[tuple([tag[0], _tuple[1]])] > _max:
						_max = self.outputs[_tuple] * self.transitions[tuple([tag[0], _tuple[1]])]
						tag[i] = _tuple[1]

		return tag


	def label_phrase(self, untagged_sentence):
		"""
		parameter:
        tagged_sequences:
            type: list of tuples, structure of tuple(TOKEN, POS)

		rtype: list of tuples, tuples contains starting and ending index of predicted cues.
               structure of tuple(starting_idx, ending_idx)
		"""

		tags = self.tag_sentence(untagged_sentence)

		output = []
		left, right = 0, 0
		while left < len(tags):
			if tags[left] == 'W':
				output.append(tuple([left, left]))
				left += 1
			elif tags[left] == 'B':
				right = left + 1
				while right < len(tags) and tags[right] != 'O':
					right += 1
				output.append(tuple([left, right - 1]))
				left = right
			else:
				left += 1

		return output
