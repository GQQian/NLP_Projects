import os
import random
from nltk import HiddenMarkovModelTagger
from preprocessor import *
from gt_ngram import gt_ngram
from ngram import ngram

class hmm_forward_model(object):
	def tag_sentence(self, untagged_sentence):
		"""
		parameter:
        tagged_sequences:
            type: list of tuples, structure of tuple(TOKEN, POS)
		rtype: list of TAG
		"""

		tags = ['O'] * len(untagged_sentence)

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
						if tuple([last_tag, curr_tag]) in self.transitions:
							curr_score[curr_tag] += self.outputs[_tuple] * self.transitions[tuple([last_tag, curr_tag])] * \
													last_score[last_tag]

			tags[i] = max(curr_score.items(), key=lambda x: x[1])[0]

		return tags
