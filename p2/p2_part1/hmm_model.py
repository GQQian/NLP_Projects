import os
import random
from nltk import HiddenMarkovModelTagger
####################################################################################################
####################################################################################################
######### Use HiddenMarkovModelTagger class in nltk package to build model #########################
######### Reference: http://www.nltk.org/_modules/nltk/tag/hmm.html#HiddenMarkovModelTagger ########
######### Can write a tranform() to optimize the model #############################################
####################################################################################################
####################################################################################################

class hmm_model(baseline_model):
	def __init__(self):
		super.(hmm_model, self).__init__()
		self.tag_prob, self. = {}


