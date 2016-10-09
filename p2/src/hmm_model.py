import os
import random
from nltk import HiddenMarkovModelTagger
from preprocessor import *
####################################################################################################
####################################################################################################
######### Use HiddenMarkovModelTagger class in nltk package to build model #########################
######### Reference: http://www.nltk.org/_modules/nltk/tag/hmm.html#HiddenMarkovModelTagger ########
######### Can write a tranform() to optimize the model #############################################
####################################################################################################
####################################################################################################

class hmm_model(HiddenMarkovModelTagger):
	def __init__():
		"""
		:param symbols: the set of output symbols (alphabet)
	    :type symbols: seq of any
	    :param states: a set of states representing state space
	    :type states: seq of any
	    :param transitions: transition probabilities; Pr(s_i | s_j) is the
	        probability of transition from state i given the model is in
	        state_j
	    :type transitions: ConditionalProbDistI
	    :param outputs: output probabilities; Pr(o_k | s_i) is the probability
	        of emitting symbol k when entering state i
	    :type outputs: ConditionalProbDistI
	    :param priors: initial state distribution; Pr(s_i) is the probability
	        of starting in state i
	    :type priors: ProbDistI
	    :param transform: an optional function for transforming training
	        instances, defaults to the identity function.
	    :type transform: callable
		"""
		pass
