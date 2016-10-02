import os
import random

class hmm_model(baseline_model):
	def __init__(self):
		super.(hmm_model, self).__init__()
		self.tag_prob = {}