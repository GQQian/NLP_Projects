#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from preprocess import preprocess_text
import re
import copy
import sys
os.system("pip install --upgrade gensim")
from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
indir_pre = os.getcwd() + "/"
topics = {'atheism':0, 'autos':1, 'graphics':2, 'medicine':3, 'motorcycles':4, 'religion':5, 'space':6}


"""
ratio = training ratio = 0.7
training the model based on the frist 70% of  training text
examine accuracy using the last 30% of training text
"""

ratio = 0.7

def model_train(min_count = 1, size = 3, window = 2):
	
	sentences = ""
	text = ""

	for topic in topics:
		test_dir = indir_pre + "data/spell_checking_task/{}/train_docs".format(topic)
		for root, dirs, filenames in os.walk(test_dir):
			for idx, f in enumerate(filenames):
				if idx < len(filenames) * ratio:
					continue
				text += open(os.path.join(root, f),'r').read()
	
	sentences = preprocess_text(text,"weTrain")
	model = word2vec.Word2Vec(sentences, size = size, min_count = min_count, window = window)

	return model


"""
load the pre-trained model for google word2vec
"""

def model_pretrained():
	os.system("wget https://s3.amazonaws.com/mordecai-geo/GoogleNews-vectors-negative300.bin.gz")
	
	model = word2vec.Word2Vec.load_word2vec_format(os.getcwd() + "/GoogleNews-vectors-negative300.bin.gz", binary=True)
	

	return model


def spellcheck_we(method = "pretrained"):

	"""
	find out which word makes more sense in the context
	return: a word in wordList
	"""
	def word_cluster_value(wordList, context, method = "pretrained"):

		if method == "train":
			similarities = []
			maxSim = -sys.maxint
			output = ""
			context_tokens = context.split()
			for word in wordList:
				for token in context_tokens:
					try:
						similarities.append(model.similarity(word, token))
					except KeyError:
						try:
							similarities.append(model.similarity('unk', token))
						except KeyError:
							try:similarities.append(model.similarity(word, 'unk'))
							except KeyError:
								similarities.append(model.similarity('unk', 'unk'))
				normalized_sim = sum(similarities)/len(similarities)
				if(normalized_sim > maxSim):
					output = word
					maxSim = normalized_sim
		else:
			similarities = []
			maxSim = -sys.maxint
			output = ""
			context_tokens = context.split()
			for word in wordList:
				for token in context_tokens:
					try:
						similarities.append(model.similarity(word, token))
					except KeyError:
						try:
							similarities.append(model1.similarity('unk', token))
						except KeyError:
							try:similarities.append(model1.similarity(word, 'unk'))
							except KeyError:
								similarities.append(model1.similarity('unk', 'unk'))
				normalized_sim = sum(similarities)/len(similarities)
				if(normalized_sim > maxSim):
					output = word
					maxSim = normalized_sim
		return output



	f = indir_pre + "data/spell_checking_task/confusion_set.txt"
	lines = [line.rstrip('\n\r') for line in open(f)]
	# key: word   value: a set of confused words
	confusion_dic = {}
	for line in lines:
		if "may" in line:
			continue
		words = line.split()
		for word in words:
			if word in confusion_dic:
				for word1 in confusion_dic[word]:
					confusion_dic[word1] = confusion_dic[word].union(set(words))
			else:
				confusion_dic[word] = set(words)

	confusion_dic['may'] = set(['may', 'may be'])

	_sum, correct = 1, 0
	for topic in topics:
		test_dir = indir_pre + "data/spell_checking_task/{}/train_docs".format(topic)

		for root, dirs, filenames in os.walk(test_dir):
			for idx, f in enumerate(filenames):
				if idx > len(filenames) * ratio:
					continue

				

				text = open(os.path.join(root, f),'r').read()
				text = preprocess_text(text, "sentences")
				for sent in text:
					j = 0
					k = 0
					newtext = sent
					while j < len(sent)-1:
						if newtext[k].isalpha() == False and newtext[k] != ' ':
							newtext = newtext[:k] + newtext[k+1:]
							j = j + 1
						else:
							k = k + 1
							j = j + 1
					sent_process = newtext
					sent_process = sent_process.split()
					for word_idx, token in enumerate(sent_process):
						if token in confusion_dic:
							min_word = token
							context = sent_process.pop(word_idx)
							min_word = word_cluster_value(confusion_dic[token], context, method)
				
							_sum += 1
							if min_word == token:
								correct += 1


	accuracy = 100.0 * correct / _sum

	print (method + " accuracy = " + " %.2f" %accuracy + "%")



"""
running instructions:
two types of tasks: "train" and "pretrained"

Edit the task string to swtich tasks

"pretrained" uses the pre-trained GoogleNews model
"train" will train the model using the data for project 1

3 arguments can be used for training the model: min_count, size, window
min_count : words occurred less than min_count times will be neglected in the model
size: the degree of freedom (# of dimensions) of the normalized vector space
window: the maximum distance between current and predicted word within a sentence



"""

task = "train"
minimum_count = 5
size_ = 100
window_ = 5

if(task == "train"):
	model = model_train(min_count = minimum_count, size = size_, window = window_)
else:
	model = model_pretrained()
	model1 = model_train(min_count = minimum_count, size = size_, window = window_)


spellcheck_we(task)
print("min_count = %.i" %minimum_count + " size = %.i" %size_ + " window = %.i" %window_)











