from memm_model import memm_model
import os
from preprocessor import process, generate_path, sent_process, sent_process_bmweo
from baseline_model import baseline_model
from hmm_model import hmm_model
from gt_ngram import gt_ngram
from ngram import ngram
from hmm_model import hmm_bw_model, hmm_forward_model, hmm_viterbi_model
import csv
import sys
dir_test = os.getcwd() + "/test-public/"
dir_train = os.getcwd() + "/train/"
data_combined = []
for root, dirs, filenames in os.walk(dir_train):
    for i, f in enumerate(filenames):
        # split data into training and test set with train_ratio
        if i > len(filenames) * 0.6:
            break
        data = sent_process_bmweo(root + f)
        data_combined += data

memm = memm_model()
memm.train(data_combined)

data_combined = []
for root, dirs, filenames in os.walk(dir_train):
    start = int(len(filenames) * 0.6 + 1)
    for i in xrange(0, len(filenames)):
        f = filenames[i]
        data = sent_process_bmweo(root + f)
        data_combined += data
for sent in data_combined:
	memm.tag_sentence(sent)