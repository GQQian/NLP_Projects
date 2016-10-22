import os
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
from preprocessor import process, generate_path, sent_process, sent_process_bmweo
import pycrfsuite
import csv

# print(sklearn.__version__)
# input 

class crf_model(object):
    def __init__(self, filename = "trained_model"):
        self.filename = filename
        self.tagger = None

    def train(self, tagged_sentence):
        X_train = [sent2features(s) for s in tagged_sentence]
        y_train = [sent2labels(s) for s in tagged_sentence]
        trainer = pycrfsuite.Trainer(verbose = False)

        # print "{}".format(X_train)
        # print "{}".format(y_train)

        for xseq, yseq in zip(X_train, y_train):
            trainer.append(xseq, yseq)

        # Set training parameters. We will use L-BFGS training algorithm 
        # (it is default) with Elastic Net (L1 + L2) regularization.
        trainer.set_params({
            'c1': 1.0,   # coefficient for L1 penalty
            'c2': 1e-3,  # coefficient for L2 penalty
            'max_iterations': 50,  # stop earlier

            # include transitions that are possible, but not observed
            'feature.possible_transitions': True
        })
        # Possible parameters for the default training algorithm:
        trainer.params()
        trainer.train(self.filename)

        self.tagger = pycrfsuite.Tagger()
        self.tagger.open(self.filename) # here??


    def tag(self, untagged_sentence):
        # print "{}".format(sent2features(untagged_sentence))
        return self.tagger.tag(sent2features(sent2features(untagged_sentence)))


# ## Features
# 
# Next, define some features. In this example we use word identity, word 
# suffix, word shape and word POS tag; also, some information from nearby words is used. 
# 
# This makes a simple baseline, but you certainly can add and remove some features to get 
# (much?) better results - experiment with it.
# print "Test print input: {}".format(test_sents[0])
# print "Test print input: {}".format(train_sents[0])

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
    ]
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('BOS')
        
    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
        ])
    else:
        features.append('EOS')
     
    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


# This is what word2features extracts:

# In[7]:

# sent2features(train_sents[0])[0]


# Extract the features from the data:

# In[8]:

# X_train = [sent2features(s) for s in train_sents]
# y_train = [sent2labels(s) for s in train_sents]

# X_test = [sent2features(s) for s in test_sents]
# y_test = [sent2labels(s) for s in test_sents]


# ## Train the model
# 
# To train the model, we create pycrfsuite.Trainer, load the training data and call 'train' method. 
# First, create pycrfsuite.Trainer and load the training data to CRFsuite:

# In[9]:

# trainer = pycrfsuite.Trainer(verbose=False)

# for xseq, yseq in zip(X_train, y_train):
#     trainer.append(xseq, yseq)


# Set training parameters. We will use L-BFGS training algorithm (it is default) with Elastic Net 
# (L1 + L2) regularization.

# In[10]:



# trainer.set_params({
#     'c1': 1.0,   # coefficient for L1 penalty
#     'c2': 1e-3,  # coeff9icient for L2 penalty
#     'max_iterations': 50,  # stop earlier

#     # include transitions that are possible, but not observed
#     'feature.possible_transitions': True
# })
# # Possible parameters for the default training algorithm:

# We can also get information about the final state of the model by looking at the 
# trainer's logparser. If we had tagged our input data using the optional group argument 
# in add, and had used the optional holdout argument during train, there would be 
# information about the trainer's performance on the holdout set as well. 

# In[ ]:

# print("Info about final state: %s\n\n" % trainer.logparser.last_iteration)


# We can also get this information for every step using trainer.logparser.iterations


# print "Length of last iteration: {}\n\n Info about last iteration: 
# {}".format(len(trainer.logparser.iterations), trainer.logparser.iterations[-1])


# ## Make predictions
# 
# To use the trained model, create pycrfsuite.Tagger, open the model and use "tag" method:

# In[ ]:

# tagger = pycrfsuite.Tagger()
# tagger.open('training data')


# Let's tag a sentence to see how it works:

# In[ ]:
####################################
########## writing output ##########
####################################

# ## Evaluate the model

# In[ ]:

def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'O'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )

# Predict entity labels for all sentences in our testing set ('testb' Spanish data):

# y_pred = [tagger.tag(xseq) for xseq in X_test]


# ..and check the result. Note this report is not comparable to results in CONLL2002 
# papers because here we check per-token results (not per-entity). Per-entity numbers will be worse.  

# print(bio_classification_report(y_test, y_pred))


# ## Let's check what classifier learned

# from collections import Counter
# info = tagger.info()

# def print_transitions(trans_features):
#     for (label_from, label_to), weight in trans_features:
#         print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

# print("Top likely transitions:")
# print_transitions(Counter(info.transitions).most_common(15))

# print("\nTop unlikely transitions:")
# print_transitions(Counter(info.transitions).most_common()[-15:])


# We can see that, for example, it is very likely that the beginning of an organization name (B-ORG) 
# will be followed by a token inside organization name (I-ORG), but transitions to I-ORG from tokens 
# with other labels are penalized. Also note I-PER -> B-LOC transition: a positive weight means that 
# model thinks that a person name is often followed by a location.
# 
# Check the state features:

# In[ ]:

# def print_state_features(state_features):
#     for (attr, label), weight in state_features:
#         print("%0.6f %-6s %s" % (weight, label, attr))    

# print("Top positive:")
# print_state_features(Counter(info.state_features).most_common(20))

# print("\nTop negative:")
# print_state_features(Counter(info.state_features).most_common()[-20:])

# Some observations:
# 
# * **8.743642 B-ORG  word.lower=psoe-progresistas** - the model remembered names of some entities - maybe 
# it is overfit, or maybe our features are not adequate, or maybe remembering is indeed helpful;
# * **5.195429 I-LOC  -1:word.lower=calle**: "calle" is a street in Spanish; model learns that if a previous 
# word was "calle" then the token is likely a part of location;
# * **-3.529449 O      word.isupper=True**, ** -2.913103 O      word.istitle=True **: UPPERCASED or TitleCased 
# words are likely entities of some kind;
# * **-2.585756 O      postag=NP** - proper nouns (NP is a proper noun in the Spanish tagset) are often entities.

# ## What to do next
# 
# 1. Load 'testa' Spanish data.
# 2. Use it to develop better features and to find best model parameters.
# 3. Apply the model to 'testb' data again.
# 
# The model in this notebook is just a starting point; you certainly can do better!
