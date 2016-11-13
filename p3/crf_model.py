import os
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
from preprocessor import process, generate_path, sent_process, sent_process_bmweo
import pycrfsuite
import csv


dir_train = os.getcwd() + "/train/"
train_sents, test_sents = [], []
train_ratio = .8

for root, dirs, filenames in os.walk(dir_train):
    for i, f in enumerate(filenames):
        # split data into training and test set with train_ratio
        if i > len(filenames) * train_ratio:
            data = sent_process_bmweo(root + f)
            test_sents += data
        data = sent_process_bmweo(root + f)
        train_sents += data

# ## Features

print "Test print input: {}".format(test_sents[0])
print "Test print input: {}".format(train_sents[0])

# i is the index of the sent in the list of sentences, sent 
# ADD: more features: considering its tense, for word w, put w[-2] as features
# ADD: 
# REMOVE: some features not important here 
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:], # related to some third person singular
        'word[-2:]=' + word[-2:], # related to tense
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
    return [label for token, postag, label in sent] # need to be altered

def sent2tokens(sent):
    return [token for token, postag, label in sent]    


# Extract the features from the data:


X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]

X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]


# ## Train the model

trainer = pycrfsuite.Trainer(verbose=False)

for xseq, yseq in zip(X_train, y_train):
    trainer.append(xseq, yseq)


# Set training parameters. We will use L-BFGS training algorithm (it is default) with Elastic Net (L1 + L2) regularization.

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coeff9icient for L2 penalty
    'max_iterations': 50,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})
# Possible parameters for the default training algorithm:
trainer.params()
trainer.train("training data")


tagger = pycrfsuite.Tagger()
tagger.open('training data')


# Let's tag a sentence to see how it works:


####################################
########## writing output ##########
####################################

def get_detection_results(tagger, type):
    
    _dir = generate_path("test-{}".format(type))
    data_combined = []
    for root, dirs, filenames in os.walk(_dir):
        for f in filenames:
            data = sent_process(_dir + f)
            data_combined += data

    phrase_ret, sent_ret = [], []
    phrase_index = 0
    sums = 0
    for sent_index, sent in enumerate(data_combined):
        labels = []
        left, right = 0, 0
        tags = tagger.tag(sent2features(sent))
        sums += len(tags)
        print "tags: {}".format(tags)
        while left < len(sent):
            if tags[left] == 'W':
                labels.append(tuple([left, left]))
                left += 1
            elif tags[left] == 'B':
                right = left + 1
                while right < len(sent) and tags[right] != 'O':
                    right += 1
                labels.append(tuple([left, right - 1]))
                left = right
            else:
                left += 1

        if len(labels) > 0:
            sent_ret.append(str(sent_index))
            for label in labels:
                phrase_ret.append("{}-{}".format(label[0] + phrase_index, label[1] + phrase_index))
        phrase_index += len(sent)

    return (" ".join(phrase_ret), " ".join(sent_ret))

# get and write results into csv
public_ret, private_ret = get_detection_results(tagger, "public"), get_detection_results(tagger, "private")

phrase_f = os.getcwd() + "/" + "crf_phrase_result.csv"
sent_f = os.getcwd() + "/" + "crf_sentence_result.csv"
print "public_ret phrase: {}".format(len(public_ret[0]))
print "public_ret sentence: {}".format(len(public_ret[1]))
with open(phrase_f, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ['Type', 'Spans'])
    writer.writeheader()
    writer.writerow({'Type': "CUE-public", 'Spans': public_ret[0]})
    writer.writerow({'Type': "CUE-private", 'Spans': private_ret[0]})

with open(sent_f, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = ['Type', 'Indices'])
    writer.writeheader()
    writer.writerow({'Type': "SENTENCE-public", 'Indices': public_ret[1]})
    writer.writerow({'Type': "SENTENCE-private", 'Indices': private_ret[1]})
# ## Evaluate the model


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

y_pred = [tagger.tag(xseq) for xseq in X_test]


# ..and check the result. Note this report is not comparable to results in CONLL2002 
# papers because here we check per-token results (not per-entity). Per-entity numbers will be worse.  

print(bio_classification_report(y_test, y_pred))


# ## Output the result of trained model

from collections import Counter
info = tagger.info()

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

print("Top likely transitions:")
print_transitions(Counter(info.transitions).most_common(15))

print("\nTop unlikely transitions:")
print_transitions(Counter(info.transitions).most_common()[-15:])


# We can see that, for example, it is very likely that the beginning of an organization name (B-ORG) will be followed by a token inside organization name (I-ORG), but transitions to I-ORG from tokens with other labels are penalized. Also note I-PER -> B-LOC transition: a positive weight means that model thinks that a person name is often followed by a location.
# 
# Check the state features:



def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-6s %s" % (weight, label, attr))    

print("Top positive:")
print_state_features(Counter(info.state_features).most_common(20))

print("\nTop negative:")
print_state_features(Counter(info.state_features).most_common()[-20:])
