import os
from preprocessor import process, generate_path, sent_process, sent_process_biweo
from baseline_model import baseline_model
from hmm_model import hmm_model
from gt_ngram import gt_ngram
from ngram import ngram
import csv

def uncertain_phrase_detection_bm():
    """
    execute uncertain phrase detection and write result to phrase_result.csv, which will be saved
    under current folder
    """
    folder_pub = "test-public"
    folder_pri = "test-private"
    bm = baseline_model()
    dir_pub = generate_path(folder_pub)
    dir_pri = generate_path(folder_pri)
    bm.train()

    csv_f = os.getcwd() + "/" + "phrase_result.csv"

    ##### predicting data in test-public folder #####
    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['Type', 'Spans'])
        writer.writeheader()

        data_combined = []
        for root, dirs, filenames in os.walk(dir_pub):
            for f in filenames:
                data = process(dir_pub + f)
                data_combined += data
        data_combined.pop(0)
        print "# tokens in public folder: {}".format(len(data_combined))
        pub_result = bm.label_phrase(data_combined)

        pub_result_str = ""
        for label in pub_result:
            pub_result_str += str(label[0]) + '-' + str(label[1]) + ' '

        writer.writerow({'Type': "CUE-public", 'Spans': pub_result_str})

    ##### predicting data in test-private folder #######

        data_combined = []
        pri_result_str = ""
        for root, dirs, filenames in os.walk(dir_pri):
            for f in filenames:
                data = process(dir_pri + f)
                data_combined += data
        print "# tokens in private folder: {}".format(len(data_combined))
        pri_result = bm.label_phrase(data_combined)

        for label in pri_result:
            pri_result_str += str(label[0]) + '-' + str(label[1]) + ' '

        writer.writerow({'Type': "CUE-private", 'Spans': pri_result_str})


def uncertain_sent_detection_bm():
    """
    execute uncertain sentence detection and write result to sentence_result.csv, which will be saved
    under current folder
    """
    folder_pub = "test-public"
    folder_pri = "test-private"
    dir_pub = generate_path(folder_pub)
    dir_pri = generate_path(folder_pri)

    bm = baseline_model()
    bm.train()

    csv_f = os.getcwd() + "/" + "sentence_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['Type', 'Indices'])
        writer.writeheader()

        data_combined = []
        for root, dirs, filenames in os.walk(dir_pub):
            for f in filenames:
                data = sent_process(dir_pub + f)
                data_combined += data

        ############ public part ############

        pub_result = []
        data_combined.pop(0)
        for sent in data_combined:
            pub_result.append(bm.label(sent))

        sent_result_pub = []
        for i, sent in enumerate(pub_result):
            for token in sent:
                if token[2] == "CUE":
                    sent_result_pub.append(i)
                    break

        pub_result_str = ""
        for label in sent_result_pub:
            pub_result_str += str(label) + ' '
        writer.writerow({'Type': "SENTENCE-public", 'Indices': pub_result_str})

        ############### Private part ################

        data_combined = []
        pri_result_str = ""
        for root, dirs, filenames in os.walk(dir_pri):
            for f in filenames:
                data = sent_process(dir_pri + f)
                data_combined += data

        pri_result = []
        for sent in data_combined:
            pri_result.append(bm.label(sent))

        sent_result_pri = []
        for i, sent in enumerate(pri_result):
            for token in sent:
                if token[2] == "CUE":
                    sent_result_pri.append(i)
                    break

        pri_result_str = ""
        for label in sent_result_pri:
            pri_result_str += str(label) + ' '

        writer.writerow({'Type': "SENTENCE-private", 'Indices': pri_result_str})


def build_hmm_biweo(train_ratio = 0.8):

    dir_train = os.getcwd() + "/train/"
    data_combined = []
    for root, dirs, filenames in os.walk(dir_train):
        for i, f in enumerate(filenames):
            # split data into training and test set with train_ratio
            if i > len(filenames) * train_ratio:
                break
            data = sent_process_biweo(root + f)
            data_combined += data

    # merge all symbols into an article
    symbol_content, state_content = [], []
    for sent in data_combined:
        for token in sent:
            symbol_content.append(token[0])
            state_content.append(token[2])

    # use ngram, gt_ngram to get states, symboles set, transitions
    symbol_ngram = gt_ngram(" ".join(symbol_content))
    state_ngram = ngram(" ".join(state_content))

    symbols = set(symbol_ngram.ntoken_count(1).keys())
    states = set(state_ngram.ntoken_count(1).keys())

    transitions = state_ngram.generate_ngram(2)


    # compute outputs
    outputs = {} # key: (symbol, state), value: probability of P(symbol, state|state)
    count_dict = {} # key: tuple(symbol, state),  value: count
    for i in xrange(len(symbol_content)):
        symbol, state = symbol_content[i], state_content[i]
        _tuple = (symbol, state)
        count_dict[_tuple] = count_dict.get(_tuple, 0) + 1

    for key, val in count_dict.items():
        outputs[key] = 1.0 * val / state_ngram.ncounter_dic[1][tuple(key[1])]


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
build_hmm_biweo()
