import os
from preprocessor import process, generate_path, sent_process, sent_process_bmweo, sent_process_bio
from baseline_model import baseline_model
from hmm_model import hmm_model
from gt_ngram import gt_ngram
from ngram import ngram
from crf_model import crf_model
from hmm_model import hmm_bw_model, hmm_forward_model, hmm_viterbi_model
import csv
import sys


def uncertain_detection_bm():

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

        csv_f = os.getcwd() + "/" + "bm_phrase_result.csv"

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
            pub_result = bm.label_phrase_untagged(data_combined)

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
            pri_result = bm.label_phrase_untagged(data_combined)

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

        csv_f = os.getcwd() + "/" + "bm_sentence_result.csv"

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

    uncertain_phrase_detection_bm()
    uncertain_sent_detection_bm()


def uncertain_detection_hmm(train_ratio = 0.8, model = hmm_forward_model):
    ############ use training data to train hmm model ############
    dir_train = os.getcwd() + "/train/"
    data_combined = []
    for root, dirs, filenames in os.walk(dir_train):
        for i, f in enumerate(filenames):
            # split data into training and test set with train_ratio
            if i > len(filenames) * train_ratio:
                break
            data = sent_process_bio(root + f)
            data_combined += data

    hmm = model()
    hmm.train(data_combined)


    ############ use test data to get Mean F Score ############
    data_combined = []
    for root, dirs, filenames in os.walk(dir_train):
        start = int(len(filenames) * train_ratio + 1)
        for i in xrange(start, len(filenames)):
            f = filenames[i]
            data = sent_process_bio(root + f)
            data_combined += data

    # Precision p is the ratio of true positives tp to all predicted positives tp + fp.
    # Recall r is the ratio of true positives to all actual positives tp + fn.
    # The F score is given by
    # F = 2pr/(p+r)
    # p = tp/(tp+fp)   r = tp/(tp+fn)

    [phrase_p_cum, phrase_r_cum, phrase_tp, sent_p_cum, sent_r_cum, sent_tp] = [0] * 6
    for sent in data_combined:
        correct_tags = [token[2] for token in sent]
        correct_labels = hmm.label_phrase_tagged(correct_tags)
        correct_sent = False if len(correct_labels) == 0 else True

        predict_labels = hmm.label_phrase_untagged(sent)
        predict_sent = False if len(predict_labels) == 0 else True

        phrase_p_cum += len(predict_labels) # tp+fp
        phrase_r_cum += len(correct_labels) # tp+fn
        sent_p_cum += 1 # tp+fp
        sent_r_cum += 1 # tp+fn

        intersect = [label for label in correct_labels if label in predict_labels]
        phrase_tp += len(intersect)
        sent_tp += predict_sent == correct_sent

    phrase_p, phrase_r = 1.0 * phrase_tp / phrase_p_cum, 1.0 * phrase_tp / phrase_r_cum
    sent_p, sent_r = 1.0 * sent_tp / sent_p_cum, 1.0 * sent_tp / sent_r_cum

    phrase_f = 1.0 * 2 * (phrase_p * phrase_r) / (phrase_p + phrase_r)
    sent_f = 1.0 * 2 * (sent_p * sent_r) / (sent_p + sent_r)

    print "Phrase F score: {}".format(phrase_f)
    print "Sentence F score: {}".format(sent_f)

    ############ prase and sent detection ############
    def get_detection_results(type):
        _dir = generate_path("test-{}".format(type))
        data_combined = []
        for root, dirs, filenames in os.walk(_dir):
            for f in filenames:
                data = sent_process(_dir + f)
                data_combined += data

        phrase_ret, sent_ret = [], []
        phrase_index = 0
        for sent_index, sent in enumerate(data_combined):
            labels = hmm.label_phrase_untagged(sent)
            if len(labels) > 0:
                sent_ret.append(str(sent_index))
                for label in labels:
                    phrase_ret.append("{}-{}".format(label[0] + phrase_index, label[1] + phrase_index))
            phrase_index += len(sent)

        return (" ".join(phrase_ret), " ".join(sent_ret))

    # get and write results into csv
    public_ret, private_ret = get_detection_results("public"), get_detection_results("private")

    phrase_f = os.getcwd() + "/" + "hmm_phrase_result_1.csv"
    sent_f = os.getcwd() + "/" + "hmm_sentence_result_1.csv"

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


def uncertain_detection_crf():
    dir_train = os.getcwd() + "/train/"
    train_ratio = .8
    train_sents, test_sents = [], []
    for root, dirs, filenames in os.walk(dir_train):
        for i, f in enumerate(filenames):
            # split data into training and test set with train_ratio
            if i > len(filenames) * train_ratio:
                data = sent_process_bmweo(root + f)
                test_sents += data
            else:
                data = sent_process_bmweo(root + f)
                train_sents += data

    crf = crf_model()
    crf.train(train_sents)

    ############ use test data to get Mean F Score ############
    data_combined = []
    for root, dirs, filenames in os.walk(dir_train):
        start = int(len(filenames) * train_ratio + 1)
        for i in xrange(start, len(filenames)):
            f = filenames[i]
            data = sent_process_bmweo(root + f)
            data_combined += data

    # Precision p is the ratio of true positives tp to all predicted positives tp + fp.
    # Recall r is the ratio of true positives to all actual positives tp + fn.
    # The F score is given by
    # F = 2pr/(p+r)
    # p = tp/(tp+fp)   r = tp/(tp+fn)

    [phrase_p_cum, phrase_r_cum, phrase_tp, sent_p_cum, sent_r_cum, sent_tp] = [0] * 6
    for sent in data_combined:
        correct_tags = [token[2] for token in sent]
        correct_labels = crf.tag(sent)
        correct_sent = False if len(correct_labels) == 0 else True

        predict_labels = crf.tag(sent)
        # print "{}".format(predict_labels)
        predict_sent = False if len(predict_labels) == 0 else True

        phrase_p_cum += len(predict_labels) # tp+fp
        phrase_r_cum += len(correct_labels) # tp+fn
        sent_p_cum += 1 # tp+fp
        sent_r_cum += 1 # tp+fn

        intersect = [label for label in correct_labels if label in predict_labels]
        phrase_tp += len(intersect)
        sent_tp += predict_sent == correct_sent

    phrase_p, phrase_r = 1.0 * phrase_tp / phrase_p_cum, 1.0 * phrase_tp / phrase_r_cum
    sent_p, sent_r = 1.0 * sent_tp / sent_p_cum, 1.0 * sent_tp / sent_r_cum

    phrase_f = 1.0 * 2 * (phrase_p * phrase_r) / (phrase_p + phrase_r)
    sent_f = 1.0 * 2 * (sent_p * sent_r) / (sent_p + sent_r)

    print "Phrase F score: {}".format(phrase_f)
    print "Sentence F score: {}".format(sent_f)


    def get_detection_results(type):
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
            tags = crf.tag(sent)
            sums += len(tags)
            # print "tags: {}".format(tags)
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

    public_ret, private_ret = get_detection_results("public"), get_detection_results("private")

    phrase_f = os.getcwd() + "/" + "crf_phrase_result.csv"
    sent_f = os.getcwd() + "/" + "crf_sentence_result.csv"
    # print "public_ret phrase: {}".format(len(public_ret[0]))
    # print "public_ret sentence: {}".format(len(public_ret[1]))
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


def get_cues():
    """
    return all cues in training data
    """
    cues = {}

    dir_train = os.getcwd() + "/train/"
    data_combined = []
    for root, dirs, filenames in os.walk(dir_train):
        for i, f in enumerate(filenames):
            data = sent_process(root + f)
            data_combined += data

    for sent in data_combined:
        left = 0
        while left < len(sent):
            curr_tag = sent[left][2]
            if curr_tag == '_':
                left += 1
                continue
            else:
                right = left + 1

            while right < len(sent) and sent[right][2] == curr_tag:
                right += 1

            cue = tuple(sent[i][0] for i in xrange(left, right))
            cues[cue] = cues.get(cue, 0) + 1

            left = right
    return cues
