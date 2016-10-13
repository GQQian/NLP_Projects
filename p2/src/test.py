import os
from preprocessor import process, generate_path, sent_process, sent_process_biweo
from baseline_model import baseline_model
from hmm_model import hmm_model
from gt_ngram import gt_ngram
from ngram import ngram
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


def uncertain_detection_hmm(train_ratio = 0.8, model = hmm_bw_model):
    ############ use training data to train hmm model ############
    dir_train = os.getcwd() + "/train/"
    data_combined = []
    for root, dirs, filenames in os.walk(dir_train):
        for i, f in enumerate(filenames):
            # split data into training and test set with train_ratio
            if i > len(filenames) * train_ratio:
                break
            data = sent_process_biweo(root + f)
            data_combined += data

    hmm = model()
    hmm.train(data_combined)

    ############ use test data to get accuracy ############
    data_combined = []
    for root, dirs, filenames in os.walk(dir_train):
        start = int(len(filenames) * train_ratio + 1)
        for i in xrange(start, len(filenames)):
            f = filenames[i]
            data = sent_process_biweo(root + f)
            data_combined += data

    correct, _sum = 0, 0
    for sent in data_combined:
        tags = hmm.tag_sentence(sent)
        for i, tag in enumerate(tags):
            if sent[i][2] != 'O' :
                _sum += 1
                correct += 1 if sent[i][2] == tag else 0

    print 1.0 * correct / _sum


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
            labels = hmm.label_phrase(sent)
            if len(labels) > 0:
                sent_ret.append(str(sent_index))
                for label in labels:
                    phrase_ret.append("{}-{}".format(label[0] + phrase_index, label[1] + phrase_index))
            phrase_index += len(sent)

        return (" ".join(phrase_ret), " ".join(sent_ret))

    # get and write results into csv
    public_ret, private_ret = get_detection_results("public"), get_detection_results("private")

    phrase_f = os.getcwd() + "/" + "hmm_phrase_result.csv"
    sent_f = os.getcwd() + "/" + "hmm_sentence_result.csv"

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
