from ngram import ngram
import os
import preprocess
from gt_ngram import gt_ngram
import sys
import operator
import csv

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
topics = {'atheism':0, 'autos':1, 'graphics':2, 'medicine':3, 'motorcycles':4, 'religion':5, 'space':6}

def random_sentence_ngram(n = 2, sent_pre = "I have"):
    # TODO: lili write print like first line in ompare_perplexity_ngram()
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        ngrams = ngram(content)
        print "\n\n\nTopic: {}\n".format(topic)
        for k in xrange(1, n + 1):
            print "[{}-gram]\n".format(k)

            print "Empty sentence"
            for i in xrange(3):
                print "[{}]  ".format(i + 1) + ngrams.generate_sentence(k)

            print "\nWith incomplete sentence: " + "\"{}\"".format(sent_pre)
            for i in xrange(3):
                print "[{}]  ".format(i + 1) + ngrams.generate_sentence(k, sent_pre)


def generate_perplexity_gt_ngram():
    print "\n\nTask: generate perplexity with good-turing ngram"

    topics = ['atheism', 'autos', 'graphics', 'medicine', 'motorcycles', 'religion', 'space']
    gt_ngrams = {}
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        gt_ngrams[topic] = gt_ngram(content)

        print "\nTopic: {}".format(topic)
        for i in xrange(1, 5):
            print "[{}-gram]: {}".format(i, gt_ngrams[topic].generate_perplexity(i, content))


def topic_classification_gt_ngram():
    """
    calculate the accuracy for topic classification with different
    n in Good-Turing ngram, then choose the best one to classify files
    in test_for_classification directory, and write results into
    gt_result.csv in classification_task directory
    """

    # get gt_ngram for each topic and read all test data
    gt_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        test_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        if not os.path.isfile(train_f) or not os.path.isfile(test_f):
            split_train_test()

        train_text[topic] = open(train_f, 'r').read()
        test_text[topic] = open(test_f, 'r').read()

        gt_ngrams[topic] = gt_ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in gt_ngram
    for i in xrange(1, 2):
        _sum, correct = 0, 0
        for label_topic, text in test_text.items():
            sentences = text.split('</s>')
            for sentence in sentences:
                sentence += ' </s>'
                min_perp, min_topic = sys.maxint, label_topic

                for topic in topics:
                    perp = gt_ngrams[topic].generate_perplexity(i, sentence)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                if label_topic == min_topic:
                    correct += 1
                _sum += 1

        accuracy[i] = 1.0 * correct / _sum
        print "[{}-gram] accuracy: {}".format(i, accuracy[i])
    #choose the best n
    n = max(accuracy.iteritems(), key = operator.itemgetter(1))[0]

    # get the result for files in test_for_classification directory
    test_dir = indir_pre + "data/classification_task/test_for_classification"
    csv_f = indir_pre + "data/classification_task/gt_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
        writer.writeheader()

        for root, dirs, filenames in os.walk(test_dir):
            for f in filenames:
                text = preprocess.preprocess_file(os.path.join(root, f))
                min_perp, min_topic = sys.maxint, ''

                for topic in topics:
                    perp = gt_ngrams[topic].generate_perplexity(n, text)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


def split_train_test():
    """
    split train_docs into     training:test = 4:1
    store the preprocessed file train.txt and test.txt in each topic directory
    """
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        tokens = content.split()

        # find the nearest </s> after 80% content
        pointer = int(len(tokens) * 0.8)
        while tokens[pointer] != '</s>':
            pointer += 1

        train_text = ' '.join(tokens[:(pointer+1)])
        test_text = ' '.join(tokens[(pointer+2):])

        train_path = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        test_path = indir_pre + "data/classification_task/{}/test.txt".format(topic)
        open(train_path, 'w').write(train_text)
        open(test_path, 'w').write(test_text)


def spell_checker_gt_nrgam():
    pass


def main():
    topic_classification_gt_ngram()


if __name__ == "__main__":
    main()
