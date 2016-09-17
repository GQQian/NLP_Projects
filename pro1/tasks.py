from ngram import ngram
import os
import preprocess
from gt_ngram import gt_ngram

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
topics = ['atheism', 'autos', 'graphics', 'medicine', 'motorcycles', 'religion', 'space']

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
    gt_ngrams = {}
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        gt_ngrams[topic] = gt_ngram(content)



    test_dir = indir_pre + "data/classification_task/test_for_classification"
    test_text = preprocess.preprocess_dir(test_dir)


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
    split_train_test()


if __name__ == "__main__":
    main()
