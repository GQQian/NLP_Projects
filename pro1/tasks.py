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


def topic_classification_gt_ngram():
    gt_ngrams = {}
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        gt_ngrams[topic] = gt_ngram(content)


    test_dir = indir_pre + "data/classification_task/test_for_classification"
    test_text = preprocess.preprocess_dir(test_dir)



def spell_checker():
    pass


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


def main():
    random_sentence_ngram()
    generate_perplexity_gt_ngram()



if __name__ == "__main__":
    main()
