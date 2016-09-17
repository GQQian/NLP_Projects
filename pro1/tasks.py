from ngram import ngram
import os
import preprocess
from gt_ngram import gt_ngram

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
topics = ['atheism', 'autos', 'graphics', 'medicine', 'motorcycles', 'religion', 'space']

def random_sentence_ngram(n = 2, sent_pre = "I have", topics = ["autos"]):
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


def topic_classification():
    pass


def spell_checker():
    pass


def compare_perplexity_ngram():
    print "[task]: compare perplexity for different n for good turing ngram model"

    indir = indir_pre + "data/classification_task/atheism/train_docs"
    test_f = indir_pre + "data/classification_task/test_for_classification/file_0.txt"

    content = preprocess.preprocess_dir(indir)
    atheism = gt_ngram(content)

    sentences = preprocess.preprocess_file(test_f)
    print atheism.generate_perplexity(1, sentences)
    print atheism.generate_perplexity(2, sentences)
    print atheism.generate_perplexity(3, sentences)
    print atheism.generate_perplexity(5, sentences)
    print atheism.generate_perplexity(6, sentences)
    print atheism.generate_perplexity(7, sentences)


def main():
    random_sentence_ngram(topics=topics)
    compare_perplexity_ngram()



if __name__ == "__main__":
    main()
