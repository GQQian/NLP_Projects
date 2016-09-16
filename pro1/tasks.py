import ngram
import os
import preprocess

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"

def random_sentence_ngram(n = 2, sent_pre = "I have", topic = "autos"):
    # TODO: lili: create ngram class for every topic, and implemente sentence generation for each topic
    indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
    content = preprocess.preprocess(indir)
    ngrams = ngram.ngram_Generator()

    for k in xrange(1, n + 1):
        print "\n\n[{}-gram]\n".format(k)

        print "Empty sentence"
        for i in xrange(3):
            print "[{}]  ".format(i + 1) + ngrams.generate_sentence(k, content)

        print "\nWith incomplete sentence: " + "\"{}\"".format(sent_pre)
        for i in xrange(3):
            print "[{}]  ".format(i + 1) + ngram.sentence_generator(k, content, sent_pre)


def topic_classification():
    pass


def spell_checker():
    pass

def main():
    random_sentence_ngram(topic="atheism")




if __name__ == "__main__":
    main()
