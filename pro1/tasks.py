import ngram
import os
import config

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"

def random_sentence_ngram(n = 2, sent_pre = "I have"):
    indir = indir_pre + "data/classification_task/test_for_classification"
    content = config.preprocess(indir)
    for k in xrange(1, n + 1):
        print "\n\n[{}-gram]\n".format(k)

        print "Empty sentence"
        for i in xrange(3):
            print "[{}]  ".format(i + 1) + ngram.sentence_generator(k, content)

        print "\nWith incomplete sentence: " + "\"{}\"".format(sent_pre)
        for i in xrange(3):
            print "[{}]  ".format(i + 1) + ngram.sentence_generator(k, content, sent_pre)



def main():
    random_sentence()




if __name__ == "__main__":
    main()
