from ngram import ngram
import os
import preprocess
from gt_ngram import gt_ngram
from li_ngram import li_ngram
from bo_ngram import bo_ngram
import sys
import operator
import csv
import numpy as linspace
import math

# TODO: delete it when not used

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
topics = {'atheism':0, 'autos':1, 'graphics':2, 'medicine':3, 'motorcycles':4, 'religion':5, 'space':6}

def random_sentence_ngram(n = 2, sent_pre = "I have"):
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


def generate_perplexity_gt_ngram(mission = 'classification_task'):
    gt_ngrams = {}
    for topic in topics:
        indir = indir_pre + "data/{}/{}/train_docs".format(mission, topic)
        content = preprocess.preprocess_dir(indir)
        gt_ngrams[topic] = gt_ngram(content)

        print "\nTopic: {}".format(topic)
        for i in xrange(1, 6):
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
    for i in xrange(1, 5):
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
        print "[{}-gram] {}".format(i, accuracy[i])
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


def split_train_test(mission = 'classification_task'):
    """
    split train_docs into     training:test = 4:1
    store the preprocessed file train.txt and test.txt in each topic directory
    """

    for topic in topics:
        indir = indir_pre + "data/{}/{}/train_docs".format(mission, topic)
        num_file = len(os.listdir(indir))
        num_train_file = math.floor(num_file * 0.8)
        train_text, test_text = "", ""

        for root, dirs, filenames in os.walk(indir):
            for i, f in enumerate(filenames):
                raw_content = preprocess.preprocess_file(os.path.join(root, f))
                if i < num_train_file:
                    train_text += raw_content
                else:
                    test_text += raw_content

        train_path = indir_pre + "data/{}/{}/train.txt".format(mission, topic)
        test_path = indir_pre + "data/{}/{}/test.txt".format(mission, topic)
        open(train_path, 'w').write(train_text)
        open(test_path, 'w').write(test_text)



def topic_classification_li_ngram():
    # TODO when li_gram done, test
    # get gt_ngram for each topic and read all test data
    li_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        test_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        if not os.path.isfile(train_f) or not os.path.isfile(test_f):
            split_train_test()

        train_text[topic] = open(train_f, 'r').read()
        test_text[topic] = open(test_f, 'r').read()

        li_ngrams[topic] = li_ngram(train_text[topic])

    accuracy, r = {}, []
    for i in xrange(0, 11):
        for j in xrange(0, 11 - i):
            r[0] = round(i * 0.1, 1)
            r[1] = round(j * 0.1, 1)
            r[2] = round(1 - r[0] - r[1], 1)

            _sum, correct = 0, 0
            for label_topic, text in test_text.items():
                sentences = text.split('</s>')
                for sentence in sentences:
                    sentence += ' </s>'
                    min_perp, min_topic = sys.maxint, label_topic

                    for topic in topics:
                        perp = li_ngrams[topic].generate_perplexity(3, sentence, r)
                        if perp < min_perp:
                            min_perp = perp
                            min_topic = topic

                    if label_topic == min_topic:
                        correct += 1
                    _sum += 1

            accuracy[tuple(r)] = 1.0 * correct / _sum
            print "{}: {}".format(r, accuracy[tuple(r)])

    #choose the best r
    r_tuple = max(accuracy.iteritems(), key = operator.itemgetter(1))[0]
    r = list(r_tuple)
    print "Best: {}: {}".format(list(r_tuple), accuracy[r_tuple])

    # get the result for files in test_for_classification directory
    test_dir = indir_pre + "data/classification_task/test_for_classification"
    csv_f = indir_pre + "data/classification_task/li_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
        writer.writeheader()

        for root, dirs, filenames in os.walk(test_dir):
            for f in filenames:
                text = preprocess.preprocess_file(os.path.join(root, f))
                min_perp, min_topic = sys.maxint, ''

                for topic in topics:
                    perp = gt_ngrams[topic].generate_perplexity(n, text, r)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


def generate_perplexity_bo_ngram():
    bo_ngrams = {}
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        bo_ngrams[topic] = bo_ngram(content)

        print "\nTopic: {}".format(topic)
        print "[{}-gram]: {}".format(3, bo_ngrams[topic].generate_perplexity(3, content))


def topic_classification_bo_ngram():
    """
    calculate the accuracy for topic classification with different
    n in Good-Turing ngram, then choose the best one to classify files
    in test_for_classification directory, and write results into
    bo_result.csv in classification_task directory
    """

    # get bo_ngram for each topic and read all test data
    bo_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        test_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        if not os.path.isfile(train_f) or not os.path.isfile(test_f):
            split_train_test()

        train_text[topic] = open(train_f, 'r').read()
        test_text[topic] = open(test_f, 'r').read()

        bo_ngrams[topic] = bo_ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in bo_ngram
    for i in xrange(3,4):
        _sum, correct = 0, 0
        for label_topic, text in test_text.items():
            sentences = text.split('</s>')
            for sentence in sentences:
                sentence += ' </s>'
                min_perp, min_topic = sys.maxint, label_topic

                for topic in topics:
                    perp = bo_ngrams[topic].generate_perplexity(i, sentence)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                if label_topic == min_topic:
                    correct += 1
                _sum += 1

        accuracy[i] = 1.0 * correct / _sum
        print "[{}-gram] {}".format(i, accuracy[i])
    # choose the best n
    n = 3

    # get the result for files in test_for_classification directory
    test_dir = indir_pre + "data/classification_task/test_for_classification"
    csv_f = indir_pre + "data/classification_task/bo_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
        writer.writeheader()

        for root, dirs, filenames in os.walk(test_dir):
            for f in filenames:
                text = preprocess.preprocess_file(os.path.join(root, f))
                min_perp, min_topic = sys.maxint, ''

                for topic in topics:
                    perp = bo_ngrams[topic].generate_perplexity(n, text)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})

def generate_perplexity_bo_ngram():
    bo_ngrams = {}
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        bo_ngrams[topic] = bo_ngram(content)

        print "\nTopic: {}".format(topic)
        print "[{}-gram]: {}".format(3, bo_ngrams[topic].generate_perplexity(3, content))

def generate_perplexity_li_ngram():
    li_ngrams = {}
    for topic in topics:
        indir = indir_pre + "data/classification_task/{}/train_docs".format(topic)
        content = preprocess.preprocess_dir(indir)
        li_ngrams[topic] = li_ngram(content)

        print "\nTopic: {}".format(topic)
        print "[{}-gram]: {}".format(3, li_ngrams[topic].generate_perplexity(3, content))

def topic_classification_li_ngram():
    """
    calculate the accuracy for topic classification with different
    n in Good-Turing ngram, then choose the best one to classify files
    in test_for_classification directory, and write results into
    li_result.csv in classification_task directory
    """

    # get li_ngram for each topic and read all test data
    li_ngrams, train_text, test_text  = {}, {}, {} #key: topic
    for topic in topics:
        train_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        test_f = indir_pre + "data/classification_task/{}/train.txt".format(topic)
        if not os.path.isfile(train_f) or not os.path.isfile(test_f):
            split_train_test()

        train_text[topic] = open(train_f, 'r').read()
        test_text[topic] = open(test_f, 'r').read()

        li_ngrams[topic] = li_ngram(train_text[topic])

    # calculate the accuracy for n-gram and choose the best one
    accuracy = {} # key: the n in li_ngram
    for i in xrange(3,4):
        _sum, correct = 0, 0
        for label_topic, text in test_text.items():
            sentences = text.split('</s>')
            for sentence in sentences:
                sentence += ' </s>'
                min_perp, min_topic = sys.maxint, label_topic

                for topic in topics:
                    perp = li_ngrams[topic].generate_perplexity(i, sentence)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                if label_topic == min_topic:
                    correct += 1
                _sum += 1

        accuracy[i] = 1.0 * correct / _sum
        print "[{}-gram] {}".format(i, accuracy[i])
    #choose the best n
    n = 3

    # get the result for files in test_for_classification directory
    test_dir = indir_pre + "data/classification_task/test_for_classification"
    csv_f = indir_pre + "data/classification_task/li_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
        writer.writeheader()

        for root, dirs, filenames in os.walk(test_dir):
            for f in filenames:
                text = preprocess.preprocess_file(os.path.join(root, f))
                min_perp, min_topic = sys.maxint, ''

                for topic in topics:
                    perp = li_ngrams[topic].generate_perplexity(n, text)
                    if perp < min_perp:
                        min_perp = perp
                        min_topic = topic

                writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})


def spell_checker_gt_nrgam(method = 'perplexity'):
    gt_ngrams, train_text, test_text, test_compare = {}, {}, {}, {} #key: topic
    
    indir_words = indir_pre + "data/spell_checking_task/confusion_set.txt"
    confused_words = open(indir_words, 'r').read().decode("utf-8-sig").encode("utf-8")
    #in the txt, the phrase "may be" confuses split(). for simplicity, we picked it out.
    #create two dictionaries for word checking
    confused_words.replace("may be", "maybe")
    confused_words = confused_words.split()
    words = {}
    changed_text = ""
    words["even"] = dict()
    words["odd"] = dict()
    end = False;
    for i in xrange(0, len(confused_words)-1, 2):
        words["even"][confused_words[i]]=[]
        words["odd"][confused_words[i+1]]=[]

    for i in xrange(0, len(confused_words)-1, 2):
        words["even"][confused_words[i]].append(confused_words[i+1])
        words["odd"][confused_words[i+1]].append(confused_words[i])

    words["even"]["maybe"] = ["may be"]

    for side in words:

        for indexw in words[side]:
            words[side][indexw].append("")

    for topic in topics:
        gt_ngrams[topic] = dict()
        test_text[topic] = dict()
        test_compare[topic] = dict()
        for i in xrange(1, 5):
            gt_ngrams[topic][i] = dict()


    #create training and testing text
    for topic in topics:
        train_f = indir_pre + "data/spell_checking_task/{}/train.txt".format(topic)
        test_f = indir_pre + "data/spell_checking_task/{}/train_modified_docs".format(topic)
        test_cf = indir_pre + "data/spell_checking_task/{}/train_docs".format(topic)

        num_file = len(os.listdir(test_f))
        num_train_file = math.floor(num_file * 0.8)

        for root, dirs, filenames in os.walk(test_f):
            for i, f in enumerate(filenames):
                if i > num_train_file:
                    raw_content = preprocess.preprocess_file(os.path.join(root, f),"sentences")
                    #test_text[topic][f] = dict()
                    test_text[topic][f] = raw_content
                    raw_content = preprocess.preprocess_file(os.path.join(test_cf, f.replace("_modified", "")),"sentences")
                    test_compare[topic][f.replace("_modified", "")] = raw_content
        if not os.path.isfile(train_f):
            split_train_test('spell_checking_task')
        train_text[topic] = open(train_f, 'r').read()
        for i in xrange(1, 5):
            gt_ngrams[topic][i] = gt_ngram(train_text[topic])

    #generate good-turing n-gram for the training set
    for i in xrange(1, 5):
        for label_topic, text in train_text.items():
            gt_ngrams[topic][i].npro_dict = gt_ngrams[topic][i].generate_ngram(i)
    
    #spell check and accuracy calculation
    #initiate variables
    correct_rate = {}
    correct = {}
    sentence_count = {}
    compare_count = {}
    compare_rate = {}
    for i in xrange(1, 5):
        sentence_count[i] = 0
        correct[i] = 0
        compare_count[i] = 0
        for topic in topics:
            #each file to be examined under the topic
            for filename in test_text[topic]:
                filename_compare = filename
                filename_compare = filename_compare.replace("_modified","")
                #each sentence in the file
                sen_processing = test_text[topic][filename][0].split()
                for j in xrange (0, len(test_text[topic][filename])):
                    #generate the perplexity of the original sentence
                    perp_origin = gt_ngrams[topic][i].generate_perplexity(i,test_text[topic][filename][j])
                    #two-way dictionary
                    for side in words:
                        for word1 in words[side]:
                            sen_tokens = test_text[topic][filename][j].split()
                            #if there's a confused word
                            if word1 in sen_tokens:
                                #if processing a new sentence, count +1
                                sen_tokens != sen_processing:
                                sentence_count[i] += 1
                                sen_processing = sen_tokens
                                #list of alternative sentences
                                alternative = []
                                #list of perplexity for the alternative sentences
                                perp_alt = []
                                #alternative.append(test_text[topic][filename][j])
                                #for each possible alternate word
                                changed_text = test_text[topic][filename][j]
                                
                                """
                                break the sentence into a list of tokens, replace the token to be examined,
                                reconstruct the list into a string
                                """
                                new_word = word1
                                for k in xrange(0,len(words[side][word1])):   
                                    #replace the word of interest in the list of tokens
                                    alternative.append(sen_tokens)
                                    alternative[k] = [w.replace(word1,''.join(words[side][word1][k])) for w in alternative[k]]
                                    
                                    #reconstructing the new sentence
                                    sent_Tobe = ""
                                    for m in xrange(0, len(alternative[k])):
                                        if alternative[k][m].isalpha():
                                            sent_Tobe += ' '.join(alternative[k][m])
                                        else:
                                            sent_Tobe += ''.join(alternative[k][m])
                                    
                                    #generate the perplexity of the new sentence
                                    if len(sent_Tobe)>0:
                                        perp_alt.append(gt_ngrams[topic][i].generate_perplexity(i,sent_Tobe))
                                    else: 
                                        perp_alt.append(1.0*sys.maxint)
                                    
                                    #if the new sentence has a lower perplexity
                                    if perp_alt[k] < perp_origin and perp_alt[k] == min(perp_alt):
                                                #replace the old sentence with the new one
                                                #test_text[topic][filename][j]=''.join(alternative[k])
                                        changed_text = sent_Tobe
                                        new_word = words[side][word1][k]

                                if new_word in test_compare[topic][filename_compare][j]:
                                #test_compare[topic][filename_compare][j]:
                                    correct[i] += 1
                                if test_compare[topic][filename_compare][j] == test_text[topic][filename][j]:
                                    compare_count[i] += 1
                                        #print i, correct[i], sentence_count[i]
                                        #print the new sentence with the new word
        #print correct
        #print sentence_count
        correct_rate[i] = 1.0 * correct[i] / sentence_count[i]
        compare_rate[i] = 1.0 * compare_count[i] / sentence_count[i]
        print "{} gram correct spell check rate = {}".format(i, correct_rate[i])
        print "{} gram correct compare rate = {}".format(i, compare_rate[i])








def main():

    #split_train_test('spell_checking_task')
    #topic_classification_gt_ngram()
    spell_checker_gt_nrgam()



if __name__ == "__main__":
    main()
