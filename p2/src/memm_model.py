####################################################################################################
####################################################################################################
######### http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf###############################
####################################################################################################
####################################################################################################
####################################################################################################

"""
 feature of interest: f<b,s>(o_t,s_t) = 1 if b(o_t) is true and s = s_t, f = 0 otherwise
 Here, b(o_t) is the feature of interest. We determined that the feature to be discussed
 is that if the corresponding state of observation o_t was B(Beginning) or I(inside), and 
 the "next state" s is within the domain of the possible domain of s_t from the training set, 
 Then we set f = 1. For the E(end) tag, if the previous tag is I, and s is determined to be an 
 O(outside), W(single word), or B, we change the previous I tag to an E tag.

 notation: state: the BIWEO tags created by preprocessor. We do not use p-o-s tags here
"""
import nltk
from nltk.stem.porter import *
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open


class memm_model(object):
    def __init__(self):
        self.maxent_classifier = object
        self.boi_full_list = [] #store all the boi tags that occur in the training set
        self.boi_end_list = [] #store boi tags that are at the end of the sentence
        self.posStartList = [] #store pos that are begining of the sentence
        self.BOI_list = ('B', 'I', 'O', 'W', 'E')
        self.labeled_features = []
        self.pos_biew = []


    def MEMM_features(self, word, pos, previous_tag):
        #stemmer = PorterStemmer() 
        features = {} 
        features['current_word'] = word
        features['current_tag'] = pos
        features['ofBIE'] = pos in self.pos_biew
        #puc = '-'.decode("utf-8")
         #some char is outof ASCII
        features['start_of_sentence'] = pos in self.posStartList
        #features['end_of_sentence'] = pos in self.boi_end_list
        features['previous_tag'] = previous_tag
        return features


    #words_content, poses_content, states_content = [], [], []
    #word2pos, word2state = {}, {}

###############################################################################################
    def train(self, tagged_sentences = None):
        """
        parameter:
        tagged_sequences:
            type: list of lists of tuples(sentences), structure of tuple(TOKEN, POS, TAG)
        """
        # merge all symbols and states
        for sent in tagged_sentences:

            #first word of the sentence
            token1 = sent[0]
            word = token1[0]
            pos = token1[1]
            tag = token1[2] 
            item = word, pos, 'start', tag
            if(tag != 'O'):
                self.pos_biew.append(pos)
            self.labeled_features.append(item)
            self.posStartList.append(pos)
            self.boi_full_list.append(tag)
            for i in xrange(1,len(sent)):
                token = sent[i]
                word = token[0]
                pos = token[1]
                tag = token[2] 
                self.boi_full_list.append(tag)
                item = word, pos, sent[i-1][2], tag
                self.labeled_features.append(item)

        train_set = []
        for (word, pos, previous_tag, tag) in self.labeled_features:

            features = self.MEMM_features(word, pos, previous_tag)
            train_set.append((features, tag))


        #f = open("my_classifier.pickle", "wb")

        self.maxent_classifier = MaxentClassifier.train(train_set, max_iter=2)
        #pickle.dump(self.maxent_classifier , f)

        #f.close() 


###############################################################################################

    def tag_sentence(self, untagged_sentence):

        wordList = []
        poslist = []
        #print(untagged_sentence)
        for i in xrange(0, len(untagged_sentence)):
            wordList.append(untagged_sentence[i][0])
            poslist.append(untagged_sentence[i][1])
        w1 = wordList[0] #the first word of the sentence
        pos1 = poslist[0]
        tRange = len(self.BOI_list)
        wRange = len(wordList)
        path = []

        probability = self.maxent_classifier.prob_classify(self.MEMM_features(w1, pos1, 'start')) 
        posterior_min = 0
        tagchosen = 0

        for t in xrange(tRange):
            posterior = float(probability.prob(self.BOI_list[t]))
            if(posterior > posterior_min):
                posterior_min = posterior
                tagchosen = t
        path.append(self.BOI_list[tagchosen])

        for w in range (1, wRange): 
            posterior_min = 0
            tagchosen = 0
            for t in range (tRange):
                word = wordList[w]
                pos = poslist[w]
                features = self.MEMM_features(word,pos,path[w-1])
                probability = self.maxent_classifier.prob_classify(features) 
                posterior = float(probability.prob(self.BOI_list[t]))
               
                #print(probability)
                #print(posterior)
                if(posterior > posterior_min):  
                    posterior_min = posterior
                    tagchosen = t

            if(tagchosen != 2):
                print(tagchosen)

            path.append(self.BOI_list[tagchosen])

        #print (path)
        return path



    def label_phrase(self, untagged_sentence):
        pass
