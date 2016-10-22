####################################################################################################
####################################################################################################
######### http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf###############################
####################################################################################################
####################################################################################################
####################################################################################################

import nltk
from nltk.classify import MaxentClassifier
import pickle
import os,sys
from io import open
from ngram import ngram


class memm_model(object):
    def __init__(self):
        self.maxent_classifier = object
        self.boi_full_list = [] #store all the boi tags that occur in the training set
        self.boi_end_list = [] #store boi tags that are at the end of the sentence
        self.posStartList = [] #store pos that are begining of the sentence
        self.BOI_list = ('B', 'M', 'O', 'W', 'E')
        self.labeled_features = []
        self.pos_biew = []
        self.state_ngram = object
        self.transitions = {}
        self.pos_B = []
        self.pos_M = []
        self.pos_E = []
        self.word_confuse = []
        self.states = set()

    def MEMM_features(self, word, pos, previous_tag):
        #stemmer = PorterStemmer() 
        features = {} 
        #features['current_word'] = word
        #features['current_tag'] = pos
        features['ofBIE'] = pos in self.pos_biew
        #puc = '-'.decode("utf-8")
         #some char is outof ASCII
        features['ofB'] = pos in self.pos_B
        features['ofM'] = pos in self.pos_M
        features['ofE'] = pos in self.pos_E
        features['start_of_sentence'] = pos in self.posStartList
        #features['end_of_sentence'] = pos in self.boi_end_list
        features['previous_tag'] = previous_tag
        features['confuse word'] = word in self.word_confuse
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
        state_content = []
        for sent in tagged_sentences:

            #first word of the sentence
            token1 = sent[0]
            word = token1[0]
            pos = token1[1]
            tag = token1[2] 
            state_content.append('start')
            state_content.append(tag)
            item = word, pos, 'start', tag
            if(tag != 'O' and tag != 'W'):
                self.pos_biew.append(pos)
                self.word_confuse.append(word)
            if(tag == 'B'):
                self.pos_B.append(pos)
            if(tag == 'M'):
                self.pos_M.append(pos)
            if(tag == 'E'):
                self.pos_E.append(pos)
            self.labeled_features.append(item)
            self.posStartList.append(pos)
            self.boi_full_list.append(tag)
            for i in xrange(1,len(sent)):
                token = sent[i]
                word = token[0]
                pos = token[1]
                tag = token[2] 
                if(i == len(sent)-1):
                    self.boi_end_list.append(tag)
                if(tag != 'O' and tag != 'W'):
                    self.word_confuse.append(word)
                    self.pos_biew.append(pos)
                if(tag == 'B'):
                    self.pos_B.append(pos)
                if(tag == 'I'):
                    self.pos_I.append(pos)
                if(tag == 'E'):
                    self.pos_E.append(pos)
                state_content.append(tag)
                self.boi_full_list.append(tag)
                item = word, pos, sent[i-1][2], tag
                self.labeled_features.append(item)
        self.state_ngram = ngram(" ".join(state_content))
        self.states = set(self.state_ngram.ntoken_count(1).keys())
        self.transitions = self.state_ngram.generate_ngram(2)
        #print(self.transitions)
        train_set = []
        for (word, pos, previous_tag, tag) in self.labeled_features:

            features = self.MEMM_features(word, pos, previous_tag)
            train_set.append((features, tag))


        f = open("my_classifier.pickle", "wb")

        self.maxent_classifier = MaxentClassifier.train(train_set, max_iter=2)
        pickle.dump(self.maxent_classifier , f)
        
        f.close() 


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
            posterior = 0
            if(('start',self.BOI_list[t]) not in self.transitions):
                posterior = 0
            else:
                posterior = float(self.transitions['start',self.BOI_list[t]]*probability.prob(self.BOI_list[t]))
            if(posterior > posterior_min):
                posterior_min = posterior
                tagchosen = t
        path.append(self.BOI_list[tagchosen])
        print(path)

        for w in range (1, wRange): 
            posterior_min = 0
            tagchosen = 0
            word = wordList[w]
            pos = poslist[w]
            features = self.MEMM_features(word,pos,path[w-1])
            probability = self.maxent_classifier.prob_classify(features) 
            for t in range (tRange):
                posterior = 0
                if((path[w-1],self.BOI_list[t]) not in self.transitions):
                    posterior = 0
                else:
                    posterior = self.transitions[path[w-1],self.BOI_list[t]] * float(probability.prob(self.BOI_list[t]))
               
                if(posterior > posterior_min):  
                    posterior_min = posterior
                    tagchosen = t


            path.append(self.BOI_list[tagchosen])
        return path

        

    def label_phrase(self, untagged_sentence):
        pass
