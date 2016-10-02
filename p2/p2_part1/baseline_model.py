import os
import random

# TODO: 1. Feature_Extractor: 
#              type: dictionary
#              keys: (TOKEN, TAGS)
#              vals: counts/ level of indications
#       2. getting data, and outputing predictions
#       3. 

class baseline_model(object):
    def __init__(self):
        self.predictors = {}

    def feature_token_counter(self, input = None):
    	pass

    def feature_extractor(self, input = None):
        self.predictors[(would,)] = 1
        self.predictors[(should,)] = 1
        self.predictors[(might,)] = 1
        self.predictors[(may,)] = 1
        self.predictors[(probably,)] = 1
        self.predictors[(uncertainly,)] = 1
        self.predictors[(unlikely,)] = 1
        self.predictors[(improbable,)] = 1
        self.predictors[(doubtful,)] = 1
        self.predictors[(usually,)] = 1

    def parse_corpus(self, file):
        # sentence should be a list of 
        pass