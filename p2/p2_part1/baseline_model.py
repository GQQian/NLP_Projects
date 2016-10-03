import os
import random

# TODO: 1. train: 
#              type: dictionary
#              keys: (TOKEN, TAGS)
#              vals: counts/ level of indications
#       2. getting data, and outputing predictions
#       3. 

class baseline_model(object):
    def __init__(self):
        self.predictors = []
        self.fpredictors = [] 

    def feature_token_counter(self, tagged_sequence = None):
        pass

    def train(self, tagged_sequence = None):
        predictors = ['consider', 'in my mind', 'doubtful', 'something or other', 'certainly', 'find', 'seemed', "if i'm understanding you correctly", 'somewhere', 'more or less', 'kind of', 'seems', 'actually', 'pretty', 'so far', 'must', 'might', 'someone', 'somebody', 'around', 'read', 'mainly', 'couple', 'overall', 'possible', 'possibly', 'speculate', 'like', 'at least', 'should', 'seldom', 'always', 'found', 'and so forth', 'says', 'often', 'some', 'somehow', 'understood', 'likely', 'assumption', 'estimated', 'thinks', 'really', 'probable', 'definite', 'tend', 'hopefully', 'probably', 'bunch', 'can', 'unsure', 'in my opinion', 'look like', 'doubt', 'consistent with', 'little', 'possibility', 'quite', 'surely', 'estimate', 'diagnostic', 'appears', 'suggested', 'about', 'rare', 'conceivably', 'many', 'could', 'suggests', 'et cetera', 'usually', 'appear to be', 'finds', 'presumably', 'probability', 'appeared', 'speculates', 'improbable', 'my impression', 'speculated', 'my thinking is', 'rarely', 'considers', 'presumable', 'guess', 'necessarily', 'would', 'indicate', 'assumed', 'alleged', 'approximately', 'few', 'much', 'be sure', 'assumes', 'in my understanding', 'supposedly', 'mostly', 'estimates', 'apparent', 'and all that', 'understand', 'basically', 'somewhat', 'believe', 'partially', 'supposes', 'almost never', 'largely', 'believes', 'will', 'assume', 'suggestive', 'suppose', 'supposed', 'believed', 'quite clearly', 'in general', 'inconclusive', 'unlikely', 'my understanding is', 'guesses', 'almost', 'certain', 'effectively', 'say', 'guessed', 'something', 'seem', 'apparently', 'evidently', 'sort of', 'their impression', 'perhaps', 'suggest', 'generally', 'clearly', 'occasionally', 'virtually', 'a bit', 'several', 'fairly', 'may', 'most', 'frequently', 'allege', 'appear', 'understands', 'practically', 'considered', 'maybe', 'clear', 'sometimes', 'roughly', 'in my view', 'think', 'thought', 'rather', 'definitely', 'looks like', "don't know"]
        for i, word in enumerate(predictors):
            self.predictors.append(tuple(word.split()))
            self.fpredictors.append(self.predictors[i][0])


    def label(self, untagged_sequence):
        # untagged_sequence should be a list of 
        output = []
        for i, word in enumerate(untagged_sequence):

            for j, cue in enumerate(self.fpredictors):
                if word[0] == cue:
                    _len = len(self.predictors[j])
                    sample = tuple()
                    for k in xrange(i, i + _len):
                        sample += (untagged_sequence[k][0],)
                    if sample == self.predictors[j]:
                        output.append((i, i + _len - 1))
        return output