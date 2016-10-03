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
        """
        self.predictors: 
            type: list of tuples
            description: list containing all the predictor phrases of uncertainty, tuples contain token(s) 
            of cue/predicting phrases
        
        self.fpredictors:
            type: dict
            description: 
                $key: first token of each self.predictors element; 
                $value: list of index of self.predictor element with $key as first token
                This variable is introduced to increase efficiency of our searching algorithm
        """

        self.predictors = []
        self.fpredictors = {}

    def feature_token_counter(self, tagged_sequence = None):
        pass

    def train(self, tagged_sequence = None):
        """
        Prepare self.predictors and self.fpredictors. 
        The source of list of predictors is:
        https://github.com/wooorm/hedges/blob/master/data.txt
        """

        predictors = ['consider', 'in my mind', 'doubtful', 'something or other', 'certainly', 'find', 'seemed', "if i'm understanding you correctly", 'somewhere', 'more or less', 'kind of', 'seems', 'actually', 'pretty', 'so far', 'must', 'might', 'someone', 'somebody', 'around', 'read', 'mainly', 'couple', 'overall', 'possible', 'possibly', 'speculate', 'like', 'at least', 'should', 'seldom', 'always', 'found', 'and so forth', 'says', 'often', 'some', 'somehow', 'understood', 'likely', 'assumption', 'estimated', 'thinks', 'really', 'probable', 'definite', 'tend', 'hopefully', 'probably', 'bunch', 'can', 'unsure', 'in my opinion', 'look like', 'doubt', 'consistent with', 'little', 'possibility', 'quite', 'surely', 'estimate', 'diagnostic', 'appears', 'suggested', 'about', 'rare', 'conceivably', 'many', 'could', 'suggests', 'et cetera', 'usually', 'appear to be', 'finds', 'presumably', 'probability', 'appeared', 'speculates', 'improbable', 'my impression', 'speculated', 'my thinking is', 'rarely', 'considers', 'presumable', 'guess', 'necessarily', 'would', 'indicate', 'assumed', 'alleged', 'approximately', 'few', 'much', 'be sure', 'assumes', 'in my understanding', 'supposedly', 'mostly', 'estimates', 'apparent', 'and all that', 'understand', 'basically', 'somewhat', 'believe', 'partially', 'supposes', 'almost never', 'largely', 'believes', 'will', 'assume', 'suggestive', 'suppose', 'supposed', 'believed', 'quite clearly', 'in general', 'inconclusive', 'unlikely', 'my understanding is', 'guesses', 'almost', 'certain', 'effectively', 'say', 'guessed', 'something', 'seem', 'apparently', 'evidently', 'sort of', 'their impression', 'perhaps', 'suggest', 'generally', 'clearly', 'occasionally', 'virtually', 'a bit', 'several', 'fairly', 'may', 'most', 'frequently', 'allege', 'appear', 'understands', 'practically', 'considered', 'maybe', 'clear', 'sometimes', 'roughly', 'in my view', 'think', 'thought', 'rather', 'definitely', 'looks like', "don't know"]
        for i, word in enumerate(predictors):
            self.predictors.append(tuple(word.split()))
            if self.predictors[i][0] not in self.fpredictors:
                self.fpredictors[self.predictors[i][0]] = [i]
            else:
                self.fpredictors[self.predictors[i][0]].append(i)


    def label_phrase(self, untagged_sequence):
        """
        parameter:
        untagged_sequence: 
            type: list of tuples, structure of tuple(TOKEN, TAG)
        
        rtype: list of tuples, tuples contains starting and ending index of predicted cues. 
               structure of tuple(starting_idx, ending_idx)
        """
        output = []
        for i, word in enumerate(untagged_sequence):

            if word[0] in self.fpredictors:
                for idx in self.fpredictors[word[0]]:
                    _len = len(self.predictors[idx])
                    sample = tuple()
                    for k in xrange(i, i + _len):
                        sample += (untagged_sequence[k][0],)
                    if sample == self.predictors[idx]:
                        output.append((i, i + _len - 1))
        return output

    def label(self, untagged_sequence):
        """
        parameter:
        untagged_sequence: 
            type: list of tuples, structure of tuple(TOKEN, TAG)
        
        rtype: list of tuples, structure of tuple(TOKEN, TAG, PREDICTION)
        """
        tagged = untagged_sequence
        for i, word in enumerate(untagged_sequence):
            tagged[i] += ("O",)

            if word[0] in self.fpredictors:
                for idx in self.fpredictors[word[0]]:
                    _len = len(self.predictors[idx])
                    sample = tuple()
                    for k in xrange(i, i + _len):
                        try:
                            sample += (untagged_sequence[k][0],)
                        except IndexError:
                            break
                    if sample == self.predictors[idx]:
                        temp = list(tagged[i])
                        temp[2] = "CUE"
                        tagged[i] = temp
        return tagged



