####################################################################################################
####################################################################################################
######### http://www.ai.mit.edu/courses/6.891-nlp/READINGS/maxent.pdf###############################
####################################################################################################
####################################################################################################
####################################################################################################



class memm_model(object):
    def __init__(self):
        self.states = set()
        self.symbols = set()
        self.transitions_f, self.transitions_b = {}, {}
        self.outputs = {} # key: (symbol, state), value: probability of P(symbol, state|symbol)

    def train(self):
        pass

    def tagged_sentence(self, untagged_sentence):
        pass

    def label_phrase(self, untagged_sentence):
        pass
