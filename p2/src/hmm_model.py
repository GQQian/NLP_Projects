from gt_ngram import gt_ngram
from ngram import ngram


class hmm_model(object):
    def __init__(self):
        self.states = set()
        self.symbols = set()
        self.transitions = {}
        self.outputs = {} # key: (symbol, state), value: probability of P(symbol, state|state)
        self.state_start_prob = {}


    def train(self, tagged_sentences = None):
        """
        parameter:
        tagged_sequences:
            type: list of tuples, structure of tuple(TOKEN, POS, TAG)
        """

        # merge all symbols and states
        symbol_content, state_content = [], []
        for sent in tagged_sentences:
            self.state_start_prob[sent[0][2]] = self.state_start_prob.get(sent[0][2], 0) + 1
            for token in sent:
                symbol_content.append(token[0])
                state_content.append(token[2])
        self.state_start_prob = dict((state, 1.0*self.state_start_prob[state]/len(tagged_sentences)) for state in self.state_start_prob)

        # use ngram, gt_ngram to get states, symbols set, transitions
        symbol_ngram = gt_ngram(" ".join(symbol_content))
        state_ngram = ngram(" ".join(state_content))

        self.symbols = set(symbol_ngram.ntoken_count(1).keys())
        self.states = set(state_ngram.ntoken_count(1).keys())

        self.transitions = state_ngram.generate_ngram(2)

        # compute self.outputs
        count_dict = {} # key: tuple(symbol, state),  value: count
        for sent in tagged_sentences:
            for token in sent:
                symbol, state = token[0], token[2]
                _tuple = (symbol, state)
                count_dict[_tuple] = count_dict.get(_tuple, 0) + 1

        for key, val in count_dict.items():
            self.outputs[key] = 1.0 * val / state_ngram.ncounter_dic[1][tuple(key[1])]


    def tag_sentence(self, untagged_sentence):
        """
        parameter:
        tagged_sequences:
            type: list of tuples, structure of tuple(TOKEN, POS)
        rtype: list of TAG
        """

        return ['O'] * len(untagged_sentence)


    def label_phrase_untagged(self, untagged_sentence):
        """
        parameter:
        tagged_sequences:
            type: list of tuples, structure of tuple(TOKEN, POS)

        rtype: list of tuples, tuples contains starting and ending index of predicted cues.
               structure of tuple(starting_idx, ending_idx)
        """

        tags = self.tag_sentence(untagged_sentence)
        return self.label_phrase_tagged(tags)

    def label_phrase_tagged(self, tags):
        output = []
        left, right = 0, 0
        while left < len(tags):
            if tags[left] == 'W':
                output.append(tuple([left, left]))
                left += 1
            elif tags[left] == 'B':
                right = left + 1
                while right < len(tags) and tags[right] != 'O':
                    right += 1
                output.append(tuple([left, right - 1]))
                left = right
            else:
                left += 1

        return output


class hmm_viterbi_model(hmm_model):
    def tag_sentence(self, untagged_sentence):
        """
        parameter:
        tagged_sequences:
            type: list of tuples, structure of tuple(TOKEN, POS)
        rtype: list of TAG
        """

        tags = ['O'] * len(untagged_sentence)

        # 1st word
        token = untagged_sentence[0]
        tuples = [tuple([token[0], state[0]]) for state in self.states]
        _max = 0

        for _tuple in tuples:
            if _tuple in self.outputs and self.outputs[_tuple] > _max and _tuple[1] in self.state_start_prob:
                _max = self.outputs[_tuple] * self.state_start_prob[_tuple[1]]
                tags[0] = _tuple[1]

        # words after 1st one
        for i in xrange(1, len(untagged_sentence)):
            token = untagged_sentence[i]
            tuples = [tuple([token[0], state[0]]) for state in self.states]
            _max = 0

            for _tuple in tuples:
                if _tuple in self.outputs and tuple([tags[0], _tuple[1]]) in self.transitions:
                    curr = self.outputs[_tuple] * self.transitions[tuple([tags[0], _tuple[1]])]
                    if curr > _max:
                        _max = curr
                        tags[i] = _tuple[1]

        return tags


class hmm_forward_model(hmm_model):
    def tag_sentence(self, untagged_sentence):
        """
        parameter:
        tagged_sequences:
            type: list of tuples, structure of tuple(TOKEN, POS)
        rtype: list of TAG
        """

        tags = ['O'] * len(untagged_sentence)

        # tag's score, key: state, value: score
        curr_score = {'O': 0}

        # 1st word
        token = untagged_sentence[0]

        tuples = [tuple([token[0], state[0]]) for state in self.states]

        for _tuple in tuples:
            if _tuple in self.outputs and _tuple[1] in self.state_start_prob:
                curr_score[_tuple[1]] = self.outputs[_tuple] * self.state_start_prob[_tuple[1]]

        tags[0] = max(curr_score.items(), key=lambda x: x[1])[0]

        # words after 1st one
        for i in xrange(1, len(untagged_sentence)):
            last_score, curr_score = curr_score, {'O': 0}
            token = untagged_sentence[i]
            tuples = [tuple([token[0], state[0]]) for state in self.states]

            for _tuple in tuples:
                if _tuple in self.outputs:
                    curr_tag, curr_score[curr_tag] = _tuple[1], 0
                    for last_tag in last_score:
                        if tuple([last_tag, curr_tag]) in self.transitions:
                            curr_score[curr_tag] += self.outputs[
                            _tuple] * self.transitions[tuple([last_tag, curr_tag])] * \
                                                    last_score[last_tag]

            # To avoid all later scorses are 0
            if len(curr_score) == 0 or max(curr_score.values()) == 0:
                curr_score['O'] = 1

            tags[i] = max(curr_score.items(), key=lambda x: x[1])[0]

        return tags


class hmm_bw_model(hmm_model):
    def __init__(self):
        self.states = set()
        self.symbols = set()
        self.transitions_f, self.transitions_b = {}, {}
        self.outputs = {} # key: (symbol, state), value: probability of P(symbol, state|state)


    def train(self, tagged_sentences):
        # merge all symbols and states
        symbol_content, state_content = [], []
        for sent in tagged_sentences:
            for token in sent:
                symbol_content.append(token[0])
                state_content.append(token[2])

        # use ngram, gt_ngram to get states, symbols set, self.transitions
        symbol_ngram = gt_ngram(" ".join(symbol_content))

        state_ngram_f = ngram(" ".join(state_content))
        state_ngram_b = ngram(" ".join(reversed(state_content)))

        self.symbols = set(symbol_ngram.ntoken_count(1).keys())
        self.states = set(state_ngram_b.ntoken_count(1).keys())

        self.transitions_f = state_ngram_f.generate_ngram(2)
        self.transitions_b = state_ngram_b.generate_ngram(2)

        # compute self.outputs
        count_dict = {} # key: tuple(symbol, state),  value: count
        for sent in tagged_sentences:
            for token in sent:
                symbol, state = token[0], token[2]
                _tuple = (symbol, state)
                count_dict[_tuple] = count_dict.get(_tuple, 0) + 1

        for key, val in count_dict.items():
            self.outputs[key] = 1.0 * val / state_ngram_f.ncounter_dic[1][tuple(key[1])]


    def tag_sentence(self, untagged_sentence):
        """
        parameter:
        tagged_sequences:
            type: list of tuples, structure of tuple(TOKEN, POS)
        rtype: list of TAG
        """
        def get_scores(untagged_sentence, transitions):
            # tag's score is a list of dict, key: state, value: score
            scores = []


            # tag's score, key: state, value: score
            curr_score = {'O': 0}
            curr_tag = "O"

            # 1st word
            token = untagged_sentence[0]
            tuples = [tuple([token[0], state[0]]) for state in self.states]

            for _tuple in tuples:
                if _tuple in self.outputs:
                    curr_score[_tuple[1]] = self.outputs[_tuple]

            curr_tag = max(curr_score.items(), key=lambda x: x[1])[0]
            scores.append(curr_score)

            # words after 1st one
            for i in xrange(1, len(untagged_sentence)):
                last_score, curr_score = curr_score, {'O': 0}
                last_tag, curr_tag = curr_tag, 'O'

                token = untagged_sentence[i]
                tuples = [tuple([token[0], state[0]]) for state in self.states]

                for _tuple in tuples:
                    if _tuple in self.outputs:
                        curr_tag, curr_score[curr_tag] = _tuple[1], 0
                        for last_tag in last_score:
                            if tuple([last_tag, curr_tag]) in transitions:
                                curr_score[curr_tag] += self.outputs[_tuple] * transitions[tuple([last_tag, curr_tag])] * \
                                                        last_score[last_tag]

                curr_tag = max(curr_score.items(), key=lambda x: x[1])[0]
                scores.append(curr_score)

            return scores

        reversed_sentence = [untagged_sentence[-i] for i in xrange(len(untagged_sentence))]
        scores_f = get_scores(untagged_sentence, self.transitions_f)
        scores_b = get_scores(reversed_sentence, self.transitions_b)


        _len = len(scores_f)
        tags = ['O'] * _len
        for i in xrange(_len):
            _max = 0
            for tag in scores_f[i]:
                if tag in scores_b[-i] and scores_b[-i][tag] * scores_f[i][tag] > _max:
                    _max = scores_b[-i][tag] * scores_f[i][tag]
                    tags[i] = tag

        return tags
