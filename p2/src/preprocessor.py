import os
from itertools import groupby
def process(file):
    """
    folder: name of folder
    TODO: wrong way to compile all files together, test files should be parsed separately,
    """
    compiled_content = []
    raw_content = open(file, 'r').read()
    compiled_content = raw_content.split('\n')

    while '' in compiled_content:
        compiled_content.remove('')

    for i, unit in enumerate(compiled_content):
        compiled_content[i] = tuple(unit.split('\t'))
    return compiled_content


def sent_process(file):
    compiled_content = []
    raw_content = open(file, 'r').read()
    compiled_content = raw_content.split('\n')

    sentence_split = [list(value) for key, value in groupby(compiled_content, lambda s: s == "") if not key]

    for i, listie in enumerate(sentence_split):
        for j, token in enumerate(listie):
            # print token
            listie[j] = tuple(token.split('\t'))
    return sentence_split


def sent_process_BIWEO(file):
    sentence_split = sent_process(file)
    for sent in sentence_split:
        left = 0
        while left < len(sent):
            sent[left] = sent[left]
            # O: outside
            if sent[left][2] == '_':
                sent[left] = (sent[left][0], sent[left][1], 'O')
                left += 1
            # W: signle word
            elif left == len(sent) - 1 or sent[left][2] != sent[left + 1][2]:
                sent[left] = (sent[left][0], sent[left][1], 'W')
                left += 1
            # BIE: begin, in, end
            else:
                right = left + 1
                while right != len(sent) - 1 and sent[right][2] == sent[left][2]:
                    right += 1
                sent[left] = (sent[left][0], sent[left][1], 'B')
                sent[right - 1] = (sent[right - 1][0], sent[right - 1][1], 'E')
                for i in xrange(left + 1, right - 1):
                    sent[i] = (sent[i][0], sent[i][1], 'I')
                left = right

    return sentence_split


def generate_path(folder):
    return os.getcwd() + "/" + folder + "/"
