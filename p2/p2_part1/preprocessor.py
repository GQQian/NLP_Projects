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


def generate_path(folder):
    return os.getcwd() + "/" + folder + "/"
