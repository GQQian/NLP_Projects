import nltk
import re
import os
from nltk.tokenize import sent_tokenize, word_tokenize


def preprocess_file(f):
    str = open(os.path.join(root, f),'r').read()
    return preprocess_str(str)


def preprocess_dir(dir):
    """
    Preprocess all the files in the directory
    input: the directory
    output: the processed content from all files and merged into one string
    """
    str = ""
    for root, dirs, filenames in os.walk(dir):
        for f in filenames:
            raw_content = open(os.path.join(root, f),'r').read()
            str += raw_content
    return preprocess_str(str)


def preprocess_str(str):
    def remove_punctuation(text):
        text = text.replace('_', '')
        result = re.findall(r'[\w\,\.\!\?]+',text)
        return ' '.join(result)

    def remove_email(text):
        result = re.sub(r'[\w\.-]+@[\w\.-]+','',text)
        return result

    # normalize
    str = str.lower()
    str = remove_email(str)
    str = remove_punctuation(str)

    # corner case
    str = str.replace(' i ', ' I ')
    str = str.replace(' i\' ', ' I\' ')
    str = str.replace(' From :', ' ')
    str = str.replace(' Subject :', ' ')
    str = str.replace(' Re :', ' ')

    sent_list = sent_tokenize(str)

    output = ""
    for sent in sent_list:
        output += " <s> " + sent + " </s> "

    return output
