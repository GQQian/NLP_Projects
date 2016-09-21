import nltk
import re
import os
from nltk.tokenize import sent_tokenize, word_tokenize


def preprocess_file(f, form = "default"):
    text = open(f, 'r').read()
    return preprocess_text(text, form)


def preprocess_dir(dir, form = "default"):
    """
    Preprocess all the files in the directory
    input: the directory
    output: the processed content from all files and merged into one texting
    """
    text = ""
    for root, dirs, filenames in os.walk(dir):
        for f in filenames:
            raw_content = open(os.path.join(root, f),'r').read()
            text += raw_content
    return preprocess_text(text, form)


def preprocess_text(text, form = "default"):
    def remove_punctuation(text):
        text = text.replace('_', '')
        result = re.findall(r'[\w\,\.\!\?]+',text)
        return ' '.join(result)

    def remove_email(text):
        result = re.sub(r'[\w\.-]+@[\w\.-]+','',text)
        return result

    # normalize
    text = text.lower()
    text = remove_email(text)
    text = remove_punctuation(text)

    # corner case
    text = text.replace(' i ', ' I ')
    text = text.replace(' i\' ', ' I\' ')
    text = text.replace(' From :', ' ')
    text = text.replace(' Subject :', ' ')
    text = text.replace(' Re :', ' ')

    sent_list = sent_tokenize(text)
    if form == "sentences":
        output = sent_list
    else:
        output = ""
        for sent in sent_list:
            output += " <s> " + sent + " </s> "
    
    return output
