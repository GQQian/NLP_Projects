import nltk
import re
import os
from nltk.tokenize import sent_tokenize, word_tokenize
class preprocess(self):
    def __init__():


    def remove_punctuation(text):
        text = text.replace('_', '')
        result = re.findall(r'[\w\,\.\!\?]+',text)
        return ' '.join(result)

    def remove_email(text):
        result = re.sub(r'[\w\.-]+@[\w\.-]+','',text)
        return result

def preprocess(indir):
    """
    Preprocess all the files in the directory
    input: the directory
    output: the processed content from all files and merged into one string
    """
    def remove_punctuation(text):
        text = text.replace('_', '')
        result = re.findall(r'[\w\,\.\!\?]+',text)
        return ' '.join(result)

    def remove_email(text):
        result = re.sub(r'[\w\.-]+@[\w\.-]+','',text)
        return result

    buffer, output = "", ""
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            raw_content = open(os.path.join(root, f),'r').read()
            buffer += raw_content

    # normalize
    buffer = buffer.lower()
    buffer = remove_email(buffer)
    buffer = remove_punctuation(buffer)

    # corner case
    buffer = buffer.replace(' i ', ' I ')
    buffer = buffer.replace(' i\' ', ' I\' ')
    buffer = buffer.replace(' From :', ' ')
    buffer = buffer.replace(' Subject :', ' ')
    buffer = buffer.replace(' Re :', ' ')

    sent_list = sent_tokenize(buffer)
    for sent in sent_list:
        output += " <s> " + sent + " </s> "

    return output
    
