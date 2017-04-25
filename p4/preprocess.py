import nltk
import re
import os
from nltk.tokenize import sent_tokenize, word_tokenize
indir_pre = os.getcwd() + "/"

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

    text = text.replace(' From :', ' ')
    text = text.replace(' Subject :', ' ')
    text = text.replace(' Re :', ' ')

    text = text.lower()
    text = remove_email(text)
    text = remove_punctuation(text)

    text = text.replace(' i ', ' I ')
    text = text.replace(' i\' ', ' I\' ')

    sent_list = sent_tokenize(text)
    if form == "sentences":
        output = sent_list
    if form == "weTrain":
        _set = set()
        output = []
        for sent in sent_list:
            j = 0
            k = 0
            newtext = sent
            while j < len(sent)-1:
                if newtext[k].isalpha() == False and newtext[k] != ' ':
                    newtext = newtext[:k] + newtext[k+1:]
                    j = j + 1
                else:
                    k = k + 1
                    j = j + 1
            sent = newtext
            tokens = sent.split()
            for i, token in enumerate(tokens):
                if token not in _set:
                    tokens[i] = 'unk'
                    _set.add(token)
            output.append(tokens)
        #corrected_f =  indir_pre + "data/spell_checking_task/weTrain.txt"
        #open(corrected_f,'w').write(output)
    if form == "default":
        output = ""
        for sent in sent_list:
            output +=  sent 

    return output
