import os
from nltk.tokenize import sent_tokenize

#preprocess docs, INPUT: a txt file, OUTPUT: a list of sentences(only include the <TEXT> section)
def doc_process(file):

	raw_content = open(file, 'r').read()
	#print (raw_content)
	i = 0
	while i < len(raw_content):
		if(raw_content[i : i+6] == "<TEXT>" or raw_content[i : i+6] == "[Text]"):
			raw_content = raw_content[(i+6):]
			i = 0

		if(raw_content[i : i+7] == "</TEXT>"):
			raw_content = raw_content[:i]
			break
		i = i + 1
	raw_content = raw_content.replace("<P>", "")
	raw_content = raw_content.replace("</P>", "")
	#raw_content = raw_content.replace(',', ' ')
	#raw_content = raw_content.replace(';', ' ')
	#raw_content = raw_content.replace(':', ' ')
	x = 0
	while x < len(raw_content):
		if(raw_content[x].isalpha() == False and raw_content[x] != " "):
			if(raw_content[x] == ',', ';', ':'):
				raw_content = raw_content[0:x] + raw_content[(x+1):]
			else:
				raw_content = raw_content[0:x] + raw_content[(x+1):]
			x = x - 1
		x = x + 1
	raw_content = raw_content.encode("utf8")
	sent_list = sent_tokenize(raw_content)
	return sent_list



#Return a dictionary with keys = ID of questions, value = qeustion
def question_preprocess(file):
	questions = {}
	#qeustion_list = []
	raw_content = open(file, 'r').read()
	raw_content = raw_content.encode("utf8")
	i = 0
    #seperate each questions by <top>, <\top> tag
	content_list = raw_content.split('\n')


	for x in xrange(0, len(content_list)):
		if(content_list[x][0:5] == "<num>"):
			question_id = content_list[x][14:]
		if(content_list[x][0:6] == "<desc>"):
			question = content_list[x+1]
			questions[question_id] = question


	return questions


def patterns_preprocess(file):
	patterns = {}

	raw_content = open(file, 'r').read()
	raw_content = raw_content.encode("utf8")
	content_list = raw_content.split('\n')
	for x in xrange(0, len(content_list)):
		#print(content_list[x])
		token = content_list[x].split()
		if(token[0] not in patterns):
			patterns[token[0]] = []
		patterns[token[0]].append(token[1])
	return patterns

"""

dir_train = os.getcwd() + "/question.txt"
question_preprocess(dir_train)
"""

"""

dir_train = os.getcwd() + "/doc_dev/100/"
data_combined = []
for root, dirs, filenames in os.walk(dir_train):
    for i, f in enumerate(filenames):
    	#print(root+f)
        data = doc_process(root + f)
        #print(data)

"""