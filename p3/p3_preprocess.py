import os
import nltk
import re

#preprocess docs, INPUT: a txt file, OUTPUT: a list of sentences(only include the <TEXT> section)
def doc_process(file):
	raw_content = open(file, 'r').read()
	raw_content = raw_content.decode('utf-8','ignore').encode("utf-8")

	# get <TEXT> </TEXT> out
	if "<TEXT>" in raw_content:
		start = raw_content.index("<TEXT>")+6
		end = raw_content.index("</TEXT>")
	elif "<LEADPARA>" in raw_content:
		start = raw_content.index("<LEADPARA>")+10
		end = raw_content.index("</LEADPARA>")
	else:
		return []

	raw_content = raw_content[start:end]

	# normalize and split to sentences
	raw_content = raw_content.replace("<P>", " ").replace("</P>", " ")
	raw_content = raw_content.replace("\n", " ")
	raw_content = raw_content.replace("\r", " ")
	sentences = nltk.sent_tokenize(raw_content)

	normalized_sent = []
	for s in sentences:
		# s = s.replace("\n", "")
		# s = s.replace("\a", "")
		s = ''.join(ch for ch in s if ch.isalnum() or ch == ' ')
		s = " ".join(s.split())
		if len(s) > 0:
			normalized_sent.append(s)

	return normalized_sent


# Return a dictionary with keys = ID of questions, value = question
def question_preprocess(file):
	questions = {}
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

# Return a dictionary with keys = ID of questions, value = list of tuples(word, pos)
def question_preprocess_with_pos(file):
	questions = question_preprocess(file)
	questions_with_pos = {}
	for index, question in questions.items():
		text = nltk.word_tokenize(question)
		questions_with_pos[index] = nltk.pos_tag(text)
	return questions_with_pos



def patterns_preprocess(file):
	patterns = {}

	raw_content = open(file, 'r').read()
	raw_content = raw_content.encode("utf8")
	content_list = raw_content.split('\n')
	for x in xrange(0, len(content_list)):
		token = content_list[x].split()
		if(token[0] not in patterns):
			patterns[token[0]] = []
		patterns[token[0]].append(token[1])
	return patterns
