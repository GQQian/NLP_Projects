import os
import sys
import p3_preprocess
from textblob import TextBlob
from gt_ngram import gt_ngram

dir_question = os.getcwd() + "/question_dev.txt"
questions = p3_preprocess.question_preprocess(dir_question)
#questions is a dict with key = ID and value = question

text = {}
data = {}

for x in xrange(89, 321):
	dir_train = os.getcwd() + "/doc_dev/" + str(x) + "/"
	text[str(x)+"\r"] = {}
	for y in xrange(1, 21):
		data = p3_preprocess.doc_process(dir_train+str(y))
		text[str(x)+"\r"][str(y)] = data
		data = []
	"""
	for root, dirs, filenames in os.walk(dir_train):
	    for i, f in enumerate(filenames):

	    	Because of the low accuracy of the baseline method, the accuracy of
	    	using 20 files to generate answers is the same as using 100 files.
	    	For faster testing, we only use 20 files for now.

	    	if(i < 20):
		        data = p3_preprocess.doc_process(root + f)
		        text[str(x)+"\r"][f] = data
	        else:
	        	break
	"""

guess = {}
question_ngram = {}

for key in questions:
	guess[key] = dict((x, ("", sys.maxint, "")) for x in xrange(0, 5))
	question_ngram[key] = gt_ngram(questions[key])


	for textfile in text[key]:
		for x in xrange(0,len(text[key][textfile])):
			sent = text[key][textfile][x]
			perp = question_ngram[key].generate_perplexity(2, sent)

			#smaller perplexity: more similar to question
			if perp < guess[key][4][1]:
				answer_blob = TextBlob(sent)

				answers = answer_blob.noun_phrases
				if len(answers) > 0:
					answer = ""
					for phrase in answers:
						if phrase not in questions[key]:
							answer = phrase
					if answer == "":
						answer = answers[0]
				else:
					sent_list = sent.split()
					if len(sent_list)> 10:
						answer = sent_list[0]
						for n_words in range(1,10):
							answer += ' ' + sent_list[n_words]
					else:
						answer = sent

				guess[key][4] = [answer, perp, textfile]
				for y in xrange(1,5):
					if perp < guess[key][4-y][1]:
						guess[key][5-y] = guess[key][4-y]
						guess[key][4-y] = [answer, perp, textfile]
					else:
						break

#output:
with open("baseline_answer_dev.txt", "w") as text_file:
	for key in questions:
		for x in xrange(0,5):
			text_file.write(key.replace("\r", "") + " " + guess[key][x][2] + " " + guess[key][x][0] + "\n")
