import p3_preprocess
import os
import sys
import re
from nltk.tag import StanfordNERTagger
from geotext import GeoText
import nltk
from location_extractor import extract_locations
from timex import *

test_or_dev = "dev"

dir_question = os.getcwd() + "/question_{}.txt".format(test_or_dev)
questions = p3_preprocess.question_preprocess_with_pos(dir_question)
passage_max_num = 10
st = StanfordNERTagger(os.getcwd()+'/english.all.3class.distsim.crf.ser.gz', os.getcwd()+"/stanford-ner.jar")

answers = {} # key: question _id, value: tuple(doc__id, result)
for _id, question in questions.items():
    # get question _type
    if question[0][0].startswith('When'):
        _type = 'time'
    elif question[0][0].startswith('Where'):
        _type = 'LOCATION'
    else:
        _type = 'PERSON'

    # get the NN words
    words_nn = set()
    for _tuple in question:
        if _tuple[1].startswith('NN'):
            words_nn.add(_tuple[0])

    # Passage retrieval and get passages with the first passage_max_num scores
    passages_dict = {} # key: text, value: score
    p_doc_dict = {} # key: passage, value: doc_id
    _id = _id.replace("\r", "")
    # dir = os.getcwd() + "/doc_dev/{}/".format(_id)
    dir = os.getcwd() + "/doc_{}/{}/".format(test_or_dev, _id)

    for filename in os.listdir(dir):
        f = dir + filename
        doc_passages = p3_preprocess.doc_process(f)

        for p in doc_passages:
            lower_p = p.lower()
            lower_words_nn = [word.lower() for word in words_nn]

            score = sum(list(lower_p.count(word) for word in lower_words_nn))
            passages_dict[p] = score
            p_doc_dict[p] = filename

    _sorted = sorted(passages_dict.items(), key=lambda x: -x[1])[:passage_max_num]
    passages = [p[0] for p in _sorted]

    p_results_dict = {} # key: passage, value: list of results
    result_len = 0
    if _type == 'PERSON':
        for p in passages:
            people = []
            tags = st.tag(p.split())
            left, right = 0, 0
            while left < len(tags):
                if tags[left][1] == 'PERSON':
                    right = left + 1
                    while right < len(tags) and tags[right][1] == 'PERSON':
                        right += 1
                    person = ' '.join([i[0] for i in tags[left:right]])
                    people.append(person)
                    left = right
                left += 1

            if len(people) > 0:
                p_results_dict[p] = people
                result_len += len(people)

    elif _type == 'LOCATION':
        for p in passages:
            geo = GeoText(p)
            places = geo.cities
            places.extend(geo.countries)
            places.extend(extract_locations(p))

            if len(places) > 0:
                p_results_dict[p] = places
                result_len += len(places)
    else:
        for p in passages:
            tagged = tag(p)
            timexes = re.findall('<TIMEX2>.*</TIMEX2>', tagged)
            timexes = [str.replace('<TIMEX2>', '').replace('</TIMEX2>', '') for str in timexes]

            if len(timexes) > 0:
                p_results_dict[p] = timexes
                result_len += len(timexes)
    print p_results_dict.values()

    # Select all complex nominals with their adjectival modifiers
    question_newtag = []
    for _tuple in question:
        if _tuple[1].startswith('NN'):
            question_newtag.append((_tuple[0], 'NN',))
        elif _tuple[1].startswith('JJ'):
            question_newtag.append((_tuple[0], 'JJ',))
        else:
            question_newtag.append(_tuple)

    chunks = []
    grammar = "NP: {<DT>?<JJ>*<NN>+}"
    cp = nltk.RegexpParser(grammar)

    t = cp.parse(question_newtag)
    for np in t.subtrees(filter=lambda x: x.label() == 'NP'):
        chunk = []
        for i in np.leaves():
            chunk.append(i[0])

        chunks.append(' '.join(chunk))

    results_dict = {}
    lower_chunks = [chunk.lower() for chunk in chunks]
    for p, results_list in p_results_dict.items():
        lower_p = p.lower()
        score = sum(list(lower_p.count(chunk) for chunk in lower_chunks))
        for result in p_results_dict[p]:
            results_dict[(p_doc_dict[p], result,)] = score

    _sorted = sorted(results_dict.items(), key=lambda x: -x[1])[:min(5, len(results_dict))]
    answers[_id] = [_tuple[0] for _tuple in _sorted]
    print "[[{}]]".format(_id)
    print answers[_id]

# write answer into file
with open("ner_answer_{}.txt".format(test_or_dev), "w") as text_file:
    for question_id, tuples in answers.items():
        for _tuple in tuples:
            doc_id, answer = _tuple[0], _tuple[1]
            text_file.write("{} {} {}\n".format(question_id, doc_id, answer))
