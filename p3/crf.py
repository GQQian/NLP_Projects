import p3_preprocess
import os
import sys
import re
from nltk.tag import StanfordNERTagger
from geotext import GeoText
import nltk
from location_extractor import extract_locations
from timex import *
import math




def compute_train_score_and_label(test_or_dev = "dev"):
    dir_question = os.getcwd() + "/question_{}.txt".format(test_or_dev)
    questions = p3_preprocess.question_preprocess_with_pos(dir_question)
    passage_max_num = 10
    st = StanfordNERTagger(os.getcwd()+'/english.all.3class.distsim.crf.ser.gz', os.getcwd()+"/stanford-ner.jar")

    # key: question id, value: set of correct answers
    correct_answers = {}
    fname = os.getcwd() + '/pattern.txt'
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            _split = line.split()
            question_id, answer = _split[0], _split[1]

            if question_id in correct_answers:
                correct_answers[question_id].add(answer)
            else:
                correct_answers[question_id] = set([answer])

    # list of answers with format:
    # question id;answer;label;tfidf score;chunk score;pre_post score
    answers = []


    for _id, question in questions.items():
        question_id = _id.replace('\r', '')
        if question_id not in correct_answers:
            continue

        # if int(question_id) % 4 != 0:
        #     continue

        print "[[{}]]".format(question_id)


        p_idx = {} # key: p, value: idx
        idx_p = {} # key: idx, value: p
        idx_count = 0

        ######################## get question_type ########################
        if question[0][0].startswith('When'):
            _type = 'time'
        elif question[0][0].startswith('Where'):
            _type = 'LOCATION'
        else:
            _type = 'PERSON'

        # get the NN, V, JJ words as t
        t_set = set()
        for _tuple in question:
            if _tuple[1].startswith('NN') or _tuple[1].startswith('V') or \
               _tuple[1].startswith('JJ'):
                t_set.add(_tuple[0])
        lower_t_set = [word.lower() for word in t_set]

        ######################## Passage retrieval ########################
        passages_tf = {} # key: text(p), value: list of tf with count
        p_doc_dict = {} # key: text(p), value: doc_id
        idf_count = {} # key: word, value: idf
        _id = _id.replace("\r", "")
        dir = os.getcwd() + "/doc_{}/{}/".format(test_or_dev, _id)

        for filename in os.listdir(dir):
            try:
                int(filename)
            except ValueError:
                print "!!!error"
                print filename
                continue

            if int(filename) > 40:
                continue

            f = dir + filename
            doc_passages = p3_preprocess.doc_process(f)

            for p in doc_passages:
                lower_p = p.lower()
                p_idx[p] = idx_count
                idx_p[idx_count] = p
                p_doc_dict[idx_count] = filename


                passages_tf[idx_count] = dict((word, lower_p.count(word)) for word in lower_t_set)

                for word, count in passages_tf[idx_count].items():
                    if count > 0:
                        idf_count[word] = idf_count.get(word, 0) + 1

                idx_count += 1

            # for different doc, use 1 to saperate
            idx_count += 1


        # compute idf and p_tfidf_score
        doc_num = len(passages_tf)
        # idf  key: word   value: log(doc total number / doc with the word number)
        idf = dict((word, math.log(doc_num/idf_count[word])) for word in idf_count)
        # p_tfidf_score    key: text(p), value: tfidf score
        p_tfidf_score = {}
        for idx, tf_count in passages_tf.items():
            p_tfidf_score[idx] = sum(list(tf_count[word] * idf[word] for word in idf))

        # rank p with p_tfidf_score, get passages with passage_max_num
        _sorted = sorted(p_tfidf_score.items(), key=lambda x: -x[1])[:passage_max_num]
        passage_idxes = [_tuple[0] for _tuple in _sorted]
        p_tfidf_score = dict((idx, p_tfidf_score[idx]) for idx in passage_idxes)

        # update p_idx and idx_p
        idx_p = dict((idx, idx_p[idx]) for idx in passage_idxes)
        p_idx = dict((p, p_idx[p]) for p in idx_p.values())

        del passages_tf

        ######################## get possible answer with scores according to types ########################
        p_results_dict = {} # key: passage idx, value: list of results
        result_len = 0
        if _type == 'PERSON':
            for p in p_idx:
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
                    p_results_dict[p_idx[p]] = people
                    result_len += len(people)

        elif _type == 'LOCATION':
            for p in p_idx:
                geo = GeoText(p)
                places = geo.cities
                places.extend(geo.countries)
                places.extend(extract_locations(p))

                if len(places) > 0:
                    p_results_dict[p_idx[p]] = places
                    result_len += len(places)
        else:
            for p in p_idx:
                tagged = tag(p)
                timexes = re.findall('<TIMEX2>.*</TIMEX2>', tagged)
                timexes = [str.replace('<TIMEX2>', '').replace('</TIMEX2>', '') for str in timexes]

                if len(timexes) > 0:
                    p_results_dict[p_idx[p]] = timexes
                    result_len += len(timexes)

        # key: tuple(answer, idx of p), value: tfidf score
        answer_tfidf_score = {}
        for idx, _list in p_results_dict.items():
            for answer in _list:
                # TODO a bug here   hard code
                if idx in p_tfidf_score:
                    answer_tfidf_score[tuple([answer, idx])] = p_tfidf_score[idx]

        del p_results_dict


        ######################## get answer chunk score ########################

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

        # key: p idx, value: chunk count
        p_chunk_score = {}
        for p, idx in p_idx.items():
            lower_p = p.lower()
            p_chunk_score[idx] = sum(list(lower_p.count(chunk) for chunk in lower_chunks))


        # key: answer, value: chunk count
        answer_chunk_score = {}
        for answer_p in answer_tfidf_score:
            p_idx = answer_p[1]
            answer_chunk_score[answer_p] = p_chunk_score[p_idx]

        del p_chunk_score


        ######################## get pre and post p score for answer ########################
        answer_pre_pro_score = {}
        for answer_p in answer_chunk_score:
            p_idx = answer_p[1]
            pre_idx, pro_idx = p_idx-1, p_idx+1

            pre_score = p_tfidf_score.get(pre_idx, 0)
            post_score = p_tfidf_score.get(pro_idx, 0)

            answer_pre_pro_score[answer_p] = 1.0 * (pre_score + post_score) / 2


        ######################## label the answer ########################
        correct_set = correct_answers[question_id]
        print correct_set

        for answer_p in answer_tfidf_score:
            answer = answer_p[0]
            tfidf_score = answer_tfidf_score[answer_p]
            chunk_score = answer_chunk_score[answer_p]
            pre_pro_score = answer_pre_pro_score[answer_p]

            label = 0
            for _correct in correct_set:
                if re.match(_correct, answer):
                    label = 1
                    break

            # answer format:
            # question id;answer;label;tfidf score;chunk score;pre_post score
            answer_str = "{0};{1};{2};{3};{4};{5}\n".format(question_id, answer, label, tfidf_score, chunk_score, pre_pro_score)

            print answer_str

            answers.append(answer_str)

    # write answers into answer_labeled.txt
    with open("train_answer_labeled.txt", "w") as f:
        for answer_str in answers:
            f.write(answer_str)

def compute_test_score(test_or_dev = "test"):
    dir_question = os.getcwd() + "/question_{}.txt".format(test_or_dev)
    questions = p3_preprocess.question_preprocess_with_pos(dir_question)
    passage_max_num = 10
    st = StanfordNERTagger(os.getcwd()+'/english.all.3class.distsim.crf.ser.gz', os.getcwd()+"/stanford-ner.jar")

    # list of answers with format:
    # question id;answer;tfidf score;chunk score;pre_post score
    answers = []

    for _id, question in questions.items():
        question_id = _id.replace('\r', '')

        print "[[{}]]".format(question_id)

        p_idx = {} # key: p, value: idx
        idx_p = {} # key: idx, value: p
        idx_count = 0

        ######################## get question_type ########################
        if question[0][0].startswith('When'):
            _type = 'time'
        elif question[0][0].startswith('Where'):
            _type = 'LOCATION'
        else:
            _type = 'PERSON'

        # get the NN, V, JJ words as t
        t_set = set()
        for _tuple in question:
            if _tuple[1].startswith('NN') or _tuple[1].startswith('V') or \
               _tuple[1].startswith('JJ'):
                t_set.add(_tuple[0])
        lower_t_set = [word.lower() for word in t_set]

        ######################## Passage retrieval ########################
        passages_tf = {} # key: text(p), value: list of tf with count
        p_doc_dict = {} # key: text(p), value: doc_id
        idf_count = {} # key: word, value: idf
        _id = _id.replace("\r", "")
        dir = os.getcwd() + "/doc_{}/{}/".format(test_or_dev, _id)

        for filename in os.listdir(dir):
            try:
                int(filename)
            except ValueError:
                print "!!!error"
                print filename
                continue

            if int(filename) > 25:
                continue

            f = dir + filename
            doc_passages = p3_preprocess.doc_process(f)

            for p in doc_passages:
                lower_p = p.lower()
                p_idx[p] = idx_count
                idx_p[idx_count] = p
                p_doc_dict[idx_count] = filename


                passages_tf[idx_count] = dict((word, lower_p.count(word)) for word in lower_t_set)

                for word, count in passages_tf[idx_count].items():
                    if count > 0:
                        idf_count[word] = idf_count.get(word, 0) + 1

                idx_count += 1

            # for different doc, use 1 to saperate
            idx_count += 1


        # compute idf and p_tfidf_score
        doc_num = len(passages_tf)
        # idf  key: word   value: log(doc total number / doc with the word number)
        idf = dict((word, math.log(doc_num/idf_count[word])) for word in idf_count)
        # p_tfidf_score    key: text(p), value: tfidf score
        p_tfidf_score = {}
        for idx, tf_count in passages_tf.items():
            p_tfidf_score[idx] = sum(list(tf_count[word] * idf[word] for word in idf))

        # rank p with p_tfidf_score, get passages with passage_max_num
        _sorted = sorted(p_tfidf_score.items(), key=lambda x: -x[1])[:passage_max_num]
        passage_idxes = [_tuple[0] for _tuple in _sorted]
        p_tfidf_score = dict((idx, p_tfidf_score[idx]) for idx in passage_idxes)

        # update p_idx and idx_p
        idx_p = dict((idx, idx_p[idx]) for idx in passage_idxes)
        p_idx = dict((p, p_idx[p]) for p in idx_p.values())

        del passages_tf

        ######################## get possible answer with scores according to types ########################
        p_results_dict = {} # key: passage idx, value: list of results
        result_len = 0
        if _type == 'PERSON':
            for p in p_idx:
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
                    p_results_dict[p_idx[p]] = people
                    result_len += len(people)

        elif _type == 'LOCATION':
            for p in p_idx:
                geo = GeoText(p)
                places = geo.cities
                places.extend(geo.countries)
                places.extend(extract_locations(p))

                if len(places) > 0:
                    p_results_dict[p_idx[p]] = places
                    result_len += len(places)
        else:
            for p in p_idx:
                tagged = tag(p)
                timexes = re.findall('<TIMEX2>.*</TIMEX2>', tagged)
                timexes = [str.replace('<TIMEX2>', '').replace('</TIMEX2>', '') for str in timexes]

                if len(timexes) > 0:
                    p_results_dict[p_idx[p]] = timexes
                    result_len += len(timexes)

        # key: tuple(answer, idx of p), value: tfidf score
        answer_tfidf_score = {}
        for idx, _list in p_results_dict.items():
            for answer in _list:
                # TODO a bug here   hard code
                if idx in p_tfidf_score:
                    answer_tfidf_score[tuple([answer, idx])] = p_tfidf_score[idx]

        del p_results_dict


        ######################## get answer chunk score ########################

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

        # key: p idx, value: chunk count
        p_chunk_score = {}
        for p, idx in p_idx.items():
            lower_p = p.lower()
            p_chunk_score[idx] = sum(list(lower_p.count(chunk) for chunk in lower_chunks))


        # key: answer, value: chunk count
        answer_chunk_score = {}
        for answer_p in answer_tfidf_score:
            p_idx = answer_p[1]
            answer_chunk_score[answer_p] = p_chunk_score[p_idx]

        del p_chunk_score


        ######################## get pre and post p score for answer ########################
        answer_pre_pro_score = {}
        for answer_p in answer_chunk_score:
            p_idx = answer_p[1]
            pre_idx, pro_idx = p_idx-1, p_idx+1

            pre_score = p_tfidf_score.get(pre_idx, 0)
            post_score = p_tfidf_score.get(pro_idx, 0)

            answer_pre_pro_score[answer_p] = 1.0 * (pre_score + post_score) / 2


        ######################## get the answer_str ########################
        for answer_p in answer_tfidf_score:
            answer = answer_p[0]
            idx = answer_p[1]
            tfidf_score = answer_tfidf_score[answer_p]
            chunk_score = answer_chunk_score[answer_p]
            pre_pro_score = answer_pre_pro_score[answer_p]
            doc_id = p_doc_dict[idx]


            # answer format:
            # question id;answer;doc id;tfidf score;chunk score;pre_post score
            answer_str = "{0};{1};{2};{3};{4};{5}\n".format(question_id, doc_id, answer, tfidf_score, chunk_score, pre_pro_score)

            print answer_str

            answers.append(answer_str)

        print ""


    # write answers into answer_labeled.txt
    with open("test_answer_unlabeled.txt", "w") as f:
        for answer_str in answers:
            f.write(answer_str)

def crf_model():
    train_data = []
    test_data = []


    fname = os.getcwd() + '/train_answer_labeled.txt'
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            _split = line.split(';')
            # answer format:
            # question id;answer;label;tfidf score;chunk score;pre_post score
            pre_post_score = round(float(_split[-1]), 3)
            chunk_score = round(float(_split[-2]), 3)
            tfidf_score = round(float(_split[-3]), 3)
            label = int(_split[-4])

            train_data.append([label, tfidf_score, chunk_score, pre_post_score])


    fname = os.getcwd() + '/test_answer_unlabeled.txt'
    with open(fname) as f:
        lines = f.readlines()
        for line in lines:
            _split = line.split(';')
            # answer format:
            # question id;answer;doc id;tfidf score;chunk score;pre_post score
            pre_post_score = round(float(_split[-1]), 3)
            chunk_score = round(float(_split[-2]), 3)
            tfidf_score = round(float(_split[-3]), 3)

            test_data.append([_split[0], _split[1], _split[2], tfidf_score, chunk_score, pre_post_score])


    # TODO: Zili Xiang





compute_train_score_and_label()
# compute_test_score()
# crf_model()
