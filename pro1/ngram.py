import os

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
sentence_maxlen = 100


def preprocess(indir):
    # TODO
    # email, Upper-lower case
    # sentence boundary

    pro_content = ""
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            raw_content = open(os.path.join(root, f),'r').read()
            pro_content += raw_content

    return pro_content


def ngram_generator(n, content):
    def ntoken_count(n, content):
        counter = {}
        tokens = content.split()
        _len = len(tokens)
        for i in xrange(_len - n + 1):
            key = tuple(tokens[i:(i + n)])
            counter[key] = counter.get(key, 0) + 1

        return counter

    counter_n = ntoken_count(n, content)
    prob_dic, hash_dic = {}, {}

    if n == 1:
        _sum = sum(counter_n.values())
        prob_dic = dict((key, num * 1.0 / _sum) for key, num in counter_n.items())
    elif n > 1:
        counter_nminus1 = ntoken_count(n - 1, content)
        for key_n, num_n in counter_n.items():
            key_nminus1 = key_n[:-1]

            hash_dic[key_nminus1] = hash_dic.get(key_nminus1, [])
            hash_dic[key_nminus1].append(key_n)

            num_nminus1 = counter_nminus1[key_nminus1]
            prob_dic[key_n] = 1.0 * num_n / num_nminus1

    return prob_dic, hash_dic


def sentence_generator(prob_dic, hash_dic, pre_sent = ""):
    # TODO:
    pass



def main():
    argv = ["test", "2"] # TODO: input
    if (len(argv) == 0):
        print "Please input a topic"

    topic = argv[0]
    n = int(argv[1]) if len(argv) > 1 and argv[1].isdigit() \
        and int(argv[1]) >= 1 else 1

    indir, outdir = indir_pre + topic, outdir_pre + topic

    if not os.path.isdir(indir):
        print "Sorry, the topic does not exist!"
        return
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    content = preprocess(indir)
    prob_dic, hash_dic = ngram_generator(n, content)
    print prob_dic
    print hash_dic


if __name__ == "__main__":
    main()
