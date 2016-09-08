import os

indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
sentence_maxlen = 100
nprob_dic, nhash_dic, ncounter = {}, {}, {}

def preprocess(indir):
    # TODO
    # email, Upper-lower case
    # sentence boundary
    # ...

    pro_content = ""
    for root, dirs, filenames in os.walk(indir):
        for f in filenames:
            raw_content = open(os.path.join(root, f),'r').read()
            pro_content += raw_content

    return pro_content


def ntoken_count(n, content):
    counter = {}
    tokens = content.split()
    _len = len(tokens)
    for i in xrange(_len - n + 1):
        key = tuple(tokens[i:(i + n)])
        counter[key] = counter.get(key, 0) + 1

    return counter


def ngram_generator(n, content):
    ncounter[n] = ncounter.get(n, ntoken_count(n, content))
    nhash_dic[n], nprob_dic[n] = {}, {}

    if n == 1:
        _sum = sum(ncounter[n].values())
        nprob_dic[n] = dict((key, num * 1.0 / _sum) for key, num in ncounter[n].items())
    elif n > 1:
        ncounter[n - 1] = ncounter.get(n - 1, ntoken_count(n - 1, content))
        for key_n, num_n in ncounter[n].items():
            key_nminus1 = key_n[:-1]

            nhash_dic[n][key_nminus1] = nhash_dic[n].get(key_nminus1, [])
            nhash_dic[n][key_nminus1].append(key_n)

            num_nminus1 = ncounter[n - 1][key_nminus1]
            nprob_dic[n][key_n] = 1.0 * num_n / num_nminus1


def sentence_generator(n, prob_dic, hash_dic, pre_sent = ""):
    # TODO:





    prob_table = {}
    if n == 1:
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
    ngram_generator(n, content)
    print nprob_dic[n]
    print nhash_dic[n]


if __name__ == "__main__":
    main()
