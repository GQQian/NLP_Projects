import os

split_token = "||"
indir_pre = os.getcwd() + "/"
outdir_pre = os.getcwd() + "/"
sen_maxlen = 100


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
    counter_n = ntoken_count(n, content)
    ngram = {}

    if n == 1:
        _sum = sum(counter.values())
        ngram = dict((combine, num * 1.0 / _sum) for combine, num in counter_n.items())
    elif n > 1:
        counter_nminus1 = ntoken_count(n - 1, content)
        for combine, num_n in counter_n.items():
            num_nminus1 = counter_nminus1[combine[(combine.find(split_token) + len(split_token)):]]
            ngram[combine] = 1.0 * num_n / num_nminus1

    return ngram


def ntoken_count(n, content):
    counter = {}
    tokens = content.split()
    _len = len(tokens)
    for i in xrange(_len - n + 1):
        combine = split_token.join(tokens[i:(i + n)])
        counter[combine] = counter.get(combine, 0) + 1

    return counter


def sentence_generator(pre_sent, prob):
    # TODO:
    _len = len(pre_sent)


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
    ngram = ngram_generator(n, content)
    print ngram


main()
