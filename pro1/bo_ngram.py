# linear interpolation ngram
class bo_ngram(gt_ngram):
    def __init__(self,content,r = [0.75, 0.25]):
        super(li_ngram,self).__init__(content)
        self.r = np.array(r)


