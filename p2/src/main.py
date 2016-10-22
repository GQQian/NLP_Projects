## TODO: 1. read files, parse files
#       2. feeding data to models
#       3. getting predictions from models and write to files
from test import *
from hmm_model import *


def main():
#############################################################################################
#############################################################################################
##### replace the function below with whatever function you want to call from test.py #######
#############################################################################################
#############################################################################################
    # print "[Viterbi]"
    # uncertain_detection_hmm(model=hmm_viterbi_model)
    # print "\n[Forward]"
    # uncertain_detection_hmm(model=hmm_forward_model)
    print "CRF"
    uncertain_detection_crf()    

if __name__ == "__main__":
    main()
