from test import *
from hmm_model import *
from preprocessor import *

def main():
<<<<<<< HEAD
    print "\n[Viterbi-bio]"
    uncertain_detection_hmm(model=hmm_viterbi_model, file_prefix="viterbi_bio")
    print "\n[Forward-bio]"
    uncertain_detection_hmm(model=hmm_forward_model, file_prefix="forward_bio")
    print "\n[FW-bio]"
    uncertain_detection_hmm(model=hmm_forward_backward_model, file_prefix="bw_bio")


    print "\n[Viterbi-bmewo]"
    uncertain_detection_hmm(model=hmm_viterbi_model, file_prefix="viterbi_bmewo", tagging=sent_process_bmweo)
    print "\n[Forward-bmewo]"
    uncertain_detection_hmm(model=hmm_forward_model, file_prefix="forward_bmewo", tagging=sent_process_bmweo)
    print "\n[FW-bmewo]"
    uncertain_detection_hmm(model=hmm_forward_backward_model, file_prefix="bw_bmewo", tagging=sent_process_bmweo)
=======
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
>>>>>>> 9f41aebd3d5ed863af21df47a3c56d0ab1a0e325

if __name__ == "__main__":
    main()
