from test import *
from hmm_model import *
from preprocessor import *

def main():
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

if __name__ == "__main__":
    main()
