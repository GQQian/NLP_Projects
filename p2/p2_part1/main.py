# TODO: 1. read files, parse files
#       2. feeding data to models
#       3. getting predictions from models and write to files
import os
from preprocessor import process, generate_path, sent_process
from baseline_model import baseline_model
import csv

def uncertain_phrase_detection():
    folder_pub = "test-public"
    folder_pri = "test-private"
    bm = baseline_model()
    directory_pub = generate_path(folder_pub)
    directory_pri = generate_path(folder_pri)
    bm.train()

    csv_f = os.getcwd() + "/" + "phrase_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['Type', 'Spans'])
        writer.writeheader()

        data_combined = []
        for root, dirs, filenames in os.walk(directory_pub):
            for f in filenames:
                data = process(directory_pub + f)
                data_combined += data
        pub_result = bm.label_temp(data_combined)

        pub_result_str = ""
        for label in pub_result:
            pub_result_str += str(label[0]) + '-' + str(label[1]) + ' '

        writer.writerow({'Type': "CUE-public", 'Spans': pub_result_str})   

        data_combined = []
        pri_result_str = ""
        for root, dirs, filenames in os.walk(directory_pri):
            for f in filenames:
                data = process(directory_pri + f)
                data_combined += data
        pri_result = bm.label_temp(data_combined)

        for label in pri_result:
            pri_result_str += str(label[0]) + '-' + str(label[1]) + ' '

        writer.writerow({'Type': "CUE-private", 'Spans': pri_result_str}) 


def uncertain_sent_detection():
    folder_pub = "test-public"
    folder_pri = "test-private"
    directory_pub = generate_path(folder_pub)
    directory_pri = generate_path(folder_pri)

    bm = baseline_model()
    bm.train()

    csv_f = os.getcwd() + "/" + "sentence_result.csv"

    with open(csv_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['Type', 'Indices'])
        writer.writeheader()

        data_combined = []
        for root, dirs, filenames in os.walk(directory_pub):
            for f in filenames:
                data = sent_process(directory_pub + f)
                data_combined += data
                # print data_combined
        # print "length for public: {}".format(len(data_combined))
        # pub_result = bm.label(data_combined)

############ public part ############

        pub_result = []
        data_combined.pop(0)
        for sent in data_combined:
            # print sent
            pub_result.append(bm.label(sent))

        sent_result_pub = []
        for i, sent in enumerate(pub_result):
            for token in sent:
                if token[2] == "CUE":
                    sent_result_pub.append(i)
                    break

        pub_result_str = ""
        for label in sent_result_pub:
            pub_result_str += str(label) + ' '
        writer.writerow({'Type': "SENTENCE-public", 'Indices': pub_result_str})   

############### Private part ################

        data_combined = []
        pri_result_str = ""
        for root, dirs, filenames in os.walk(directory_pri):
            for f in filenames:
                data = sent_process(directory_pri + f)
                data_combined += data

        pri_result = []
        for sent in data_combined:
            pri_result.append(bm.label(sent))

        sent_result_pri = []
        for i, sent in enumerate(pri_result):
            for token in sent:
                if token[2] == "CUE":
                    sent_result_pri.append(i)
                    break

        pri_result_str = ""
        for label in sent_result_pri:
            pri_result_str += str(label) + ' '

        pri_result = bm.label(data_combined)

        # for label in pri_result:
        #     pri_result_str += str(label[0]) + '-' + str(label[1]) + ' '
        writer.writerow({'Type': "SENTENCE-private", 'Indices': pri_result_str}) 


def main():
    # writing to uncertainty phrase detection
    uncertain_sent_detection()




if __name__ == "__main__":
    main()