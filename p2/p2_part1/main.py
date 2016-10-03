# TODO: 1. read files, parse files
#       2. feeding data to models
#       3. getting predictions from models and write to files
import os
from preprocessor import process, generate_path
from baseline_model import baseline_model
import csv

# INPUT: FOLDER NAME
folder_pub = "test-public"
folder_pri = "test-private"
bm_pub = baseline_model()
bm_pri = baseline_model()
directory_pub = generate_path(folder_pub)
directory_pri = generate_path(folder_pri)
bm_pub.train()
bm_pri.train()

txt_f = os.getcwd() + "/" + "result.csv"
def main():
    # with open(csv_f, 'w') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames = ['ID', 'Prediction'])
    #     writer.writeheader()

    #     for root, dirs, filenames in os.walk(test_dir):
    #         for f in filenames:
    #             text = preprocess.preprocess_file(os.path.join(root, f))
    #             min_perp, min_topic = sys.maxint, ''

    #             for topic in topics:
    #                 perp = gt_ngrams[topic].generate_perplexity(n, text)
    #                 if perp < min_perp:
    #                     min_perp = perp
    #                     min_topic = topic

    #             writer.writerow({'ID': f, 'Prediction': '{}'.format(topics[min_topic])})



    with open(txt_f, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames = ['Type', 'Spans'])
        writer.writeheader()

        data_combined = []
        for root, dirs, filenames in os.walk(directory_pub):
            for f in filenames:
                data = process(directory_pub + f)
                data_combined += data
        pub_result = bm_pub.label(data_combined)

        pub_result_str = ""
        for label in pub_result:
            if label[1] - label[0] == 0:
                pub_result_str += str(label[0]) + " "
            else:
                pub_result_str += str(label[0]) + '-' + str(label[1]) + ' '

        writer.writerow({'Type': "CUE-public", 'Spans': pub_result_str})   

        data_combined = []
        pri_result_str = ""
        for root, dirs, filenames in os.walk(directory_pri):
            for f in filenames:
                data = process(directory_pri + f)
                data_combined += data
        pri_result = bm_pri.label(data_combined)

        for label in pri_result:
            if label[1] - label[0] == 0:
                pri_result_str += str(label[0]) + ' '
            else:
                pri_result_str += str(label[0]) + '-' + str(label[1]) + ' '

        writer.writerow({'Type': "CUE-private", 'Spans': pri_result_str})   



if __name__ == "__main__":
    main()