# TODO: 1. read files, parse files
#       2. feeding data to models
#       3. getting predictions from models and write to files
import os
from preprocessor import process, generate_path
from baseline_model import baseline_model

# INPUT: FOLDER NAME
folder = "test-private"  
directory = generate_path(folder)

def main():
    for root, dirs, filenames in os.walk(directory):
        for f in filenames:
            temp = process(directory + f)
            print "File: {}".format(f)
            print temp

if __name__ == "__main__":
    main()