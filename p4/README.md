
Dependencies:
nltk package
wget
pip
gensim

instructions:

run word2VecModel.py

switch between the pre-trained model and custom train model by editing
task at the end of the script

task = “train” if you want to train a custom model
task = “pretrained” if you want to use the GoogleNews pre-trained model

Note: running pre-trained model, the script will download the pre-trained file
and it’s 1.65G, so it takes time to download and load. Please be patient.