# README

## Environment
* #### Put all code files and doc_dev, doc_test directories in the   same folder and make sure the Python version is 2.7
* #### Downlaod nltk data
	Open python 2.7 in terminal then input:
	<pre><code>
	\>\>\> import nltk
	\>\>\> nltk.download()
	</code></pre>

* #### Install textblob package
	<pre><code>
	$ pip install -U textblob
	$ python -m textblob.download_corpora
	</code></pre>
* #### Intall GeoText package
	<pre><code>$ easy_install geotext</code></pre>
* #### Download Location_Extraction package
	https://pypi.python.org/pypi/location-extractor
	
* #### Install sklearn package
	<pre><code>$ pip install -U scikit-learn</code></pre>
* #### Download StanfordNERTagger relative 	
	In the current folder and get "english.all.3class.distsim.crf.ser.gz"
	
	<pre><code>$ git clone https://github.com/Berico-Technologies/CLAVIN-NERD/blob/master/src/main/resources/models/english.all.3class.caseless.distsim.crf.ser.gz</code></pre>
	
	Download "stanford-ner.jar" in the same folder: 
	http://www.java2s.com/Code/Jar/s/Downloadstanfordnerjar.htm

## Run Code
*	#### Baseline
	Run baseline.py and answers for dev will be in baseline_answer_dev.txt 
	<pre><code>$ python baseline.py</code></pre>	
*	#### NER model
	Run ner.py and answers will be in ner_answer_dev.txt and ner_anser_test.txt
	<pre><code>$ python ner.py</code></pre>
	
*	#### Logistic regression model
	Run logistics.py and answers will be in logistics_answer_dev.txt and logistics_anser_test.txt
	<pre><code>$ python logistics.py</code></pre>