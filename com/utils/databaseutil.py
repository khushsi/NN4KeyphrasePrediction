# encoding: utf-8
from __future__ import generators

import os
import re
import sys
import spacy
import nltk
from nltk.stem.snowball import SnowballStemmer



nlp = spacy.load('en')
reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')


data_folder = '/Users/khushsi/Downloads/data/'

training_file='HulthTraining.csv'
validation_file='HulthValidation.csv'
test_file='HulthTest.csv'

def myownstem(word):
    stemword = word
    stemword = stemmer.stem(word)
    return stemword


__author__ = 'KThaker'
'''
A self-implenmented trie keyword matcher
'''

OUTPUT_DIR = 'src/keyphrase_output_199/'

stemmer = SnowballStemmer("english", ignore_stopwords=True)


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

def load_stopwords():
    print(os.getcwd())
    STOPWORD_PATH = data_folder+ 'stopword_en.txt'
    dict = set()
    file = open(STOPWORD_PATH, 'r')
    for line in file:
        dict.add(line.lower().strip())
    return dict


stopwords = load_stopwords()


def processText(itext,IS_Stem=True,IS_Stopwords=False):
    text =  itext.lower().decode('utf-8', 'ignore').split(" ")
    ptext  = [re.sub(r'\W+|\d+', '', tok)for tok in text]

    if IS_Stem:
        ptext = [stemmer.stem(tok) for tok in text]

    if IS_Stopwords:
        ptext = [tok for tok in text if tok not in stopwords]

    return ptext

def multiwordstem(word_list):
    for i in range(len(word_list)):
        word_list[i] = myownstem(word_list[i])
    return ' '.join(word_list)

class Document:
    def __init__(self, *args, **kwargs):
        self.title = ""
        self.abstract = ""
        self.controlled_keywords = []
        self.uncontrolled_keywords = []

        if len(args) > 1:
            self.id = args[0]
            self.title=args[1]
            self.abstract=args[2]
            self.controlled_keywords=args[3].split(",")
            self.uncontrolled_keywords = args[4].split(",")

def load_document(path):
    print('Start loading documents from %s' % path)
    doc_list = []
    file = open(path, 'r')
    import csv

    with file as tsv:
        tsvin = csv.reader(file, delimiter=',')
        next(tsvin)
        for row in tsvin:
            doc = Document(row[0], row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip())
            doc_list.append(doc)
    return doc_list

def load_document_json(path):
    print('Start loading documents from %s' % path)
    doc_list = []
    file = open(path, 'r')
    import json

    pfile = open(path)
    ic = 0

    for line in pfile.readlines():
        text = ""
        pubs = json.loads(line)
        if "abstract" in pubs:
            text = pubs["abstract"]
            title = pubs["title"]
            doc = Document(ic, title,text,"","")
            doc_list.append(doc)
            ic += 1

    return doc_list





