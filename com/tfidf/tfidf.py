# encoding: utf-8
from __future__ import generators
from collections import Counter
import pickle
import os
import sys
import spacy
import re
nlp = spacy.load('en')


import numpy as np
import marisa_trie
import nltk
from nltk.stem.snowball import SnowballStemmer
import heapq
import itertools

reload(sys)  # Reload does the trick!
sys.setdefaultencoding('UTF8')
from sklearn.feature_extraction.text import TfidfVectorizer
IS_STEM=True
REMOVE_STOPWORDS=True

stemmer = SnowballStemmer("english", ignore_stopwords=True)

def myownstem(word):
    stemword = word
    if IS_STEM:
        stemword = stemmer.stem(word)
    return stemword

def multiwordstem(word_list ):
    for i in range(len(word_list)):
        word_list[i] = myownstem(word_list[i])
    return ' '.join(word_list)

class KeywordList:
    def __init__(self, name):
        self.name = name
        self.wordlist = WORDLIST_PATH[name]
        print name
        self.triepath = TRIE_CACHE_DIR+name+'_trie_dict.cache'
        self.trie = self.load_trie(self.triepath)

    def load_trie(self, trie_cache_file):
        '''
        Load a prebuilt tree from file or create a new one
        :return:
        '''
        trie = None

        if os.path.isfile(trie_cache_file):
            print('Start loading trie from %s' % trie_cache_file)
            with open(trie_cache_file, 'rb') as f:
                trie = pickle.load(f)
        else:
            print('Trie not found, creating %s' % trie_cache_file)
            count = 0
            # transtable = str.maketrans('', '', string.punctuation)
            printable = set(string.printable)
            #print dict_files
            listwords = []
            for dict_file in dict_files:
                file = open(dict_file, 'r')
                for line in file:
                    count+=1
                    if count % 10000==0:
                        print(count)
                    # if is stopword, pass
                    line = line.lower().strip()
                    if (line in stopwords):
                        print(line)
                        continue
                    tokens = line.lower().split()

                    # do stemming and remove punctuations
                    if IS_STEM:
                        tokens = [ stemmer.stem(filter(lambda x:x in printable, token)) if len(token) > 2 else token for token in tokens ]

                    #print(tokens)
                    if(len(tokens)>0):
                        listwords.append(tokens)
            trie = MyTrie(listwords)
            #trie.scan(["hello"])
            with open(trie_cache_file, 'wb') as f:
                pickle.dump(trie, f)
        return trie

def KnuthMorrisPratt(text, pattern):

    '''Yields all starting positions of copies of the pattern in the text.
Calling conventions are similar to string.find, but its arguments can be
lists or iterators, not just strings, it returns all matches, not just
the first one, and it does not need the whole text in memory at once.
Whenever it yields, it will have read the text exactly up to and including
the match that caused the yield.'''

    # allow indexing into pattern and protect against change during yield
    pattern = list(pattern)

    # build table of shift amounts
    shifts = [1] * (len(pattern) + 1)
    shift = 1
    for pos in range(len(pattern)):
        while shift <= pos and pattern[pos] != pattern[pos-shift]:
            shift += shifts[pos-shift]
        shifts[pos+1] = shift

    # do the actual search
    startPos = 0
    matchLen = 0
    for c in text:
        while matchLen == len(pattern) or \
              matchLen >= 0 and pattern[matchLen] != c:
            startPos += shifts[matchLen]
            matchLen -= shifts[matchLen]
        matchLen += 1
        if matchLen == len(pattern):
            yield startPos

__author__ = 'Memray'
'''
A self-implenmented trie keyword matcher
'''


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True

import string

# def isEnglish(s):
#     return s.translate(None, string.punctuation).isalnum()

def load_stopwords():

    if not REMOVE_STOPWORDS :
        return set()

    print (os.getcwd())
    STOPWORD_PATH = 'data/stopword/stopword_min.txt'
    dict = set()
    file = open(STOPWORD_PATH, 'r')
    for line in file:
        dict.add(line.lower().strip())
    return dict

stopwords = load_stopwords()





def isInChunk(word,chunklist):
    wordlist = word.split(" ")
    if word in chunklist:
        return True
    if len(wordlist) > 1:
        for chunk in chunklist:
            listchunk = chunk.split(" ")
            for s in KnuthMorrisPratt(listchunk,wordlist):
                return True
    return False

def isAlreadyPresent(word,presentlist):
    # print(presentlist)
    # print(word)
    for chunk in presentlist:
        listchunk = chunk[0].split(" ")
        for s in KnuthMorrisPratt(listchunk,word.split(" ")):
            # print(word)
            return True
    return False



class MyTrie:
    """
    Implement a static trie with  search, and startsWith methods.
    """
    def __init__(self,words):
        newlist = self.maketrie(words)
        self.nodes = marisa_trie.Trie(newlist)

    # Inserts a phrase into the trie.
    def maketrie(self, words):
        makelist = []
        for word  in words:
            current_word = ' '.join(word)
            if IS_STEM:
                current_word = multiwordstem(word)
            makelist.append(current_word)
        return makelist

    # Returns if the word is in the trie.
    def search(self, words):
        if( words in self.nodes ):
            #print words
            return True
        else:
            return False

    # Scan a sentence and find any ngram that appears in the sentence
    def scan(self, sentence, min_length=1, max_length=3):
        keyword_list = []
        sentence = sentence.lower().translate(None,string.punctuation).decode('utf8')
        original_tokens = sentence.split()
        tokens = sentence.split()
        if IS_STEM:
            original_tokens = [myownstem(token) for token in original_tokens]

        tokens = [word for word in original_tokens if word not in stopwords]
        ngrams = []
        for i in range(min_length, max_length+1):
            ngrams += nltk.ngrams(tokens, i)
        #print ngrams

        for ngram in ngrams:
            #print ' '.join(ngram)
            if(self.search(' '.join(ngram))):
                keyword_list.append(' '.join(ngram))

        return keyword_list

    def ngram_stem(self,str):
        tokens = []
        for token in str.split():
            tokens.append(myownstem(token))
        return ' '.join(tokens)


from nltk.tokenize import sent_tokenize
class Document:
    def __init__(self, *args, **kwargs):
        self.sentences = []
        self.npchunks = []

        if len(args)==1:
            line = args[0]
            self.id = line[:line.index('\t')]
            self.text = line[line.index('\t')+1:].lower()
        elif len(args) ==2:
            self.id = args[0]
            self.text = args[1]
        elif len(args) ==4:
            self.id = args[0]
            self.title=args[1]
            self.concepts=args[2]
            self.text = args[3]
        elif len(args) ==5:
            self.id = args[0]
            self.title=args[1]
            self.concepts=args[2]
            self.wikipediafiles=args[3]
            self.text = args[4]


        sen_list = sent_tokenize(self.text.decode('utf-8', 'ignore'))

        for sen in sen_list:
            self.sentences.append(sen)
            # print(sen)
        self.no_sent = len(self.sentences)

    def __str__(self):
        return '%s\t%s' % (self.id, self.text)

# IR_CORPUS = 'data/keyphrase/textbook/final_merged_mir_iir.csv'
IR_CORPUS = 'data/keyphrase/textbook/final_merged.csv'


def load_document_k(path):
    print('Start loading documents from %s' % path)
    doc_list = []
    file = open(path, 'r')
    import csv

    with file as tsv:
        tsvin = csv.reader(file, delimiter=',')
        for row in tsvin:
            # print(row[0])
            doc = Document(row[0],row[1].strip(),row[2].strip(),row[3].strip(),row[4].strip())

            # print(row[4])
            doc_list.append(doc)
    return doc_list

def load_document(path):
    print('Start loading documents from %s' % path)
    doc_list = []
    file = open(path, 'r')
    import csv

    with file as tsv:
        tsvin = csv.reader(file, delimiter=',')
        for row in tsvin:
            # print(row[0])
            doc = Document(row[0],row[1].strip())

            # print(row[4])
            doc_list.append(doc)
    return doc_list

def getGlobalngrams(grams,min_df,corpus,threshold):
    ncorpus = [myownstem(x) for x in corpus]
    tf = TfidfVectorizer(analyzer='word', ngram_range=grams, stop_words=stopwords)

    tfidf_matrix = tf.fit_transform(ncorpus)
    feature_names = tf.get_feature_names()
    doc = tfidf_matrix.todense()
    temptokens = zip(doc.tolist()[0], itertools.count())
    temptokens = [(x, y) for (x, y) in temptokens if x > threshold]
    tokindex = heapq.nlargest(len(temptokens), temptokens)
    global1grams = dict([(feature_names[y],x) for (x, y) in tokindex ])
    topindex = [ (feature_names[y],x)  for (x,y) in tokindex ]
    # print(global1grams)
    return  global1grams,topindex



def extract_high_tfidf_words( documents, top_k=200, ngram=(1,1), OUTPUT_FOL='TFIDF',is_global=True):
    '''
    Return the top K 1-gram terms according to TF-IDF
    Load corpus and convert to Dictionary and Corpus of gensim
    :param corpus_path
    :param num_feature, indicate how many terms you wanna retain, not useful now
    :return:
    '''

    if not os.path.exists(OUTPUT_DIR + OUTPUT_FOL):
        os.makedirs(OUTPUT_DIR + OUTPUT_FOL)

    texts = [[[re.sub(r'\W+|\d+', '', word.strip()) for word in sen.split() ]
             for sen in document.sentences] for document in documents]



    stemDictionary = {}
    # remove words that appear only once
    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for sen in text:
            sen = re.sub("[ ]{1,}",r' ',' '.join(sen)).lower().split()
            for token in sen:
                stemtoken = myownstem(token)
                # stemtoken = token
                frequency[stemtoken] += 1
                if(stemtoken not in stemDictionary):
                    stemDictionary[stemtoken] = set()
                stemDictionary[stemtoken].add(token)

    print("stem dictionary length",len(stemDictionary.keys()))

    texts = [[[myownstem(token) for token in sen if (frequency[myownstem(token)]  > 3)] for sen in text]
             for text in texts]


             #### Create Scikitlearn corpus
    top_k_list = {}

    corpus = []
    singlecorpus = ""
    for text in texts:
        tempcor = ""
        for sen in text:
            tempcor += ' '+' '.join(sen)
        tempcor = re.sub("[ ]{1,}",r' ',tempcor) + ". ."
        singlecorpus += ' ' + tempcor
        corpus.append(tempcor)
    singlecorpus = [re.sub("[ ]{1,}", r' ', singlecorpus)]

    top_g1gram ={}
    top_g2gram = {}
    top_g3gram = {}
    tokindex1g = []
    tokindex3g=[]
    tokindex2g = []
    if is_global:

        top_g1gram,tokindex1g = getGlobalngrams(grams=(1,1),min_df=2,threshold=0.01,corpus=singlecorpus)
        top_g2gram,tokindex2g = getGlobalngrams(grams=(2,2),min_df=2,threshold=0.005,corpus=singlecorpus)
        top_g3gram,tokindex3g = getGlobalngrams(grams=(3,3),min_df=2,threshold=0.005,corpus=singlecorpus)


    tf = TfidfVectorizer(analyzer='word', ngram_range=ngram,stop_words=stopwords,min_df=2,max_df=1000)
    tfidf_matrix = tf.fit_transform(corpus)
    feature_names = tf.get_feature_names()

    doc_id=0

    for doc in tfidf_matrix.todense():
        temptokens = zip(doc.tolist()[0], itertools.count())
        temptokens1=[]
        for (x, y) in temptokens:
            stemy = feature_names[y]
            if x > 0.0:
                z=x
                if is_global:
                    if( stemy in top_g3gram ):
                        z += top_g3gram[stemy]
                    elif(stemy in top_g2gram):
                        z += top_g2gram[stemy]
                    elif (stemy in top_g1gram):
                        z += top_g1gram[stemy]
                temptokens1.append((z,x,y))

        # temptokens = [(x,y)  for (x,y) in temptokens if x > 0.001]
        tokindex = heapq.nlargest(len(temptokens1), temptokens1)
        # print(temptokens1)

        top_k_list[documents[doc_id].id] = []
        for (x, y,z) in tokindex:
            if len(feature_names[z].decode('utf-8')) > 2 :
                top_k_list[documents[doc_id].id].append((feature_names[z],x,y) )
        doc_id += 1


    output_file = open(TOP_TFIDF, 'w')

    for doc in documents:

        # output_file.write('{0}\t{1}\n'.format(doc.id, ','.join(top_k_list[doc.id])))
        f = open(OUTPUT_DIR + OUTPUT_FOL + "/" + doc.id.replace("\\","_") + ".txt.phrases", 'w')
        writeformat = [ str(x)+","+str(round(y,4))+","+str(round(z,4)) for (x,y,z) in top_k_list[doc.id]]
        f.write('\n'.join(writeformat[0:top_k]))
        # f.write('\n'.join(top_k_list[doc.id][0:top_k]))
        f.write('\n')
        f.close()

    output_file.close()



if __name__=='__main__':

    documents = load_document(IR_CORPUS)
    extract_high_tfidf_words(documents=documentsl, top_k=20, ngram=(1, 1), OUTPUT_FOL='gTFIDF1120', is_global=False)


