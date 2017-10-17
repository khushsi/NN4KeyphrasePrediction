from com.utils.databaseutil import load_document,processText,load_document_json
from gensim import corpora, models
import gensim
from com.utils.evaluationutil import  getTopNSimilar,getPrecisionRecall
import numpy as np
import pickle
import json
import os

data_folder = 'data/'
output_folder = 'output/'

trainingfile = 'HulthTraining.csv'
validationfile = 'HulthValidation.csv'
testfile = 'SemevalTest.csv'

print("testfile",testfile)
microsoft_paper_json = 'mag_fos.txt'

IS_STEMMING = True

keywords_dist = {}

tok_set = []
data_training = load_document(data_folder+trainingfile)
data_validation = load_document(data_folder+validationfile)
data_testing = load_document(data_folder+testfile)
data_testing = data_testing

#Combine training and validation
data_training = data_training + data_validation

json_data = []

if(not os.path.isfile('save_json')):
    data_json = load_document_json(data_folder+microsoft_paper_json)
    pickle.dump(data_json,open('save_json','wb'))

data_json = pickle.load(open('save_json','rb'))
data_training = data_training + data_validation + data_json
keywords = []

for doc in data_testing:
    keywords += doc.controlled_keywords
    keywords += doc.uncontrolled_keywords

keywords = list(set(keywords))

TOP_N = 10
K = 200


for doc in data_training:
    tok_set.append(doc.title + ". "+doc.abstract)

tok_set = [processText(tok) for tok in tok_set]
if not os.path.isfile('corpus.pk'):
    print("create dictionary")
    dictionary = corpora.Dictionary(tok_set)
    corpus = [dictionary.doc2bow(text) for text in tok_set]
    pickle.dump([dictionary,corpus], open("corpus.pk",'wb'))

dictionary,corpus = pickle.load(open('corpus.pk','rb'))
print("No of topics" , K)


print("## Training LDA")

if not os.path.isfile('ldamodle'+str(K)):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=K, id2word=dictionary)
    pickle.dump(ldamodel,open('ldamodle'+str(K),'wb'))
ldamodel = pickle.load(open('ldamodle'+str(K),'rb'))

vector_test = {}
vector_keywords = {}
predict_doc_keywords = {}

print("# Predicting Keywords Topic")

for keyword in keywords:
    bow = dictionary.doc2bow(processText(keyword))
    predict = ldamodel[bow]
    t_predict_vector = np.zeros(K)
    for (x, y) in predict:
        t_predict_vector[x] = y

    vector_keywords[keyword] = list(t_predict_vector)

print("# Predicting testset Topic")
for doc in data_testing:

    bow = dictionary.doc2bow(processText(doc.title + ". "+doc.abstract))
    predict = ldamodel[bow]
    t_predict_vector = np.zeros(K)
    for (x, y) in predict:
        t_predict_vector[x] = y
    vector_test[doc.id] = list(t_predict_vector)
    predict_doc_keywords[doc.id] = getTopNSimilar(N=TOP_N,document=vector_test[doc.id],all_vectors=vector_keywords)


predPrec = []
predRec = []
predFscore = []
for doc in data_testing:
    prec,recall = getPrecisionRecall(doc.uncontrolled_keywords,predict_doc_keywords[doc.id])
    predPrec.append(prec)
    predRec.append(recall)
    if(prec+recall > 0):
        predFscore.append(((2* (prec * recall)) / (prec+recall)))
    else:
        predFscore.append(0)

print("Precision" , np.mean(predPrec))
print("Recall" , np.mean(predRec))
print("F1" , np.mean(predFscore))