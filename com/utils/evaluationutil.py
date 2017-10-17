from __future__ import division
import Queue as Q
from scipy import spatial
from com.utils.databaseutil import processText


def getTopNSimilar(N=10, document="", all_vectors={}):
    topnq =  Q.PriorityQueue()
    topn = []

    for keyword  in all_vectors.keys():
        topnq.put((spatial.distance.cosine(document,all_vectors[keyword]),keyword))

    for i in range(N):
        v1 = topnq.get(0)
        topn.append(v1[1])

    return topn


def getPrecisionRecall(true_set,pred_set):
    precision = 0
    recall = 0

    count = 0

    true_set = [ ' '.join(processText(truev)) for truev in true_set]
    pred_set = [' '.join(processText(predv)) for predv in pred_set]

    for truev in true_set:
        if(truev in pred_set):
            count = count + 1

    precision = count/len(pred_set)
    if(precision > 1):
        precision = 1

    recall = count / len(true_set)



    return(precision,recall)


