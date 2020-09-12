import nltk
#nltk.download('brown')
from nltk.corpus import brown
import numpy as np
from collections import Counter
from collections import defaultdict
from math import log
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from nltk import word_tokenize

#For dividing into 5 folds and store in train,test 
def K_fold(sentences,k_fold=5):
    LS=len(sentences)
    length_of_fold=LS/k_fold
    
    divisions=[]
    part=[]
    for i in range(LS):
        part.append(sentences[i])
        if (i+1)%length_of_fold == 0:
            divisions.append(part)
            part=[]
    
    train=[]
    test=[]
    for i in range(k_fold):
        test.append(divisions[i])
        set4=[]
        for j in range(k_fold):
            if i!=j:
                set4.extend(divisions[j])
        train.append(set4)
    return (train,test)

#To learn mapping of universal tags to numbers and vice versa
def mapping(sentences): 
    Universal_tags=set([pair[1] for pair in [tup for sent in sentences for tup in sent]])
    #Create a mapping function for tags<--->numbers
    mapping_tag_to_num=defaultdict(int)
    mapping_num_to_tag=defaultdict(str)
    k=0
    for x in Universal_tags:
        mapping_tag_to_num[x]=k
        mapping_num_to_tag[k]=x
        k+=1
    return (mapping_num_to_tag,mapping_tag_to_num)


def preprocess(sentences,words):
    #Create Vocabulary and calculate size of vocab in V
    first=[f for f,s in words]
    V=set(first)
    
    #Calculate Start Probabilities and store in startProb
    LL=[]
    no_of_sentences=len(sentences)
    for i in sentences:
        LL.append(i[0][1])
    startprob=Counter(LL)
    for i in startprob.keys():
        startprob[i]=(startprob[i])/(no_of_sentences)
    
    tokens,taged = zip(*words)
    total=len(words)
    
    #Calculate Count(w,t) and store in tokenTags
    wordcount=Counter(tokens)
    tokenTags=defaultdict(Counter)
    for token, tag in words:
        tokenTags[token][tag]+=1
    
    #Calculate Count(t) and store in tagcount
    tagcount=Counter(taged)
    
    #Calculate Count(t1,t2) and store in tagtags
    bgram = nltk.ngrams(taged,2)
    tagtags = defaultdict(Counter)
    for tag1, tag2 in bgram:
        tagtags[tag1][tag2] += 1
    
    return (len(V),startprob,tokenTags,tagcount,tagtags)


def Viterbi(obs,startprob,tagcount,tokentags,tagtags,mapping_tag_to_num,mapping_num_to_tag,V):
    T=len(obs)
    N=len(startprob)
    viterbi=np.zeros((N,T),dtype='float64')
    backpointer=np.zeros((N,T),dtype='float64')
    for s in range(N):
        w=obs[0]
        t=mapping_num_to_tag[s]
        viterbi[s][0]=(startprob[t])*((tokentags[w][t]+1)/(tagcount[t]+V))
        backpointer[s][0]=-1
    for t in range(1,T):
        for s in range(N):
            s1=mapping_num_to_tag[s]
            for sd in range(N):
                sd1=mapping_num_to_tag[sd]
                prob=viterbi[sd][t-1]*((tagtags[sd1][s1]+1)/(tagcount[sd1]+V))*((tokentags[obs[t]][s1]+1)/(tagcount[s1]+V))#Used add one smoothing
                if prob > viterbi[s][t]:
                    viterbi[s][t]=prob
                    backpointer[s][t]=sd
    mini=0
    backy=0
    for s in range(N):
        if viterbi[s][T-1] > mini:
            mini=viterbi[s][T-1]
            backy=s
    ttt=T-1
    answer=[]
    while ttt>=0:
        answer.append(mapping_num_to_tag[backy])
        backy=backpointer[int(backy)][ttt]
        ttt-=1
    answer.reverse()
    return answer


def main():
    #nltk.download('brown')
    sentences=np.array(brown.tagged_sents(tagset='universal'))
    #words=brown.tagged_words(tagset='universal')
    
    # K-fold=5 calculation to get train and test lists of length 5
    k_fold=5
    train,test=K_fold(sentences,k_fold)
    
    #Calculate mapping num_to_tag and vice versa
    mapping_num_to_tag,mapping_tag_to_num=mapping(sentences)
    
    #Some initializations
    Universal_tags=[r for r in mapping_tag_to_num.keys()]
    No_of_tags=len(Universal_tags)
    confusion_matrix=np.zeros((No_of_tags,No_of_tags),dtype='int')
    tag_count_original=np.zeros(No_of_tags,dtype='float32')
    tag_count_predicted=np.zeros(No_of_tags,dtype='float32')
    Final_accuracy=0

    #Used Viterbi function here to calculate outputs and print accuracy
    for index in range(k_fold):
        sentences=train[index]
        words=[tup for sen in sentences for tup in sen]
        V,startprob,tokenTags,tagcount,tagtags=preprocess(train[index],words)
        test_sentences=[]
        test_sentences_tag=[]
        for i in range(len(test[index])):
            senti=[]
            tagi=[]
            for j,k in test[index][i]:
                senti.append(j)
                tagi.append(k)
            test_sentences.append(senti)
            test_sentences_tag.append(tagi)

        predicted_tag=[]
        for i in test_sentences:
            predicted_tag.append(Viterbi(i,startprob,tagcount,tokenTags,tagtags,mapping_tag_to_num,mapping_num_to_tag,V))
        counting=0
        correct=0
        for i in range(len(predicted_tag)):
            counting+=len(predicted_tag[i])
            for j in range(len(predicted_tag[i])):
                original=test_sentences_tag[i][j]
                pre=predicted_tag[i][j]
                confusion_matrix[mapping_tag_to_num[original]][mapping_tag_to_num[pre]]+=1
                tag_count_original[mapping_tag_to_num[original]]+=1
                if original==pre:
                    correct+=1
                    tag_count_predicted[mapping_tag_to_num[original]]+=1
        print("Fold:",(index+1)," accuracy= ",(correct/counting)*100)  
        Final_accuracy+=(correct/counting)*100
    

    print("Final Accuracy= ",Final_accuracy/5)    
    print()
    #Printing confusion matrix and per pos tag accuracy
    print("Printing Per POS tag accuracy")
    for i in range(len(tag_count_original)):
        if tag_count_original[i]!=0:
            #tag_accuracy.append(tag_count_predicted[i]/tag_count_original[i])
            print(mapping_num_to_tag[i]," ",(tag_count_predicted[i]/tag_count_original[i])*100)
        else:
            #tag_accuracy.append(0)
             print(mapping_num_to_tag[i]," ",str(0))

    print()
    print("Printing confusion matrix")
    print(confusion_matrix)
    print()
    print("Printing heatmap of confusion matrix too!!")
    tags_df=pd.DataFrame(confusion_matrix, columns = list(Universal_tags), index = list(Universal_tags))
    plt.figure(figsize=(10,8))
    sns.heatmap(tags_df)
    plt.show()


if __name__ == "__main__":
	confusion_matrix=main()