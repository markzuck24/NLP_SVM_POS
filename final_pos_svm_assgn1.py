# -*- coding: utf-8 -*-


import nltk
from nltk.corpus import brown
from sklearn.svm import SVC
import numpy as np
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.feature_extraction import DictVectorizer
from tqdm import tqdm
from matplotlib import pyplot as plt

nltk.download('brown')
nltk.download('universal_tagset')

total = len(brown.tagged_sents())
tags = 'universal'
train_size = int(total * 0.8)
test_size = total - train_size
train_set = brown.tagged_sents(tagset = tags)[:train_size]
test_set =  brown.tagged_sents(tagset = tags)[-test_size:]

w2v_words = Word2Vec(brown.sents(), size=30)

# tags_list = list(set([tag for (_, tag) in brown.tagged_words(tagset=tags)]))
all_tags = ['DET','NOUN','ADP','ADJ','NUM','X','PRON','PRT','VERB','.','CONJ','ADV']

all_tags.append("<s>")
all_tags.append("</s>")

tags = {'ADJ':0,
 'CONJ':1,
 'NOUN':2,
 '.':3,
 'X':4,
 'NUM':5,
 'ADP':6,
 'VERB':7,
 'PRON':8,
 'DET':9,
 'PRT':10,
 'ADV':11,
 '<s>':12,
 '</s>':13}

def onehot(n, l):
    v = np.zeros(l)
    v[n] = 1
    return v

def features_svm(wid, sen, w2v,tags):
    ft = []
    keys = w2v.wv.vocab.keys()

    words_prev_n = 2
    for i in reversed(range(0,words_prev_n+1)):
        if sen[wid-i][0] not in keys:
            vec = np.zeros(w2v.vector_size)
            ft.append(vec)
        else:
            vec = w2v[sen[wid-i][0]]
            ft.append(vec)
        #print("--->1",w2v.vector_size)

    words_next_n = 2
    for i in range(1,words_next_n+1):
        if sen[wid+i][0] not in keys:
            vec = np.zeros(w2v.vector_size)
            ft.append(vec)
        else:
            vec = w2v[sen[wid+i][0]]
            ft.append(vec)
        #print("--->2",w2v.vector_size)
    tags_prev_n = 2
    for i in reversed(range(1,tags_prev_n+1)):
        tag = sen[wid-i][1]
        vec = onehot(tags.index(tag), len(tags))
        ft.append(vec)
        #print("--->3",OneHotEncoder(tags.index(tag),len(tags)).shape)

    if sen[wid][0][0].isupper():
        ft.append([1])
    else:
        ft.append([0])
    #print("---->4",(np.asarray(features).shape))

    if len(sen[wid][0]) > 4:
        if sen[wid][0][-4:].lower() == 'able' or sen[wid][0][-3:].lower() == 'ful' or sen[wid][0][-4:].lower() == 'less':
            ft.append([1])
        else: 
            ft.append([0])
    else:
        ft.append([0])

    if len(sen[wid][0]) > 5:
        if sen[wid][0][-4:].lower() in ["ship","ness","sion","ment"] or sen[wid][0][-3:].lower() in ["eer","ion","ity"] or sen[wid][0][-2:].lower() in ["er","or","th"]:
            ft.append([1])
        else: 
            ft.append([0])
    else:
        ft.append([0])

    if len(sen[wid][0]) > 5:
        if sen[wid][0][-4:].lower() in ["able","ible","less","ious"] or sen[wid][0][-3:].lower() in ["ant","ary","ful","ous","ive"] or sen[wid][0][-2:].lower() in ["al","ic"]:
            ft.append([1])
        else: 
            ft.append([0])
    else:
        ft.append([0])

    if len(sen[wid][0]) > 5:
        if sen[wid][0][-3:].lower() in ["ing","ize","ise"] or sen[wid][0][-2:].lower() in ["ed","en","er"]:
            ft.append([1])
        else: 
            ft.append([0])
    else:
        ft.append([0])

    if len(sen[wid][0]) > 5:
        if sen[wid][0][-4:].lower() in ["wise","ward"] or sen[wid][0][-2:].lower() in ["ly"]:
            ft.append([1])
        else: 
            ft.append([0])
    else:
        ft.append([0])
    



    if wid ==0:
      ft.append([1])
    else:
        ft.append([0])

    if wid == len(sen)-1:
        ft.append([1])
    else:
        ft.append([0])
      
    if sen[wid][0].upper() == sen[wid][0]:
        ft.append([1])
    else:
        ft.append([0])

    if sen[wid][0].lower() == sen[wid][0]:
        ft.append([1])
    else:
        ft.append([0])

    if '-' in sen[wid][0]:
      ft.append([1])
    else:
        ft.append([0])

    if sen[wid][0].isdigit():
      ft.append([1])
    else:
        ft.append([0])
    
    if sen[wid][0][1:].lower() != sen[wid][0][1:]:
      ft.append([1])
    else:
        ft.append([0])

    #print("---->5",(np.asarray(ft).shape))

    total_feature_list = []
    for sublist in ft:
        for item in sublist:
            total_feature_list.append(item)

        
    #print("---->6",(np.asarray(total_feature_list).shape))
    return total_feature_list

x_text = []
y_test = []
for each_sent in tqdm(test_set):
    each_sent = [("","<s>") , ("","<s>")] + each_sent + [("","</s>"),  ("","</s>")]

    for j in range(2,len(each_sent)-2):
        x_text.append(features_svm(j, each_sent, w2v_words, all_tags))
        y_test.append(tags[each_sent[j][1]])


fet_train = []
y_train = []
for each_sent in tqdm(train_set):
    each_sent = [("","<s>") , ("","<s>")] + each_sent + [("","</s>"), ("","</s>")]

    for j in range(2,len(each_sent)-2):
        fet_train.append(features_svm(j, each_sent, w2v_words, all_tags))
        y_train.append(tags[each_sent[j][1]])

X_train=np.array(fet_train).astype(np.float)
y_train=np.array(y_train)

x_text=np.array(x_text).astype(np.float)
y_test=np.array(y_test)

x_text=np.transpose(x_text)

y_train
X_train=np.transpose(X_train)
print(X_train.shape)

print(x_text.shape)

def vectorized_grad_loss(weights, x_train, y, reg):
    # print("------>")
    # dW = np.zeros(weight.shape)
    loss = 0.0
    delta = 1.0

    train_num = y.shape[0]
    current_label = weights.dot(x_train)
 
    actual_class = current_label[y, range(train_num)] 
    
    diff = current_label - actual_class + delta

    diff = np.maximum(0, diff)
    diff[y, range(train_num)] = 0

    loss = np.sum(diff) / train_num

    loss += 0.5 * reg * np.sum(weights * weights)

    current_label_grad = np.zeros(current_label.shape)

    pos_num = np.sum(diff > 0, axis=0)
    current_label_grad[diff > 0] = 1
    current_label_grad[y, range(train_num)] = -1 * pos_num

    dW = current_label_grad.dot(x_train.T) / train_num + reg * weights
    
    return loss, dW



class classifier:

    def __init__(self):
        self.weights = None 

    def train(self, X, y, gradients='sgd', lr=0.1,reg = 0, num_iters=1000, batch_size=128):
        
        dim, num_train = X.shape
        class_num = np.max(y) + 1
        
        # self.weights = np.random.normal(0,1, size=(class_num, dim))
        self.weights = np.random.randn(class_num, dim) * 0.001

        list_losses = []
        method='sgd'

        for i in range(num_iters):
            if method == 'bgd':
                loss, grad = self.loss_grad(X, y, reg, vectorized)
            else:
                idxs = np.random.choice(num_train, batch_size, replace=True)
                loss, grad = self.loss_grad(X[:, idxs], y[idxs], reg) 
            # loss, grad = self.loss_grad(X, y, reg)
            list_losses.append(loss)

            self.weights -= lr * grad
            
            if i % 100 == 0:
                print ('iteration %d/%d: loss %f' % (i, num_iters, loss) )

        return list_losses

    def predict(self, x_train):
        
        pred_val = self.weights.dot(x_train)
        # if self.__class__.__name__ == 'Logistic':
        #     y = f_x_mat.squeeze() >=0
        # else: 
        y = np.argmax(pred_val, axis=0)

        hx = 1.0 / (1.0 + np.exp(-pred_val))
        hx = hx.squeeze()
        return y, hx

class SVM(classifier):
    def loss_grad(self, X, y, reg):
        return vectorized_grad_loss(self.weights, X, y, reg)

grad_svm = SVM()
losses_sgd = grad_svm.train(X_train, y_train, lr=0.1,reg = 0, num_iters=10000, batch_size=256)
y_train_pred_sgd = grad_svm.predict(X_train)[0]

print(y_train_pred_sgd.shape,y_train_pred_sgd);
print(y_train.shape,y_train)

print ('Training accuracy: %f' % (np.mean(y_train == y_train_pred_sgd)) )
y_val_pred_sgd = grad_svm.predict(x_text)[0]
print ('test accuracy: %f' % (np.mean(y_test == y_val_pred_sgd)) )

from sklearn import metrics
from sklearn.metrics import confusion_matrix
tags_plot = ['NOUN','NUM','CONJ','DET','VERB','PRT','ADV','PRON','ADP','ADJ','.','X']

tags = {0:'ADJ',
 1:'CONJ',
 2:'NOUN',
 3:'.',
 4:'X',
 5:'NUM',
 6:'ADP',
 7:'VERB',
 8:'PRON',
 9:'DET',
 10:'PRT',
 11:'ADV',
 12:'<s>',
 13:'</s>'}


# for label in y_test[:10]:
#   print(label)
y_test_conf_matrix = [tags[label] for label in y_test]
y_val_pred_sgd_conf_matrix = [tags[label] for label in y_val_pred_sgd]

cm = confusion_matrix(y_test_conf_matrix, y_val_pred_sgd_conf_matrix, labels = tags_plot)

import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label',fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)

plot_confusion_matrix(cm, tags_plot,normalize=True)
plt.savefig('SVM_train_sentences.png', bbox_inches="tight", transparent = True)

