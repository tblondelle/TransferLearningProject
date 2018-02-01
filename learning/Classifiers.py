import time
import numpy as np

from sklearn.naive_bayes import GaussianNB    
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
    




class BaseClassifier():
    def __init__(self):
        self.dct = {
                #'Naive Bayes': GaussianNB(),
                #'CART':DecisionTreeClassifier(criterion='gini', splitter='best'),
                #'Id3':DecisionTreeClassifier(criterion='entropy', splitter='best'),
                #'Decision stump':DecisionTreeClassifier(splitter='best', max_depth = 1),
                ###'Multilayer Perceptron':MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', learning_rate='invscaling'),
                #'KNN':KNeighborsClassifier(n_neighbors=50),
                'TreeBagging':BaggingClassifier(n_estimators = 75),
                #'AdaBoost':AdaBoostClassifier(n_estimators = 15),
                #'Random Forest':RandomForestClassifier(n_estimators = 25) 
                }  # dictionnaire des classifieurs que l'on va utiliser
        self.successes = {}  # performances de chacun des classifieurs
                            # sera calculée plus tard  



    def train(self,X,Y):
        # Entrées :
            # X = numpy array (N_instances,N_features)
            # Y = numpy array (N_instances)
        # Sorties :
            # None
        limit = (9*X.shape[0])//10
        
        X_train,Y_train = X[:limit,:],Y[:limit]
        
        X_val,Y_val= X[limit:,:],Y[limit:]  # sert à calculer les performances
        
        for clf_name in self.dct:
            start = time.time()
            clf = self.dct[clf_name]
            clf.fit(X_train,Y_train)
            preds = clf.predict(X_val)
            
            self.successes[clf_name] = np.mean(preds == Y_val)
            
            print(clf_name,'\t\t\t',time.time()-start)  # on affiche le temps mis pour traiter N lignes
            
    def predict(self,X):
        # Entrées :
            # X = numpy array (N_instances,N_features)
        # Sorties :
            # Y = numpy array (N_instances)
        # renvoie les prédictions du classifieur à partir des prédictions de chacun des classifieurs de base,
        # la décision est rendue après un vote pondéré par l'efficacité de chacun des classifieurs
            
        probas = np.zeros((X.shape[0],))
        
        for name in self.dct:
            clf = self.dct[name]
            probas += clf.predict(X)#*self.successes[name]
        
        probas /= len(self.dct)  #sum([self.successes[name]  for name in self.successes ])
        classes = np.array([1 if proba > 0.5 else 0 for proba in probas])
        return classes
    
    
    
    
    
     

from itertools import islice

filename = "../../data/data_books_cleaned/books_aa.txt"
N = 1000  # nombre de lignes à examiner

def tokenize(textList):
    # Entrées :
        # textList : liste de strings de taille N
    # Sorties :
        # X : numpy array de taille Nx100
    countvectorizer = CountVectorizer(ngram_range=(1,2))    
    X_token = countvectorizer.fit_transform(textList)
    X_token = X_token.toarray()
        
    # réduction de dimension
    truncatedsvd = TruncatedSVD(n_components=100)
    X_reduced_dim = truncatedsvd.fit_transform(X_token)
    
    return(X_reduced_dim)

"""
# Option 1 : rien changé

with open(filename) as myfile:
    head = list(islice(myfile,N))

head = [line.split('\t') for line in head]

labels = [line[0] for line in head]
X = [line[1] for line in head]


X_token = tokenize(X)

labels_bin = [ 0 if label in ['Negative','Neutral'] else 1 for label in labels]  
labels_bin = np.array(labels_bin)

print(np.mean(labels_bin))

N = len(head)//2

X_train = X_token[:N,:]    
X_test = X_token[N:,:]

Y_train = labels_bin[:N]    
Y_test = labels_bin[N:]

C = BaseClassifier()
C.train(X_train,Y_train)
Y_pred = C.predict(X_test)



print(Y_pred[:10])
print(Y_test[:10])
print(np.mean([ Y_test]))
print(np.mean([Y_pred]))
print('%success',np.mean([Y_pred == Y_test]))
"""



# Option 2: probas équitables sur les 2 ensembles

with open(filename) as myfile:
    total_head = list(islice(myfile,N))

total_head = [line.split('\t') for line in total_head]



N = len(total_head)
head = total_head[:(3*N)//4]
head_test = total_head[(3*N)//4:]


I_keep = []
for i in range(len(head)):
    label = (head[i][0])
    if label in ['Negative','Neutral'] :
        I_keep.append(i)

N_small = len(I_keep)
i = 0
while len(I_keep)<2*N_small:
    label = (head[i][0])
    if label in ['Positive'] :
        I_keep.append(i)
    i+=1


from random import shuffle
shuffle(I_keep)

head = [head[i] for i in I_keep]


labels = [line[0] for line in head]
X = [line[1] for line in head]

X_token = tokenize(X)

labels_bin = [ 0 if label in ['Negative','Neutral'] else 1 for label in labels]  
labels_bin = np.array(labels_bin)


N_sep = len(labels_bin)//2  # separation

X_train = X_token[:N_sep,:]    
X_test = X_token[N_sep:,:]

Y_train = labels_bin[:N_sep]    
Y_test = labels_bin[N_sep:]

C = BaseClassifier()
C.train(X_train,Y_train)
Y_pred = C.predict(X_test)


print(Y_pred[:10])
print(Y_test[:10])
print(np.mean([ Y_test]))
print(np.mean([Y_pred]))
print('%success',np.mean([Y_pred == Y_test]))





"""
# Option 3 : probas équitables sur l'ensemble d'apprentissage, mais pas de test

# Un échec

with open(filename) as myfile:
    total_head = list(islice(myfile,N))

head = total_head[:1500]
head_test = total_head[1500:]



I_keep = []
for i in range(len(head)):
    n = int(head[i][0])
    if n < 4:
        I_keep.append(i)

N_small = len(I_keep)
i = 0
while len(I_keep)<2*N_small:
    n = int(head[i][0])
    if n >= 4:
        I_keep.append(i)
    i+=1

head = [head[i] for i in I_keep]


labels_training = [line[0] for line in head]
X_training = [line[2:] for line in head]


X_token_tr = tokenize(X_training)

labels_bin_tr = [ 0 if label in ['1','2','3'] else 1 for label in labels_training]  
labels_bin_tr = np.array(labels_bin_tr)




labels_learning = [line[0] for line in head_test]
X_learning = [line[2:] for line in head_test]


X_token_te = tokenize(X_learning)

labels_bin_te = [ 0 if label in ['1','2','3'] else 1 for label in labels_learning]  
labels_bin_te = np.array(labels_bin_te)




print(np.mean(labels_bin_te))


C = BaseClassifier()

C.train(X_token_tr,labels_bin_tr)
Y_pred = C.predict(X_token_te)

print(Y_pred[:10])
print(labels_bin_te[:10])

print(np.mean([Y_pred == labels_bin_te]))

Y_test = labels_bin_te
"""





"""
TP = len([i for i in range(len(Y_pred)) if Y_pred[i] == 1 and Y_test[i] == 0])
TN = len([i for i in range(len(Y_pred)) if Y_pred[i] == -1 and Y_test[i] == 1])
FP = len([i for i in range(len(Y_pred)) if Y_pred[i] == 1 and Y_test[i] == 1])
FN = len([i for i in range(len(Y_pred)) if Y_pred[i] == -1 and Y_test[i] == 0])


print('')
print('Confusion matrix :')
print()
print(" \t\t Actual class")
print(" \t\t Good \t Bad")
print("Predicted Good \t {} \t {}".format(TP,FN))
print("          Bad \t {} \t {}".format(FP,TN))

"""
