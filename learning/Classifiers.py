import time
import numpy as np

from sklearn.naive_bayes import GaussianNB    
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier



class BaseClassifier():
    def __init__(self):
        self.dct = {
                'Naive Bayes': GaussianNB(),
                'CART':DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_split =12),
                'Id3':DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split =12),
                'Decision stump':DecisionTreeClassifier(splitter='best', max_depth = 1),
                #'Multilayer Perceptron':MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', learning_rate='invscaling'),
                'KNN':KNeighborsClassifier(n_neighbors=10),
                'TreeBagging':BaggingClassifier(n_estimators = 75),
                'AdaBoost':AdaBoostClassifier(n_estimators = 15),
                'Random Forest':RandomForestClassifier(n_estimators = 25) 
                }
        self.successes = {}

    def train(self,X,Y):
        # Entrées :
            # X = numpy array (N_instances,N_features)
            # Y = numpy array (N_instances)
        # Sorties :
            # None
        limit = (9*X.shape[0])//10
        
        X_train,Y_train = X[:limit,:],Y[:limit]
        
        X_val,Y_val= X[limit:,:],Y[limit:]
        
        for clf_name in self.dct:
            clf = self.dct[clf_name]
            clf.fit(X_train,Y_train)
            preds = clf.predict(X_val)
            self.successes[clf_name] = np.mean(preds == Y_val)
            
            
    def predict(self,X):
        # Entrées :
            # X = numpy array (N_instances,N_features)
        # Sorties :
            # Y = numpy array (N_instances)
            
        probas = np.zeros((X.shape[0],))
        
        for name in self.dct:
            clf = self.dct[name]
            probas += clf.predict(X)*self.successes[name]
        
        probas /= sum([self.successes[name]  for name in self.successes ])
        classes = np.array([1 if proba > 0.5 else 0 for proba in probas])
        return classes
    
    
    
    
    
    
    
    
    
from itertools import islice
from Tokenizers import tokenize

filename = "../data/instruments.txt"
N = 1000


with open(filename) as myfile:
    head = list(islice(myfile,N))


labels = [line[0] for line in head]
X = [line[2:] for line in head]


X_token = tokenize(X)

labels_bin = [ 0 if label in ['1','2','3'] else 1 for label in labels]  
labels_bin = np.array(labels_bin)



X_train = X_token[:500,:]    
X_test = X_token[500:,:]

Y_train = labels_bin[:500]    
Y_test = labels_bin[500:]



C = BaseClassifier()

C.train(X_train,Y_train)
Y_pred = C.predict(X_test)

print(np.mean([Y_pred == Y_test]))







    