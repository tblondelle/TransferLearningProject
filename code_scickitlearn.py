# -*- coding:utf-8 -*-
import time
from sklearn.naive_bayes import GaussianNB    
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier


from  sklearn.model_selection import KFold,cross_val_score,cross_val_predict
from sklearn.metrics import recall_score, roc_auc_score 



Clf_dict = {}

# Classifieurs :
# Bayésien naïf
Clf_dict['Naive Bayes'] = GaussianNB()

# arbre CART 
Clf_dict['CART'] = DecisionTreeClassifier(criterion='gini', splitter='best', min_samples_split =12)

# arbre Id3
Clf_dict['Id3'] = DecisionTreeClassifier(criterion='entropy', splitter='best', min_samples_split =12)

# decision stump
Clf_dict['Decision stump'] = DecisionTreeClassifier(splitter='best', max_depth = 1)

# Multilayer perceptron
Clf_dict['Multilayer Perceptron'] = MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', learning_rate='invscaling')

# nearest neighbors
Clf_dict['KNN'] = KNeighborsClassifier(n_neighbors=10)

# treebagging
Clf_dict['TreeBagging'] = BaggingClassifier(n_estimators = 75)

# AdaBoost 
Clf_dict['AdaBoost'] = AdaBoostClassifier(n_estimators = 15)

# Random Forest
Clf_dict['Random Forest'] = RandomForestClassifier(n_estimators = 25)



def run_classifiers(D,X,Y):
    kf10 = KFold(n_splits=10, shuffle=False, random_state=None)
    kf5 = KFold(n_splits=5, shuffle=False, random_state=None)
    print("    Name"+26*' '+"Accuracy"+11*" "+"AUC"+8*" "+"Recall"+4*" "+"Running time")
    
    for name in D:
        start_time = time.time()

	# Accuracy et AUROC : 10-fold XV        
        classifier = D[name]
        accuracy = cross_val_score(classifier, X, Y, cv=kf10)      
        
        predicted_Y = cross_val_predict(classifier, X, Y, cv=kf10)
        AUC = roc_auc_score(Y,predicted_Y)

	# on nous demandait de calculer le rappel avec 5-fold XV seulement
        predicted_Y = cross_val_predict(classifier, X, Y, cv=kf5)  # avec 5-fold XV
        recall = recall_score(Y,predicted_Y)  
        
        running_time = (time.time()-start_time)
        
        # Affichage des résultats
        Accuracy_string = "{0:.3f} +/- {1:.3f}".format(np.mean(accuracy), np.std(accuracy))
        AUC_string = "{0:.3f}".format(AUC)
        recall_string = "{0:.3f}".format(recall)
        time_string = "{0:.2f}s".format(running_time)
        print(name+' '*(30-len(name))+Accuracy_string+' '*(7)+AUC_string+' '*(7)+recall_string+' '*(7)+time_string)

            


import numpy as np
import pandas as pd
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


sms_data = pd.read_csv('SMSSpamCollection.data', sep='\t')

X_sms = sms_data.ix[:, 1].values
labels_sms = sms_data.ix[:, 0].values

# transformation spam --> 1 / ham --> 0
labels_sms_bin = np.copy(labels_sms)
labels_sms_bin[labels_sms_bin == 'ham'] = 0
labels_sms_bin[labels_sms_bin == 'spam'] = 1
labels_sms_bin = list(labels_sms_bin)

countvectorizer = CountVectorizer(ngram_range=(2, 2))
# ngram_range=(2, 2) permet d'obtenir les bigrammes : les couples de mots qui se suivent
# On obtient des vecteurs de dimension 5500 env. => il faut réduire cette dim
X_sms_token = countvectorizer.fit_transform(X_sms)


X_sms_token = X_sms_token.toarray()
#run_classifiers(Clf_dict,X_sms_token,labels_sms_bin) # ne fonctionne pas car dim. trop élevée


tfidfvectorizer = TfidfVectorizer()
X_sms_token_2 = tfidfvectorizer.fit_transform(X_sms)
#run_classifiers(Clf_dict,X_sms_token_2,labels_sms_bin) # ne fonctionne pas car dim. trop élevée


# réduction de dimension
# c'est la seule qui marche
truncatedsvd = TruncatedSVD(n_components=100)

X_sms_token_3 = truncatedsvd.fit_transform(X_sms_token_2)


run_classifiers(Clf_dict,X_sms_token_3,labels_sms_bin)
