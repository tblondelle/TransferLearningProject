# -*- coding: utf-8 -*-
import time
import numpy as np
import os
from random import shuffle, sample, seed

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


seed(8080)
np.random.seed(8080)


def balanceData(data):
    """
    Return a shorter version of data where an equal number of
    negative/neutral and positif lines are returned.
    """

    neg_neutral_indexes, pos_indexes = [], []
    for index, line in enumerate(data):
        label = line[0]
        if label in ['Negative','Neutral'] :
            neg_neutral_indexes.append(index)
        else :
            pos_indexes.append(index)

    small_n = min(len(neg_neutral_indexes), len(pos_indexes))

    all_indexes = neg_neutral_indexes[:small_n] + pos_indexes[:small_n]

    shuffle(all_indexes)

    balancedData = [data[i] for i in all_indexes]

    #print("Proportion of lines kept while balancing data: {}".format(len(balancedData)/len(data)))

    return balancedData


def getData(folder):
    """
    Input:
     - folder: string of the path of a folder containing txt files.
    Output:
     - listdata: list of [Y, X] (e.g. Y = 'Positive', X = "very cool")
    """

    listdata = []

    filenames = os.listdir(folder)
    for filename in filenames:
        print(os.path.join(folder, filename))
        with open(os.path.join(folder, filename), 'r') as f:
            for line in f:
                line2 = line.strip().split('\t')
                if len(line2) == 2:
                    listdata.append(line2)

    return listdata

def binariseLabels(Y):
    Y_bin = []
    for i, label in enumerate(Y):
        if Y[i] in ['Negative', 'Neutral']:
            Y_bin.append(0)
        elif Y[i] == 'Positive':
            Y_bin.append(1)

    return Y_bin

class MetaClassifier():
    def __init__(self, validation_rate=0.1, n_features=150):
        """
        Input:
         - validation_rate: ratio of training data dedicated to validation.
         - n_features: integer for n_features for SVD (/!\ : exponential complexity !!)
        Output:
         - None
        """

        self.name = "sklearn_classifier"

        self.tfidf_vectorizer = None
        self.truncatedsvd = None

        self.n_features = n_features

        self.validation_rate = validation_rate
        self.classifiers = {
            'Naive Bayes': GaussianNB(),
            'CART':DecisionTreeClassifier(criterion='gini', splitter='best'),
            #'Id3':DecisionTreeClassifier(criterion='entropy', splitter='best'),
            'Decision stump':DecisionTreeClassifier(splitter='best', max_depth = 1),
            #'Multilayer Perceptron':MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', learning_rate='invscaling'),
            #'KNN':KNeighborsClassifier(n_neighbors=50),
            'TreeBagging':BaggingClassifier(n_estimators=50),
            'AdaBoost':AdaBoostClassifier(n_estimators=50),
            'Random Forest':RandomForestClassifier(n_estimators=50)
        }  # dictionnaire des classifieurs que l'on va utiliser

        self.successes = {
            'Naive Bayes': 0,
            'CART':0,
            #'Id3':0,
            'Decision stump':0,
            #'Multilayer Perceptron':0,
            #'KNN':0,
            'TreeBagging':0,
            'AdaBoost': 0,
            'Random Forest':0
        }  # performances de chacun des classifieurs



    def train(self, X_source, Y_source, X_dest):
        """
        Based on X and Y, create a TFIDF matrix, then apply
        an SVD. Then train on all classifiers defined in
        the constructor.

        Input:
         - X_source: A list of n sentences from the source space. A sentence is
                one string.
         - Y_source: A list of n labels from the source space. Labels are in {0, 1}
                (negative/positive)
         - X_dest: A list of n sentences from the destination space. A sentence is
                one string.
        Output:
         - None
        """

        print("Tokenisation...")
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
        # to fit the tfidf, we need a list with the same number of elements as
        # each of the lists we will use
        X_chosen = sample(X_source,len(X_source)//2) + sample(X_dest,len(X_dest)//2+1)
        tfidf_vectorizer.fit(X_chosen)
        X_token_both = tfidf_vectorizer.fit_transform(X_source+X_dest)


        truncatedsvd = TruncatedSVD(n_components=self.n_features)
        truncatedsvd.fit(X_token_both)
        X_token = tfidf_vectorizer.transform(X_source)



        X_source = truncatedsvd.transform(X_token)
        """



        print("Tokenisation...")
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
        X_token = tfidf_vectorizer.fit_transform(X_source)

        truncatedsvd = TruncatedSVD(n_components=self.n_features)
        print('train1_______________________________________________________________',len(X_source))

        X_source = truncatedsvd.fit_transform(X_token)
        print('train2_______________________________________________________________',X_source.shape)
        #
"""

        self.tfidf_vectorizer = tfidf_vectorizer
        self.truncatedsvd = truncatedsvd

        print("Training...")

        limit = int((1 - self.validation_rate) * X_source.shape[0])

        X_train, Y_train = X_source[:limit,:], Y_source[:limit] # ne sert qu'à l'entrainement
        X_val, Y_val = X_source[limit:,:], Y_source[limit:]  # validation, sert à calculer les performances


        print("Training: {} lines of data".format(len(X_train)))
        for clf_name in self.classifiers:
            start = time.time()
            clf = self.classifiers[clf_name]  # on sort l'objet Scikit-learn
            clf.fit(X_train, Y_train)  # Entraînement

            # Validation
            if len(X_val) > 0:
                Y_val_pred = clf.predict(X_val)
                self.successes[clf_name] = np.mean(Y_val_pred == Y_val)
            else:
                self.successes[clf_name] = 1 # Pas de validation ==> Poids identiques

            print("   {:20} --> {:.3f}s".format(clf_name, time.time()-start))  # on affiche la durée de calcul.

        print("Validation: {} lines of data".format(len(X_val)))
        if len(X_val) > 0:
            for clf_name in self.classifiers:
                print("   {:20} --> {:.3f}".format(clf_name, self.successes[clf_name]))


    def predict(self, X):
        """
        Renvoie les prédictions du classifieur à partir des prédictions de chacun des classifieurs de base,
        La décision est rendue après un vote pondéré par l'efficacité de chacun des classifieurs

        Input:
         - X: list of sentences. A sentence is a sequence of words ([a-zA-Z]*) separated by a space.
        Output:
         - Y_pred: numpy array (N_instances)
        """

        print("Tokenisation...")
        X = self.tfidf_vectorizer.transform(X)
        X = self.truncatedsvd.transform(X) # numpy array (N_instances,N_features) carac d'entrée des données à prédire

        print("Predicting...")
        probas = np.zeros((X.shape[0], ))
        for name in self.classifiers:
            probas += self.classifiers[name].predict(X) * np.exp(self.successes[name])
            pred_class = [1 if proba > 0.5 else 0 for proba in probas]

        probas /= sum([np.exp(self.successes[name]) for name in self.successes ])
        Y_pred = np.array([1 if proba > 0.5 else 0 for proba in probas])

        return Y_pred



    def test(self, X, Y):
        """
        Renvoie les prédictions du classifieur à partir des prédictions de chacun des classifieurs de base,
        La décision est rendue après un vote pondéré par l'efficacité de chacun des classifieurs

        Input:
         - X: list of sentences. A sentence is a sequence of words ([a-zA-Z]*) separated by a space.
         - Y: numpy array (N_instances) (optionnel) classes réelles
                # Quand il est précisé, le taux de succès de chacun des sous_classifieurs s'affiche (print)
        Output:
         - Y_pred: numpy array (N_instances)
         - success_rate: self explanatory.
        """

        print('test1________________________________________________________________',len(X))
        print("Tokenisation...")
        X = self.tfidf_vectorizer.transform(X)

        print('test2________________________________________________________________',X.shape)
        X = self.truncatedsvd.transform(X) # numpy array (N_instances,N_features) carac d'entrée des données à prédire

        print('test3________________________________________________________________',X.shape)

        probas = np.zeros((X.shape[0], ))

        print('\nSuccess rate per classifier:')
        for name in self.classifiers:
            probas += self.classifiers[name].predict(X) * np.exp(self.successes[name])
        probas /= sum([np.exp(self.successes[name]) for name in self.successes ])

        Y_pred = np.array([1 if proba > 0.5 else 0 for proba in probas])
        success_rate = np.mean(Y_pred == np.array(Y))

        return Y_pred, success_rate



if __name__ == "__main__":

    TRAINING_SET_FOLDER_1 = "../../data/data_videos_training_set"
    TESTING_SET_FOLDER_1 = "../../data/data_videos_testing_set"


    print("========================")
    print("|        TRAIN         |")
    print("========================")
    data = getData(TRAINING_SET_FOLDER_1)[:10000]
    data = balanceData(data)
    labels, X = zip(*data)
    Y = binariseLabels(labels)

    model = MetaClassifier(validation_rate=0.1, n_features=150)
    model.train(X, Y)

    print("========================")
    print("|        TEST          |")
    print("========================")
    data = getData(TESTING_SET_FOLDER_1)[:100]
    labels, X = zip(*data)
    Y = binariseLabels(labels)

    Y_pred, success_rate = model.test(X, Y)
    print(Y_pred, success_rate)
