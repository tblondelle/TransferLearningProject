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

from sklearn.feature_extraction.text import CountVectorizer
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
    for filename in filenames[:5]:
        print(os.path.join(folder, filename))

        with open(os.path.join(folder, filename), 'r') as f:
            for line in f:

                line2 = line.strip().split('\t')
                if len(line2) == 2:
                    listdata.append(line2)

    return listdata


class MetaClassifier():
    def __init__(self, validation_rate=0.1, n_features=150):

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



    def train(self, training_set_folder, dataBalancing=False):
        """
        Input :
         - training_set_folder: string of the path of the training_set_folder
         - dataBalancing: boolean to balance data or not
         - n_features: integer for n_features for SVD (/!\ : exponential complexity !!)
        Output :
         - model = 3 elements.

        A partir d'un dossier composé de fichiers texte d'entrainement,
        on entraine un certain modèle. On retourne ce modèle composé
        de trois élements : CountVectorizer, TruncatedSVD et une instance
        de MetaClassifier.
        On a la possibilité d'équilibrer les données (avoir autant
        de reviews positives que négatives) avec l'argument dataBalancing à True.
        """

        start_time = time.time()

        print("\n== DATA RETRIEVAL ==")
        data = getData(training_set_folder)#[:50] #Pour ALLER PLUS VITE LORS DES TEST !!
        print("{} lines of data".format(len(data)))

        if dataBalancing:
            data = balanceData(data)
            print("{} lines of data kept after balancing".format(len(data)))


        print("\n== TOKENISATION ==")
        # Write labels in labels and X
        labels, X = zip(*data)


        # On va trier les données, on trie aussi les X :
        X_without_neutral = [X[i][:] for i in range(len(X)) if labels[i] in ['Negative']  ]
        X_without_neutral += [X[i][:] for i in range(len(X)) if labels[i] in ['Positive'] ]
        X = X_without_neutral

        labels_bin = [ 0 for  label in labels if label in ['Negative']]
        labels_bin += [ 1 for  label in labels if label in ['Positive']]
        Y = np.array(labels_bin)



        #countvectorizer = CountVectorizer(ngram_range=(1,2))
        #X_token = countvectorizer.fit_transform(X)

        tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
        X_token = tfidf_vectorizer.fit_transform(X)

        truncatedsvd = TruncatedSVD(n_components=self.n_features)
        X = truncatedsvd.fit_transform(X_token)

        self.tfidf_vectorizer = tfidf_vectorizer
        self.truncatedsvd = truncatedsvd

        print("Time spent for tokenisation: {:.3f}s".format(time.time() - start_time))


        print("\n== TRAINING ==")
        



        # Entrées :
            # X = numpy array (N_instances,N_features)
            # Y = numpy array (N_instances)
        # Sorties :
            # None

        # Séparation entraînement/validation
        #   La validation sert à mesurer les taux de succès de chaque algo pour
        #   donner plus de poids aux bons algos.
        limit = int((1 - self.validation_rate) * X.shape[0])

        X_train, Y_train = X[:limit,:], Y[:limit] # ne sert qu'à l'entraînement
        X_val, Y_val = X[limit:,:], Y[limit:]  # validation, sert à calculer les performances


        print("\nTemps d'entrainement")
        print("  {} individus pour l'entrainement".format(len(X_train)))
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

        print("\nRésultat de la validation")
        print("  {} individus pour la validation".format(len(X_val)))
        if len(X_val) > 0:
            for clf_name in self.classifiers:
                print("   {:20} --> {:.1f}%".format(clf_name, self.successes[clf_name]*100))


        print("Time spent for training: {:.3f}s".format(time.time() - start_time))

        print("\n== TRAINING RESULTS ==")
        print("DATA")
        print("  Taux de revues avec 4,5 étoiles (données réelles) : {:.3f}".format(np.mean(Y_train)))




    def predict(self, X, Y=None):
        """
        Entrées :
            X = numpy array (N_instances,N_features) carac d'entrée des données à prédire
            Y = numpy array (N_instances) (optionnel) classes réelles
                # Quand il est précisé, le taux de succès de chacun des sous_classifieurs s'affiche (print)
        Sorties :
            Y = numpy array (N_instances)
        Renvoie les prédictions du classifieur à partir des prédictions de chacun des classifieurs de base,
        La décision est rendue après un vote pondéré par l'efficacité de chacun des classifieurs
        """

        probas = np.zeros((X.shape[0],))

        print('\nTaux de succès')
        for name in self.classifiers:
            clf = self.classifiers[name]

            probas += clf.predict(X) * np.exp(self.successes[name])

            if type(Y) != type(None) :  # Y == None renvoie un array si Y est un array
                pred_class = [1 if proba > 0.5 else 0 for proba in probas]
                #Success_tab = [1 if  pred_class[i] == Y[i] else 0 for i in range(len(Y))]
                #SuccessRate = np.mean(Success_tab)
                SuccessRate = np.mean(pred_class == Y)

                print("   {:20} --> {:.1f}%".format(name, SuccessRate*100))

        probas /= sum([np.exp(self.successes[name]) for name in self.successes ])

        classes = np.array([1 if proba > 0.5 else 0 for proba in probas])
        return classes



    def showResults(self, testing_set_folder):
        """
        Reprend un ancien modèle composé des trois élements et
        l'utilise pour la prédiction sur l'ensemble des fichiers
        du testing_set_folder.
        """

        print("\n== DATA RETRIEVAL ==")
        data = getData(testing_set_folder)
        print("{} lines of data".format(len(data)))

        # Write labels in labels and X_token
        labels, X = zip(*data)

        # On va trier les données, on trie aussi les X :
        X_without_neutral = [X[i][:] for i in range(len(X)) if labels[i] in ['Negative']  ]
        X_without_neutral += [X[i][:] for i in range(len(X)) if labels[i] in ['Positive'] ]
        X = X_without_neutral

        labels_bin = [ 0 for  label in labels if label in ['Negative']]
        labels_bin += [ 1 for  label in labels if label in ['Positive']]
        #Y_train = np.array(labels_bin)


        print("\n== TOKENISATION ==")
        X_test = self.tfidf_vectorizer.transform(X)
        X_test = self.truncatedsvd.transform(X_test)
        Y_pred = self.predict(X_test, Y=labels_bin)
        
        index_sample = sample(range(len(labels_bin)), k=10)

        print("\n== TEST RESULTS ==")
        print("  Exemples de prédictions : {}".format([Y_pred[i] for i in index_sample]))
        print("  Classes réelles :         {}".format([labels_bin[i] for i in index_sample]))
        print("  Taux de revues avec 4,5 étoiles (données réelles) : {:.3f}".format(np.mean([labels_bin])))
        print("  Taux de revues avec 4,5 étoiles (selon la prédiction) : {:.3f}".format(np.mean([Y_pred])))
        print("  Taux de succès : ................................................. {:.3f}".format(np.mean([Y_pred == labels_bin])))

        with open('results','w') as f:
            f.write("\n== TEST RESULTS ==")
            f.write("\n  Exemples de prédictions : {}".format([Y_pred[i] for i in index_sample]))
            f.write("\n  Classes réelles :         {}".format([labels_bin[i] for i in index_sample]))
            f.write("\n  Taux de revues avec 4,5 étoiles (données réelles) : {:.3f}".format(np.mean([labels_bin])))
            f.write("\n  Taux de revues avec 4,5 étoiles (selon la prédiction) : {:.3f}".format(np.mean([Y_pred])))
            f.write("\n  Taux de succès : {:.3f}".format(np.mean([Y_pred == labels_bin])))



if __name__ == "__main__":

    TRAINING_SET_FOLDER_1 = "../../data/data_videos_training_set"
    TESTING_SET_FOLDER_1 = "../../data/data_videos_testing_set"
    #TESTING_SET_FOLDER_1 = TRAINING_SET_FOLDER_1


    print("========================")
    print("|        TRAIN         |")
    print("========================")
    model = MetaClassifier(validation_rate=0.1, n_features=150)
    model.train(TRAINING_SET_FOLDER_1, dataBalancing=False)


    print("========================")
    print("|        TEST          |")
    print("========================")
    model.showResults(TESTING_SET_FOLDER_1)


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
