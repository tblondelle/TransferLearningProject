import time
import numpy as np
import os
from random import shuffle

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer



class MetaClassifier():
    def __init__(self, validation_rate=0.1):
        self.classifiers = {
            'Naive Bayes': GaussianNB(),
            'CART':DecisionTreeClassifier(criterion='gini', splitter='best'),
            'Id3':DecisionTreeClassifier(criterion='entropy', splitter='best'),
            'Decision stump':DecisionTreeClassifier(splitter='best', max_depth = 1),
            #'Multilayer Perceptron':MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', learning_rate='invscaling'),
            'KNN':KNeighborsClassifier(n_neighbors=50),
            'TreeBagging':BaggingClassifier(n_estimators=75),
            'AdaBoost':AdaBoostClassifier(n_estimators=15),
            'Random Forest':RandomForestClassifier(n_estimators=25)
        }  # dictionnaire des classifieurs que l'on va utiliser

        self.successes = {
            'Naive Bayes': "",
            'CART':"",
            'Id3':"",
            'Decision stump':"",
            #'Multilayer Perceptron':"",
            'KNN':"",
            'TreeBagging':"",
            'AdaBoost': "",
            'Random Forest':""
        }  # performances de chacun des classifieurs

        self.validation_rate = validation_rate


    def train(self, X, Y):
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

            probas += clf.predict(X) * self.successes[name]

            if type(Y) != type(None) :  # Y == None renvoie un array si Y est un array
                pred_class = [1 if proba > 0.5 else 0 for proba in probas]
                #Success_tab = [1 if  pred_class[i] == Y[i] else 0 for i in range(len(Y))]
                #SuccessRate = np.mean(Success_tab)
                SuccessRate = np.mean(pred_class == Y)

                print("   {:20} --> {:.1f}%".format(name, SuccessRate*100))

        probas /= sum([self.successes[name] for name in self.successes ])

        classes = np.array([1 if proba > 0.5 else 0 for proba in probas])
        return classes


def tokenize(textList,n_features=100):
    """
    Entrées :
        # textList : liste de strings de taille N
    # Sorties :
        # X : numpy array de taille Nx100
    # Transforme chaque phrase (string) en un vecteur grâce à CountVectorizer et TruncatedSVD
    """
    countvectorizer = CountVectorizer(ngram_range=(1,2))

    X_token = countvectorizer.fit_transform(textList)
    """
    Countvectorizer.fit_transform réalise 2 opérations :
    - `countvectorizer.fit(textlist)`  ne renvoie rien, associe un indice
        (un entier) à chaque mot dans la liste de strings textlist.
        Ex : si textlist1 = ['aa bb','aa cc','dd'], la fonction prépare
        l'objet à faire 'aa'-> 1, 'bb'-> 2, 'cc'-> 3, 'dd'-> 4.
    - `X_token = countvectorizer.transform(textList)` crée un array numpy de la
        forme A(i,j) = nombre de mots d'indice j dans le string i.
        Ex : si textlist2 = ['aa aa','bb cc','dd aa zz'] et si on a fait
        `countvectorizer.fit(textlist1)` (cf exemple précédent), alors

              colonne 2 = nb de mots "bb" colonne 3 = nb de mots "cc"
             colonne 1 = nb de mots "aa"| | colonne 4 = nb de mots "dd"
                                      | | | |
                                      V V V V
        la fonction renverra    M = [[2,0,0,0],  <- string 1 = 'aa aa'
                                     [0,1,1,0],  <- string 2 = 'bb cc'
                                     [1,0,0,1]]  <- string 3 = 'dd aa zz'

        Comme le mot "zz" ne faisait pas partie de textlist1 (la liste utilisée en argument de countvectorizer.fit)
        ce mot n'est associé à rien

    Rq : l'array M est une matrice sparse (majoritairement vide), c'est un type d'objet qui permet de
    ne pas stocker des tas de zéros en mémoire. Pour la transformer en array normal, on peut faire
    M.toarray(), mais le tableau ainsi crée est souvent trop gros pour être géré.
    Le mieux est d'utiliser la décomposition en valeurs singulières, cf plus loin:
    """

    # Réduction de dimension
    truncatedsvd = TruncatedSVD(n_components=n_features) # prépare à projeter les données dans un espace à n_components=100 dimensions

    X_reduced_dim = truncatedsvd.fit_transform(X_token)
    """
    Comme Countvectorizer.fit_transform, cette instruction réalise 2 opérations
    - `truncatedsvd.fit(X_token)` prépare l'objet, lui dit d'utiliser les mots
       avec la distribution de probabilité de  X_token

    - `X_reduced_dim = truncatedsvd.fit_transform(X_token)` fait la Décomposition
        en valeurs singulières (SVD), qui est l'équivelent d'une diagonalisation
        pour des matrices rectangles. On calcule U,V,D, tq :
            - U carrée, U*transposée(U) = I_m
            - D rectancle, diagonale
            - V carrée, V*transposée(V) = I_n
        On renvoie ensuite U[:n_components], la matrice U dont on a tronqué les
        coordonnées qui a concentré l'information, un peu à la manière d'une
        ACP (pour les maths, cf Wikipédia).
    """

    return(X_reduced_dim)


def balanceData(data):
    """
    Return a shorter version of data where an equal number of
    (negative and neutral) and positif lines are returned.
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


def learn(training_set_folder, dataBalancing=False,n_features=100):
    """
    Input :
     - training_set_folder: string of the path of the training_set_folder
     - dataBalancing: boolean to balance data or not.
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
    data = getData(training_set_folder)
    print("{} lines of data".format(len(data)))

    if dataBalancing:
        data = balanceData(data)
        print("{} lines of data kept after balancing".format(len(data)))


    print("\n== TOKENISATION ==")
    # Write labels in labels and X
    labels, X = zip(*data)

    countvectorizer = CountVectorizer(ngram_range=(1,2))
    X_token = countvectorizer.fit_transform(X)
    truncatedsvd = TruncatedSVD(n_components=n_features)
    X_train = truncatedsvd.fit_transform(X_token)

    labels_bin = [ 0 if label in ['Negative','Neutral'] else 1 for label in labels]
    Y_train = np.array(labels_bin)
    print("Time spent for tokenisation: {:.3f}s".format(time.time() - start_time))


    print("\n== TRAINING ==")
    metaClassifier = MetaClassifier(validation_rate = 0.09)
    metaClassifier.train(X_train,Y_train)
    print("Time spent for training: {:.3f}s".format(time.time() - start_time))


    print("\n== TRAINING RESULTS ==")
    print("DATA")
    print("  Taux de revues avec 4,5 étoiles (données réelles) : {:.3f}".format(np.mean(Y_train)))

    return [countvectorizer, truncatedsvd, metaClassifier]


def showResults(model, testing_set_folder):
    """
    Reprend un ancien modèle composé des trois élements et
    l'utilise pour la prédiction sur l'ensemble des fichiers
    du testing_set_folder.
    """

    [countvectorizer, truncatedsvd, metaClassifier] = model

    print("\n== DATA RETRIEVAL ==")
    data = getData(testing_set_folder)
    print("{} lines of data".format(len(data)))

    # Write labels in labels and X_token
    labels, X = zip(*data)

    labels_bin = [ 0 if label in ['Negative','Neutral'] else 1 for label in labels]
    labels_bin = np.array(labels_bin)


    print("\n== TOKENISATION ==")
    X_test = countvectorizer.transform(X)
    X_test = truncatedsvd.transform(X_test)
    Y_pred = metaClassifier.predict(X_test, Y=labels_bin)


    print("\n== TEST RESULTS ==")
    print("  Exemples de prédictions : {}".format(Y_pred[:10]))
    print("  Classes réelles :         {}".format(labels_bin[:10]))
    print("  Taux de revues avec 4,5 étoiles (données réelles) : {:.3f}".format(np.mean([labels_bin])))
    print("  Taux de revues avec 4,5 étoiles (selon la prédiction) : {:.3f}".format(np.mean([Y_pred])))
    print("  Taux de succès : {:.3f}".format(np.mean([Y_pred == labels_bin])))



if __name__ == "__main__":

    TRAINING_SET_FOLDER_1 = "../../data/data_videos_training_set"
    TESTING_SET_FOLDER_1 = "../../data/data_videos_testing_set"
    #TESTING_SET_FOLDER_1 = TRAINING_SET_FOLDER_1


    print("========================")
    print("|        TRAIN         |")
    print("========================")
    model = learn(TRAINING_SET_FOLDER_1, dataBalancing=False)


    print("========================")
    print("|        TEST          |")
    print("========================")
    showResults(model, TESTING_SET_FOLDER_1)



"""
def option3():
    # Option 3 : probas équitables sur l'ensemble d'apprentissage, mais pas de test
    # Un échec
    # En cours de deboggage...

    with open(filename) as myfile:
        head = list(islice(myfile,N))


    indexes_to_keep = []
    for i in range(len(head)):
        n = int(head[i][0])
        if n < 4:
            indexes_to_keep.append(i)

    N_small = len(indexes_to_keep)
    i = 0
    while len(indexes_to_keep)<2*N_small:
        n = int(head[i][0])
        if n >= 4:
            indexes_to_keep.append(i)
        i+=1

    head = [head[i] for i in indexes_to_keep]


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


    C = SklearnClassifiers()

    C.train(X_token_tr,labels_bin_tr)
    Y_pred = C.predict(X_token_te)

    print("exemples de predictions :",Y_pred[:10])
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
