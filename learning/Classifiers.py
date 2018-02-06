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
    



#filename = "../../data/data_books_cleaned/books_aa.txt"  
    #sur données équilibrées : 75% ; sur données réelles 58% 
#filename = "../../data/data_videos_cleaned/datavideo_aa.txt"
    #sur données équilibrées : 58% ; sur données réelles 561%

N = 1000  # nombre de lignes à examiner
PERCENTAGE_TRAIN = 0.75

TRAINING_SET_FOLDER_1 = "../../data/data_videos_training_set"

class SklearnClassifiers():
    def __init__(self):
        self.dct = {
                'Naive Bayes': GaussianNB(),
                'CART':DecisionTreeClassifier(criterion='gini', splitter='best'),
                'Id3':DecisionTreeClassifier(criterion='entropy', splitter='best'),
                'Decision stump':DecisionTreeClassifier(splitter='best', max_depth = 1),
                ###'Multilayer Perceptron':MLPClassifier(hidden_layer_sizes=(20,10), activation='relu', learning_rate='invscaling'),
                'KNN':KNeighborsClassifier(n_neighbors=50),
                'TreeBagging':BaggingClassifier(n_estimators=75),
                'AdaBoost':AdaBoostClassifier(n_estimators=15),
                'Random Forest':RandomForestClassifier(n_estimators=25) 
                }  # dictionnaire des classifieurs que l'on va utiliser
   
        self.successes = {}  # performances de chacun des classifieurs
                            # sera calculée plus tard  



    def train(self,X,Y):
        # Entrées :
            # X = numpy array (N_instances,N_features)
            # Y = numpy array (N_instances)
        # Sorties :
            # None
        limit = (9*X.shape[0])//10  # séparation entraînement/validation
            # la validation sert à mesurer les taux de succès de chaque algo pour
            # donner plus de poids aux bons algos
        X_train,Y_train = X[:limit,:],Y[:limit] # ne sert qu'à l'entraînement
        
        X_val,Y_val= X[limit:,:],Y[limit:]  # validation, sert à calculer les performances
        
        print('\n[TRAIN] Temps de fonctionnement')
        for clf_name in self.dct:
            start = time.time()
            clf = self.dct[clf_name]  # on sort l'objet Scikit-learn
            clf.fit(X_train,Y_train)  # entraînement
            
            preds = clf.predict(X_val)
            
            self.successes[clf_name] = np.mean(preds == Y_val)
            
            print("\t{:20} --> {:.3f}s".format(clf_name, time.time()-start))  # on affiche le temps mis pour traiter N lignes
            
            
            
    def predict(self,X,Y_test = None):
        # Entrées :
            # X = numpy array (N_instances,N_features) carac d'entrée des données à prédire
            # Y_test = numpy array (N_instances) (optionnel) classes réelles 
                # Quand il est précisé, le taux de succès de chacun dessous_classifieurs s'affiche (print)
        # Sorties :
            # Y = numpy array (N_instances)
        # renvoie les prédictions du classifieur à partir des prédictions de chacun des classifieurs de base,
        # la décision est rendue après un vote pondéré par l'efficacité de chacun des classifieurs
            
        probas = np.zeros((X.shape[0],))
        
        print('\n[TEST] Taux de succès')
        for name in self.dct:
            clf = self.dct[name]
            probas += clf.predict(X)  #*self.successes[name]
                            # Actuellement, on ne prend pas en compte les taux de succès
                            # dé-commenter la ligne pour les prendre en compte
            if type(Y_test) !=  type(None) :  # Y_test == None renvoie un array si Y_test est un array
                pred_class = [1 if proba > 0.5 else 0 for proba in probas]
                #Success_tab = [1 if  pred_class[i] == Y_test[i] else 0 for i in range(len(Y_test))]
                #SuccessRate = np.mean(Success_tab)
                SuccessRate = np.mean(pred_class == Y_test)
                
                print("\t{:20} --> {:.3f}".format(name,SuccessRate))
        
        probas /= len(self.dct)  #sum([self.successes[name]  for name in self.successes ])
                            # Actuellement, on ne prend pas en compte les taux de succès
                            # remplacer le len(self.dct) par le commentaire pour les prendre en compte
        classes = np.array([1 if proba > 0.5 else 0 for proba in probas])
        return classes
    
    
    
    
def tokenize(textList):
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
    1. countvectorizer.fit(textlist) 
       ne renvoie rien, associe un indice (un entier) à chaque mot dans 
       la liste de strings textlist
       ex : si textlist1 = ['aa bb','aa cc','dd']
       la fonction prépare l'objet à faire 'aa'-> 1
                                           'bb'-> 2
                                           'cc'-> 3
                                           'dd'-> 4

    2. X_token = countvectorizer.transform(textList) : 
       crée un array numpy de la forme A(i,j) = nombre de mots d'indice I dans la string I
       ex : si textlist2 = ['aa aa','bb cc','dd aa zz']
         et si on a fait countvectorizer.fit(textlist1) (cf exemple précédent),
      
        
    colonne 2 = nombre de mots "bb" colonne 3 = nombre de mots "cc"      
    colonne 1 = nombre de mots "aa" | | colonne 4 = nombre de mots "dd" 
                                  | | | | 
                                  V V V V
         la fonction renverra M=[[2,0,0,0],  <- string 1 = 'aa aa'
                                 [0,1,1,0],  <- string 1 = 'bb cc'
                                 [1,0,0,1]]  <- string 1 = 'dd aa zz'
           
    Comme le mot "zz" ne faisait pas partie de textlist1 (la liste utilisée en argument de countvectorizer.fit)                       
    ce mot n'est associé à rien

    Rq : l'array M est une matrice sparse (majoritairement vide), c'est un type d'objet qui permet de 
    ne pas stocker des tas de zéros en mémoire. Pour la transformer en array normal, on peut faire
    M.toarray(), mais le tableau ainsi crée est souvent trop gros pour être géré.
    Le mieux est d'utiliser la décomposition en valeurs singulières, cf plus loin:
    """                              
    
    # réduction de dimension
    truncatedsvd = TruncatedSVD(n_components=100) # prépare à projeter les données dans un espace à n_components=100 dimensions 
    
    X_reduced_dim = truncatedsvd.fit_transform(X_token)
    """Comme Countvectorizer.fit_transform, cette instruction réalise 2 opérations
       truncatedsvd.fit(X_token) 
           prépare l'objet, lui dit d'utiliser les mots avec la distribution de probabilité de 
           X_token
    
       X_reduced_dim = truncatedsvd.fit_transform(X_token)
           fait la Décomposition en valeurs singulières (SVD), qui est l'équivelent d'une diagonalisation
             pour des matrices rectangles. On calcule U,V,D, tq : 
                 - U carrée, U*transposée(U) = I_m
                 - D rectancle, diagonale
                 - V carrée, V*transposée(V) = I_n
            On renvoie ensuite U[:n:n_components], la matrice U dont on a tronqué les coordonnées
            qui a concentré l'information, un peu à la manière d'une ACP (pour les maths, cf Wikipédia)
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

    filenames = os.listdir(folder)

    listdata = []
    for filename in filenames[:6]: # A résoudre : au bout de 9000 exemples, il crashe.
        absolute_path =  os.path.join(folder, filename)
     
        with open(absolute_path, 'r') as f:
            for line in f:
                listdata.append(line.strip().split('\t'))

    return listdata




def main(filename, balancing=False):

    starttime = time.time()

    print("== DATA RETRIEVAL ==")
    data = getData(filename)
    print("{} lines of data".format(len(data)))

    if balancing:
        data = balanceData(data)
        print("{} lines of data kept after balancing".format(len(data)))



    # Write labels in labels and X
    labels = [line[0] for line in data]
    X = [line[1] for line in data]
    
    # labels, X = zip(*data) # ZIP A CREUSER

    print("== TOKENISATION ==")
    X_token = tokenize(X)
    print("Time spent for tokenisation: {:.3f}s".format(time.time() - starttime))


    print("== TRAINING TIME ==")
    labels_bin = [ 0 if label in ['Negative','Neutral'] else 1 for label in labels]
    labels_bin = np.array(labels_bin)


    # Split lines in train and test
    N_sep = int(len(labels_bin) * PERCENTAGE_TRAIN)

    X_train = X_token[:N_sep,:]    
    X_test = X_token[N_sep:,:]
    Y_train = labels_bin[:N_sep]    
    Y_test = labels_bin[N_sep:]


    # Training
    C = SklearnClassifiers()
    C.train(X_train,Y_train)
    print("Time spent for training: {:.3f}s".format(time.time() - starttime))

    Y_pred = C.predict(X_test, Y_test=Y_test)

    # Show results
    print("== RESULTS ==")
    print("DATA")
    print("  Taux de revues positives : ",np.mean(labels_bin))
    print("RESULTS")
    print("  Exemples de predictions :",Y_pred[:10])
    print("  Classes réelles :        ",Y_test[:10])
    print("  Taux de revues avec 4,5 étoiles (données réelles) :",np.mean([Y_test]))
    print("  Taux de revues avec 4,5 étoiles (selon la prédiction) :",np.mean([Y_pred]))
    print("  Taux de succès", np.mean([Y_pred == Y_test]))


if __name__ == "__main__":
    main(TRAINING_SET_FOLDER_1, balancing=False)





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
