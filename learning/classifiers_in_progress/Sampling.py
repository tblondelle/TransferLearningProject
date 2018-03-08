# -*- coding: utf-8 -*-
import numpy as np
import random
import time
# on définit des fonctions qui permettent de sous-échantillonner/sur-échantillonner
# les ensembles d'apprentissage(et pas les ensebles de test) pour améliorer les performances de l'algo de ML
# cf http://conteudo.icmc.usp.br/pessoas/gbatista/files/sigkdd2004.pdf
#
# La fonction Sampling appelée a la signature suivante:
# Entrées :
#   X : Numpy Array  de taille (Nb_instances,Nb_features)
#   Y : Numpy Array  de taille (Nb_instances,), qui ne contient que des classes BINAIRES
#   sampling_type : String, le type d'échantillonnage réalisé.
#       Les différents types d'échantillonnage sont disponibles dans la variable Available_Samplings
# Sortie :
#   (I,New_X,New_Y), un triplet qui contient :
#   I : Liste d'indices des instances à conserver dans X,Y
#   New_X : Numpy Array  de taille (Nb_inst_2,Nb_features)
#   New_Y : Numpy Array  de taille (Nb_inst_2,)
#     Ces deux arrays contiennent des instances nouvellement créées pour l'over-sampling,
#     grâce à l'interpolation des instances existantes




# Auxiliary function
def Majority_class(Y):
    # trouve la classe majoritaire dans Y
    count = {} # dictionnaire de la forme :
      #  classe: nombre d'instance de cette classe

    for C in Y:
        if C not in count :
            count[C] = 1
        else :
            count[C] += 1

    # on trouve le max :
    majority_class = max(count,key = lambda x:count[x])
    minority_class = [Class for Class in count if Class != majority_class][0]
    return majority_class,minority_class






Available_Samplings = ["Tomek","SMOTE","OSS"]


def Sampling(X,Y,sampling_type):
    if sampling_type == "Tomek":
        I_kept =  apply_Tomeks(X,Y)
        return (I_kept,None,None)

    elif sampling_type == "SMOTE":
        Percent = .55 # *100%
        k = 30
        # On créé Percent*N_min nouveaux individus, où N_min est le nombre d'instances de la classe minoritaire
        (X_new,Y_new) =  apply_SMOTE(X,Y,Percent,k)
        return (None,X_new,Y_new)


    elif sampling_type == "OSS":
        I_kept =  apply_OSS(X,Y)
        return (I_kept,None,None)


    else :
        raise ValueError('Unknown sampling type : {}'.format(sampling_type))




# ================================================================
# ==================== All samplings are coded here ==============
# ================================================================


def apply_Tomeks(X,Y):
    # calcule les indices des instances à virer, selon la méthode Tomek links :
    # Examiner tous les couples (i,j) d'instances de la classe majoritaire et minoritaire
    # S'il n'y a pas d'instance majoritaire plus proche de i que j, i.e si pour toute instance majoritaire k,
    #   d(i,k) > d(j,k)
    # et s'il n'y a pas d'instance minoritaire plus proche de j que i, i.e si pour toute instance minoritaire m,
    #   d(j,m) > d(i,j)
    # alors (i,j) est une paire de Tomek. On supprime donc l'instance majoritaire i

    global start
    (N_instances,N_features) = X.shape
    majority_class,minority_class = Majority_class(Y)  # classe Majoritaire


    TAKING_INTO_ACCOUNT = 500 # on évite la complexité en n2 en
        # ne prenant en compte aue les N premières instances de
        # chaque classe au moment de la comparaison

    # liste des indices des instances des classes majoritaires
    I_maj = [ i for i in range(N_instances) if Y[i] == majority_class ]
    I_min = [ i for i in range(N_instances) if Y[i] != majority_class ]

    print("Tomeks : calcul dess indices : ",time.time()-start,"s")
    start = time.time()

    distances_inter = {} # dictionnaire de la forme
        # (i,j) = distance de l'instance i à l'instance j
        # où i est dans la classe majoritaire
        # et j est dans la classe minoritaire
    for iM in I_maj :
        for im in I_min[:TAKING_INTO_ACCOUNT] :
            if (iM,im) not in distances_inter:
                distances_inter[(iM,im)] = np.linalg.norm(X[iM,:]-X[im,:])


    print("calcul des distances inter-classes : ",time.time()-start,"s")
    start = time.time()

    # idem for intra_class distances :
    distances_intra_maj = {}
        # ATTENTION : ici, distance(i,j) = distance(j,i)
        # du coup, pour des raisons d'emplacement en mémoire, on ne stocke que
        # les valeurs (i,j) où i<j
        # on bricolera quand il faudra lire les valeurs (les lignes où l'on se
        # sert de cette info sont marquées par le hastag #bricolage )

   # distances_m = {}
        # on passera sur tous les iMaj, puis sur tous les imin.
        # pour ne par les re-calculer
    for i in I_maj[:TAKING_INTO_ACCOUNT] :
        for j in I_maj :
            if ((i,j) not in distances_intra_maj) and ((j,i) not in distances_intra_maj):  #bricolage
                distances_intra_maj[(i,j)] = np.linalg.norm(X[i,:]-X[j,:])

    print("calcul des distances intra-classes (Maj) : ",time.time()-start,"s")
    start = time.time()


    distances_intra_min = {}
    for i in I_min :
        for j in I_min :
            if ((i,j) not in distances_intra_min) and ((j,i) not in distances_intra_min): #bricolage
                distances_intra_min[(i,j)] = np.linalg.norm(X[i,:]-X[j,:])

    # Maintenant qu'on a bien toutes les distances, trions :
    I_kept = I_min+I_maj
        # on retirera les indices au fur et à mesure

    print("calcul des distances intra-classes (min) : ",time.time()-start,"s")
    start = time.time()


    for iM in I_maj :
        distances_M = [distances_intra_maj[(iM,j)] for j in I_maj[:TAKING_INTO_ACCOUNT] if iM<j] #bricolage
        distances_M += [distances_intra_maj[(j,iM)] for j in I_maj[:TAKING_INTO_ACCOUNT] if j<iM] #bricolage
        for im in I_min[:TAKING_INTO_ACCOUNT] :
            d_M_m = distances_inter[(iM,im)]

            condition_m_list = [ distances_intra_min[(im,j)]>d_M_m for j in I_min if j > im] #bricolage
            condition_m_list += [ distances_intra_min[(j,im)]>d_M_m for j in I_min if j < im] #bricolage
            condition_m = all(condition_m_list)
                # partie de la définition de Tomek pair qui concerne la classe minoritaire

            condition_M = all([d_M_M > d_M_m for d_M_M in distances_M])

            if condition_M and condition_m:
                # this is a Tomek pair, DELET THIS
                try :
                    I_kept.remove(iM)
                except :
                    # iM already removed
                    # too bad
                    pass

    print("Application de Tomeks : ",time.time()-start,"s")
    start = time.time()


    return I_kept












def apply_SMOTE(X,Y,Percentage,k):
    # Cree des instances de la classe minoritaires par la méthode SMOTE :
    # Tant que l' nombre d'instances de la classe minoritaire n'a pas augmenré de Percentage (relativement):
        # Choisir une instance minoritaire A au hasard
        # Choisir une instance minoritaire B parmi ses K plus proches voisins (K est un parametre)
        # Choisir un réel X entre 0 et 1
        # Créer l'instance A+X(B-A)


    global start
    (N_instances,N_features) = X.shape
    majority_class,minority_class = Majority_class(Y)  # classe Majoritaire

    # liste des indices des instances des classes minoritaire
    I_min = [ i for i in range(N_instances) if Y[i] != majority_class ]

    print("SMOTE : calcul des indices : ",time.time()-start,"s")
    start = time.time()


    distances_intra_min = {}
    # ATTENTION : ici, distance(i,j) = distance(j,i)
        # du coup, pour des raisons d'emplacement en mémoire, on ne stocke que
        # les valeurs (i,j) où i<j
        # on bricolera quand il faudra lire les valeurs (les lignes où l'on se
        # sert de cette info sont marquées par le hastag #bricolage )

   # distances_m = {}
        # on passera sur tous les iMaj, puis sur tous les imin.
        # pour nbe par les re-calculer

    for i in I_min :
        for j in I_min :
            if ((i,j) not in distances_intra_min) and ((j,i) not in distances_intra_min): #bricolage
                distances_intra_min[(i,j)] = np.linalg.norm(X[i,:]-X[j,:])

    # Nombre d'individus à créer:
    N_to_create = int(len(I_min)*Percentage)

    print("calcul des distances intra-classes (min) : ",time.time()-start,"s")
    start = time.time()


    X_new = np.zeros( (N_to_create,N_features))
    Y_new = np.zeros( (N_to_create,)) + minority_class

    #print(I_min)

    for i in range(N_to_create):
        i_chosen_inst = random.choice(I_min)
        # on suhaite trouver les k plus proches voisins
        distances = []
        for j in I_min:
            #print(i_chosen_inst,j)
            if j>i_chosen_inst :  #bricolage
                distances.append((distances_intra_min[(i_chosen_inst,j)],j))
            elif i_chosen_inst>j :
                distances.append((distances_intra_min[(j,i_chosen_inst)],j))
                # on ne fait rien si i==j
        # on conserve les couples (distance, indice)
        distances.sort(key = lambda x:x[0] )
        # on trie sur les distances
        kpp = [j for (_,j) in distances[:k]]

        chosen_inst_2 = random.choice(kpp)

        X1 = X[i_chosen_inst][:]
        X2 = X[chosen_inst_2][:]

        New_instance = X1 + random.random()*(X2-X1)
        # 0 < random.random() < 1

        X_new[i][:] = New_instance

    print("Application de SMOTE : ",time.time()-start,"s")
    start = time.time()


    return (X_new,Y_new)











def apply_OSS(X,Y):
    # calcule les indices des instances à virer, selon la méthode One-sided samplinq :
    # Initialiser l'ensemble C : contient toutes les instances minoritaires + une instance majoritaire
    # Ensuite, classer *toutes* les instances : leur attribuer la classe de leur plus proche voisin dans C
    # Mettre toutes les instances mal classées dans C renvoyer C


    global start
    (N_instances,N_features) = X.shape
    majority_class,minority_class = Majority_class(Y)  # classe Majoritaire


    # liste des indices des instances des classes majoritaires
    I_maj = [ i for i in range(N_instances) if Y[i] == majority_class ]
    I_min = [ i for i in range(N_instances) if Y[i] != majority_class ]


    print("creation des listes d'indices : ",time.time()-start,"s")
    start = time.time()
    distances_inter = {} # dictionnaire de la forme
        # (i,j) = distance de l'instance i à l'instance j
        # où i est dans la classe majoritaire
        # et j est dans la classe minoritaire
    for iM in I_maj :
        for im in I_min :
            if (iM,im) not in distances_inter:
                distances_inter[(iM,im)] = np.linalg.norm(X[iM,:]-X[im,:])


    print("calcul des distances inter-classes : ",time.time()-start,"s")
    start = time.time()
    # idem for intra_class distances :
    distances_intra_maj = {}
        # ATTENTION : ici, distance(i,j) = distance(j,i)
        # du coup, pour des raisons d'emplacement en mémoire, on ne stocke que
        # les valeurs (i,j) où i<j
        # on bricolera quand il faudra lire les valeurs (les lignes où l'on se
        # sert de cette info sont marquées par le hastag #bricolage )

   # distances_m = {}
        # on passera sur tous les iMaj, puis sur tous les imin.
        # pour ne par les re-calculer
    for i in I_maj :
        for j in I_maj :
            if ((i,j) not in distances_intra_maj) and ((j,i) not in distances_intra_maj):  #bricolage
                distances_intra_maj[(i,j)] = np.linalg.norm(X[i,:]-X[j,:])

    print("calcul des distances intra-classes (Maj) : ",time.time()-start,"s")
    start = time.time()

    distances_intra_min = {}
    for i in I_min :
        for j in I_min :
            if ((i,j) not in distances_intra_min) and ((j,i) not in distances_intra_min): #bricolage
                distances_intra_min[(i,j)] = np.linalg.norm(X[i,:]-X[j,:])

    print("calcul des distances intra-classes (min) : ",time.time()-start,"s")
    start = time.time()

    # Maintenant qu'on a bien toutes les distances, élagons
    chosen_Maj = random.choice(I_maj)
    I_kept = I_min+[chosen_Maj]
        # on ajoutera les indices au fur et à mesure



    for iM in I_maj :
        distances_M = [ (distances_inter[(iM,j)],j) for j in I_min]
        distances_M += [(distances_intra_maj[(chosen_Maj,iM)],chosen_Maj) if chosen_Maj<iM else (distances_intra_maj[(iM,chosen_Maj)],chosen_Maj)] #bricolage
        # cette fois, on conserve les couples (distance,indice), vu qu'on aura besoin de l'indice
        # Classification à l'aide de 1-NN

        # On prend le min des distances
        Couple = min(distances_M, key=lambda couple:couple[0])
        classified_ind = Couple[1]

        if classified_ind != chosen_Maj :# i.e., if Y[Classified_ind] == Y[chosen_Maj]
            I_kept.append(iM)

    print("Application des algorithmes: ",time.time()-start,"s")
    start = time.time()


    return I_kept




# ================================================================
# ============================= Test =============================
# ================================================================


# Test de Tomek
"""
import matplotlib.pyplot as plt

A = np.random.random((100,2))
B = np.random.random((1000,2))

# concatenate A and B
X = [A[i,:] for i in range(A.shape[0])]
X += [B[i,:] for i in range(B.shape[0])]
X = np.array(X)

Y = [0 for i in range(A.shape[0])]
Y += [1 for i in range(B.shape[0])]
Y = np.array(Y)
import time

start = time.time()
(I_kept,_,_) = Sampling(X,Y,"Tomek")
print(time.time()-start)
#print(I_kept)


plt.plot(B[:,0],B[:,1],'.r')
plt.plot(X[I_kept,0],X[I_kept,1],'.g')
plt.plot(A[:,0],A[:,1],'.b')


print(time.time()-start)
"""

# Test de SMOTE
"""
import matplotlib.pyplot as plt

A = np.random.random((100,2))*1
B = np.random.random((1000,2))*1

# concatenate A and B
X = [A[i,:] for i in range(A.shape[0])]
X += [B[i,:] for i in range(B.shape[0])]
X = np.array(X)

Y = [0 for i in range(A.shape[0])]
Y += [1 for i in range(B.shape[0])]
Y = np.array(Y)


start = time.time()
(_,X_new,Y_new) = Sampling(X,Y,"SMOTE")
print(time.time()-start)
#print(I_kept)


plt.plot(B[:,0],B[:,1],'.g')
plt.plot(A[:,0],A[:,1],'.b')
plt.plot(X_new[:,0],X_new[:,1],'.r')




print(time.time()-start)
"""

# Test des deux :
if __name__ == "__main__":
    import Classifiers as c
    start  = time.time()

    TRAINING_SET_FOLDER = "../../data/data_books_cleaned"


    #with open(TRAINING_SET_FOLDER) as myfile:
    #    head = list(islice(myfile,1000)) # on prend les 1000 premières lignes


    #data = c.getData(TRAINING_SET_FOLDER)

    import os
    data = []
    filenames = os.listdir(TRAINING_SET_FOLDER)
    filenames = filenames[:3] # mon ajout
    for filename in filenames:
        print(os.path.join(TRAINING_SET_FOLDER, filename))

        with open(os.path.join(TRAINING_SET_FOLDER, filename), 'r') as f:
            for line in f:

                line2 = line.strip().split('\t')
                if len(line2) == 2:
                    data.append(line2)
    #print(len(data))
    #data = data[:500]

    N = len(data)
    limit = (4*N)//5   # limite entre indices d'entraînement et de test

    labels_training = [line[0] for line in data[:limit]]
    X_training = [line[1] for line in data[:limit]]

    X_token_tr = c.tokenize(X_training)
    labels_bin_tr = [ 1 if label in ['Positive'] else 0 for label in labels_training]
    labels_bin_tr = np.array(labels_bin_tr)




    labels_learning = [line[0] for line in data[limit:]]
    X_learning = [line[1] for line in data[limit:]]

    X_token_te = c.tokenize(X_learning)
    labels_bin_te = [ 1 if label in ['Positive'] else 0 for label in labels_learning]
    labels_bin_te = np.array(labels_bin_te)




    print("Proportion de revues positives:",np.mean(labels_bin_te))


    C = c.MetaClassifier()
    # Under sampling : OSS
    """(I_kept,_,_) = Sampling(X_token_tr,labels_bin_tr,"OSS")
    X_token_tr,labels_bin_tr = X_token_tr[I_kept][:],labels_bin_tr[I_kept]
    """
    # Under sampling : Tomek
    (I_kept,_,_) = Sampling(X_token_tr,labels_bin_tr,"Tomek")
    X_token_tr,labels_bin_tr = X_token_tr[I_kept][:],labels_bin_tr[I_kept]


    # Over sampling : SMOTE
    (_,X_new,Y_new) = Sampling(X_token_tr,labels_bin_tr,"SMOTE")

    X_token_tr =  np.array(list(X_token_tr)+list(X_new))
    labels_bin_tr = np.array(list(labels_bin_tr)+list(Y_new))

    C.train(X_token_tr,labels_bin_tr)
    Y_pred = C.predict(X_token_te,Y = labels_bin_te)

    print("exemples de predictions :",Y_pred[:10])
    print(labels_bin_te[:10])

    print(np.mean([Y_pred == labels_bin_te]))
    print("entraînement et test :",time.time()-start)
