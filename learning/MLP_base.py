# -*- coding: utf-8 -*-
from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import os
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


from numpy.random import permutation

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import time
import math

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
print("Utilisation de la carte graphique :",use_cuda)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


class my_RNN(nn.Module):
    def __init__(self, input_size, hidden_size,batch_size, n_layers=1):
        super(my_RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size


        self.linear1 = nn.Linear(input_size,hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)


    def forward(self, input):
        # Entrées :
        #   input (variable(mat)) : les instances
        # Sortie
        #   Variable(vect) : les prédictions
        X_int_1 = F.relu(self.linear1(input))
        X_int_2 = F.relu(self.linear2(X_int_1))
        return torch.tanh(self.linear3(X_int_2))





    def train_once(self, input_variable, target_variable,  optimizer, criterion):
        # Réalise l'entraînement pour une seule epoch
        # Entrées :
        #   - n_epochs (int) : nombre de fois qu'on applique toutes les instance de l'ensemble d'apprentissage
        #   - input_variable Variable(mat) :  instances d'apprentissage
        #   - target_variable Variable(vect(+1|-1))) : labels
        #   - optimizer (pytorch object) : le résultat de optim.SGD ou optim.Adam
        #   - criterion (pytorch object) : le résultat de nn.L1Loss ou nn.MSELoss
        # Sorties :
        #   perte (float) : la valeur de la perte globale
        optimizer.zero_grad()
        input_length = input_variable.size()[0]

        output= self(input_variable)

        loss = criterion(output.view(1,-1), target_variable.view(-1))
        loss.backward()

        optimizer.step()

        return loss.data[0]



    def trainIters(self, n_epochs, training_pairs, te_pairs, learning_rate,  print_every=1000, eval_every = 1000):
        # Réalise l'entraînement complet, à partir des ensembles d'apprentissage
        # Entrées :
        #   - n_epochs (int) : nombre de fois qu'on applique toutes les instance de l'ensemble d'apprentissage
        #   - training_pairs (Variable(mat), Variable(vect(+1|-1))) ) : instances d'apprentissage
        #   - te_pairs (list of (Variable(vect), Variable(+1|-1))) : instances de test
        #   - learning_rate (float) : devine ;)
        #   - print_every (int) : imprime l'erreur moyenne toutes les print_every epochs
        #   - eval_every (int) : teste le NN sur la base de test et imprime la matrice de confusion
        # Sorties :
        #   none

        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every

        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        # Autre choix possible :
        #optimizer = optim.Adam(self.parameters(), lr=learning_rate)


        criterion = nn.L1Loss()
        #criterion = nn.MSELoss()

        for epoch in range(1, n_epochs + 1):

            input_variable = training_pairs[0]
            target_variable = training_pairs[1]

            loss = self.train_once(input_variable, target_variable,  optimizer, criterion)
            print_loss_total += loss

            if epoch % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                             epoch, epoch / n_epochs * 100, print_loss_avg))

            if epoch % eval_every == 0:
                self.evaluateRandomly(te_pairs) # show global results







    def evaluateRandomly(self, pairs):
        # evaluate on all pairs, print the confusion matrix
        n_successes = 0
        n_pos = 0 # also computes the proportion of positive reviews

        TP,TN,FP,FN = 0,0,0,0
        for pair in pairs:  # replace with pairs[:n] for testing
            output = self(pair[1])
            #success = (output[int(pair[1])] == max(output))
            note = pair[0].data[0,0]
            predicted = output.data[0]

            success = (note*predicted > 0)
            if success :
                n_successes += 1
                if note>0:
                    TP += 1
                else:
                    TN += 1
            else:
                if note>0:
                    FP += 1
                else:
                    FN += 1

            n_pos = n_pos+1 if note==1 else n_pos

        print('')
        print('')
        print('Confusion matrix ')
        print()
        print(" \t\t Actual class")
        print(" \t\t Pos \t Neg")
        print("Predicted Pos  \t {} \t {}".format(TP,FN))
        print("          Neg \t {} \t {}".format(FP,TN))
        print('')
        print('\t \t \t \t Positive reviews (%)) : ',100*n_pos/len(pairs))
        print('\t \t \t \t Success rate (%) : ',100*n_successes/len(pairs))


# overriding getData to only load 1 folder
def getData(folder):
    """
    Input:
     - folder: string of the path of a folder containing txt files.
    Output:
     - listdata: list of [Y, X] (e.g. Y = 'Positive', X = "very cool")
    """
    listdata = []

    filenames = os.listdir(folder)
    for filename in filenames: #[:1]:  # change here

        with open(os.path.join(folder, filename), 'r') as f:
            for line in f:

                line2 = line.strip().split('\t')
                if len(line2) == 2:
                    listdata.append(line2)
    return listdata


def folder2data(train_filename,test_filename,balanced_tr ,balanced_te, n_features):
    # Entrées :
    #   - train_filename (str) : le nom du **dossier** (et pas le nom du fichier) où se trouvent les instances d'apprentissage
    #   - test_filename (str) : le nom du **dossier** (et pas le nom du fichier) où se trouvent les instances de test
    #   - balanced_tr (bool) : True si l'ensemble d'apprentissage est équilibré; False s'il est laissé tel quel
    #   - balanced_te (bool) : True si l'ensemble de test est équilibré; False s'il est laissé tel quel
    #   - n_features (int) : nombre de variables pour coder chaque instance
    # Sorties :
    #   - cuple (new_tr_pairs, new_te_pairs):
    #       new_tr_pairs :  (Variable(mat), Variable(vect(+1|-1))) )
    #       new_te_pairs : (list of (Variable(vect), Variable(+1|-1)))

    tr_te_pairs = {}

    pairs = getData(train_filename)

    print(pairs[:2])


    if balanced_tr :
        #Pour un équilibrage 75/25
        pairs_using_numbers = [(-1,text)  for (target,text) in pairs  if (target == 'Negative' or target == 'Neutral')]
        Positive_reviews =  [(1,text) for (target,text) in pairs if target == 'Positive']
        pairs_using_numbers += Positive_reviews[:int(len(pairs_using_numbers)*3)] # différence ici
        tr_pairs = pairs_using_numbers
        """
        #Pour un équilibrage 50/50
        pairs_using_numbers = [(-1,text)  for (target,text) in pairs  if target == 'Negative']
        Positive_reviews =  [(1,text) for (target,text) in pairs if target == 'Positive']
        pairs_using_numbers += Positive_reviews[:int(len(pairs_using_numbers))]
        tr_pairs = pairs_using_numbers
        """
    else :
        pairs_using_numbers = [(1,text) for (target,text) in pairs if target == 'Positive']
        pairs_using_numbers += [(-1,text)  for (target,text) in pairs  if (target == 'Negative' or target == 'Neutral')]
        tr_pairs = pairs_using_numbers

    pairs = getData(test_filename)

    print(pairs[:2])

    if balanced_te :
        pairs_using_numbers = [(-1,text)  for (target,text) in pairs  if (target == 'Negative' or target == 'Neutral')]
        Positive_reviews =  [(1,text) for (target,text) in pairs if target == 'Positive']
        pairs_using_numbers += Positive_reviews[:int(len(pairs_using_numbers))]
        te_pairs = pairs_using_numbers

    else :
        pairs_using_numbers = [(1,text) for (target,text) in pairs if target == 'Positive']
        pairs_using_numbers += [(-1,text)  for (target,text) in pairs  if (target == 'Negative' or target == 'Neutral')]
        te_pairs = pairs_using_numbers


    print([text for (_,text) in tr_pairs[:2]])

    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,2))
    tfidf_vectorizer.fit([ text for (_,text) in tr_pairs+te_pairs])

    # fitting
    X_tr_token = tfidf_vectorizer.transform([ text for (_,text) in tr_pairs])
    X_te_token = tfidf_vectorizer.transform([ text for (_,text) in te_pairs])

    truncatedsvd = TruncatedSVD(n_components=n_features) # prépare à projeter les données dans un espace à n_components dimensions
    truncatedsvd.fit(X_tr_token)
    truncatedsvd.fit(X_te_token)


    # Réduction de dimension
    X_tr_reduced_dim = truncatedsvd.transform(X_tr_token)
    X_te_reduced_dim = truncatedsvd.transform(X_te_token)


    new_tr_pairs = [Variable(torch.FloatTensor(X_tr_reduced_dim)),Variable(torch.FloatTensor([[note for (note,_) in tr_pairs]]))]


    new_te_pairs = []
    for i in range(len(te_pairs)):
        (note,_) = te_pairs[i]
        note = Variable(torch.FloatTensor([[note]]))

        vect = X_te_reduced_dim[i,:]
        variable_vect = torch.autograd.Variable(torch.Tensor(vect))
        new_te_pairs.append((note,variable_vect))
    return new_tr_pairs, new_te_pairs












# ==================================================================
# ================ Using the MLP in itself =========================
# ==================================================================

training_set_folder = "../../data/data_books_training_set"
#test_set_folder = "../../data/data_videos_testing_set"
test_set_folder = "../../data/data_books_testing_set"



n_features=10
tr_pairs,te_pairs = folder2data(training_set_folder,test_set_folder,balanced_tr = True,balanced_te = True,n_features=n_features)

hidden_size = 5
batch_size = tr_pairs[0].data.size()[0]

RNN = my_RNN(n_features, hidden_size, batch_size, n_layers = 1)
#RNN.evaluateNpairs(te_pairs,1) # show some examples


lr = 0.005
N_epochs = 3
print("learning rate",lr)
print("batch_size",batch_size)
RNN.trainIters( N_epochs,tr_pairs,te_pairs,lr,500,5000)


RNN.evaluateRandomly(te_pairs) # show global results

torch.save(RNN,'MLP')
#cours ; cd 2eme_partie_S9/Transfer_learning/TransferLearningProject/learning/ ; python MLP_base.py


print('')
print('')

print('         Done')
print('')
print('')
print('')
