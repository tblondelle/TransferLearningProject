import os
import numpy as np
import multiprocessing
from random import shuffle, seed
from gensim.models.word2vec import Word2Vec


np.random.seed(1000)
seed(1000)

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

def binariseLabels(Y):
    Y_bin = []
    for i, label in enumerate(Y):
        if Y[i] in ['Negative', 'Neutral']:
            Y_bin.append(0)
        elif Y[i] == 'Positive':
            Y_bin.append(1)
    return Y_bin

class W2V():
    def __init__(self, vector_size, window_size, threshold_factor=0.8):
        '''
        This class is used to train a word2vec model and use it to 
        determine the note of a review.

        First, the word2vec model is trained on the model_train_data corpus.
        Then the correlations between the values of the vector and the 
        note of the reviews from the correlation_train_data corpus are 
        computed.
        Then we use those correlations to determine the notes of the 
        correlation_test_data corpus.
        
        We use a threshold in order to not take into account the reviews 
        for which the note is unsure (if the score is too close to 0), 
        this threshold is the mean value of the score times threshold_factor. 
        the higher threshold factor, the more reviews will be ignored.
        
        NOTE : The model_train_data corpus should contain reviews from 
        both the train and test corpuses for increased efficiency.
        '''
        self.name = "word2vec_classifier"
        self.vector_size = vector_size
        self.window_size = window_size 
        self.model = None
        self.training_data = []
        self.threshold = 0
        self.threshold_factor = threshold_factor
        self.std_vector = np.zeros(vector_size)
    
    def train(self, X, Y, save_filename="save_filename"):
        '''
        Train the Word2Vec model and save it in the file save_filename, 
        the model can be loaded later with the load_model method.
        '''


        # Add to the model the data as a list of [label, sentence].
        # If train is run multiple times, old data is not lost.
        self.training_data += list(zip(Y, X))
        

        # Run the word2vec model on the data only (no label)
        labels, train_data = zip(*self.training_data)
        self.model = Word2Vec(sentences=[sentence.split() for sentence in list(train_data)],
            size=self.vector_size, 
            window=self.window_size, 
            negative=20,
            iter=50,
            seed=1000,
            workers=multiprocessing.cpu_count())
        
        # Store the global standart error vector of the model.
        vects = self.sentencesToVects(train_data)
        self.std_vector = self.setCorrelation(labels, vects)

        
    def setCorrelation(self, labels, vects):
        '''
        Computes the correlation between the values of the vectors and 
        the notes of the reviews from the training set
        '''

        mean_vector = np.zeros(self.vector_size)
        for vect in vects:
            mean_vector += vect
        mean_vector /= len(vects)
        
        std_vector = np.zeros(self.vector_size)
        for i in range(vects.shape[0]):
            if labels[i] == 1:
                std_vector += (vects[i, :] - mean_vector)
            elif labels[i] == 0:
                std_vector -= (vects[i, :] - mean_vector)
            else:
                print('There is probably an error')

        return std_vector

    def sentencesToVects(self, X):
        '''
        Transforms the tokenized text from the train and test 
        datasets into vectors using the Word2Vec model 
        '''

        sentences = [sentence.split() for sentence in X]

        vects = np.zeros((len(X), self.vector_size))
        for i, sentence in enumerate(sentences):
            for token in sentence:
                try:
                    self.model[token]
                except KeyError as err:
                    #print("KeyError for the word: {}".format(token))
                    pass 
                else:               
                    vects[i, :] += self.model[token]
            # Normalize the vector.
            norm = np.linalg.norm(vects[i, :])
            if norm != 0:
                vects[i, :] /= norm

        return vects

    def compute_threshold(self):
        '''
        Compute the threshold above which a score is considered as significant 
        '''
        n = len(self.correlation_test_data)
        self.means = [0 for i in range(self.vector_size)]
        for line in self.correlation_test_data:
            for i in range(self.vector_size):
                self.means[i] += line[1][i]/n
        mean_deviation = 0
        for line in self.correlation_test_data:
            score = 0
            for i in range(self.vector_size): 
                score += self.correlations[i] * (line[1][i]-self.means[i])
            mean_deviation += abs(score)/n
        self.threshold = self.threshold_factor*mean_deviation
        

    def predict(self, X):
        '''
        Returns 0 if the score of the review is too close to 0 (can not determine if positive or negative)
        Returns -1 if negative and 1 if positive
        '''

        X = model.sentencesToVects(X)

        n = len(X)
        mean_vector = np.mean(X)

        Y_pred = np.zeros(n)
        for i, vect in enumerate(X):
            score = np.matmul(self.std_vector, vect - mean_vector)
            if score > self.threshold:
                Y_pred[i] = 1
            elif score < -self.threshold:
                Y_pred[i] = 0

            if abs(score)<self.threshold:
                n_ignored += 1
                Y_pred[i] = -1

        return Y_pred, n


    def test(self, Y, X):


        X = model.sentencesToVects(X)

        n = len(Y)
        mean_vector = np.mean(X)

        n_ignored = 0
        Y_pred = np.zeros(n)
        for i, vect in enumerate(X):
            score = np.matmul(self.std_vector, vect - mean_vector)
            if score > self.threshold:
                Y_pred[i] = 1
            elif score < -self.threshold:
                Y_pred[i] = 0

            if abs(score)<self.threshold:
                n_ignored += 1
                Y_pred[i] = -1

        success_rate = np.mean(Y_pred == np.array(Y))

        return Y_pred, success_rate, n



if __name__ == "__main__":

    TRAINING_SET_FOLDER_1 = "../../data/data_books_training_set"
    TESTING_SET_FOLDER_1 = "../../data/data_books_testing_set"
    #TESTING_SET_FOLDER_1 = TRAINING_SET_FOLDER_1
    


    print("========================")
    print("|        TRAIN         |")
    print("========================")
    # On fournit les labels Y sous forme binaire (0 ou 1)
    # et les données X sous forme de liste de phrases.
    data = getData(TRAINING_SET_FOLDER_1)[:100]
    data = balanceData(data)

    labels, X = zip(*data)
    Y = binariseLabels(labels)    


    model = W2V(20,5)
    model.train(X, Y)


    print("========================")
    print("|        TEST          |")
    print("========================")


    data = getData(TESTING_SET_FOLDER_1)[:60]
    labels, X = zip(*data)
    Y = binariseLabels(labels)



    preds, success_rate, n_treated = model.test(Y, X) 
    print(preds)
    print(success_rate)
    print(n_treated)
