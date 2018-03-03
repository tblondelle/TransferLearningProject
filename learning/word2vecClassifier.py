import numpy as np
import multiprocessing
from gensim.models.word2vec import Word2Vec
import os
from random import shuffle, seed



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



class W2V():
    def __init__(self, vector_size, window_size):
        '''
        This class is used to train a word2vec model and use it to determine the note of a review
        First, the word2vec model is trained on the model_train_data corpus
        Then the correlations between the values of the vector and the note of the reviews from the correlation_train_data corpus are computed
        Then we use those correlations to determine the notes of the correlation_test_data corpus
        
        NOTE : The model_train_data corpus should contain reviews from both the train and test corpuses for increased efficiency.
        '''
        self.name = "word2vec_classifier"
        self.vector_size = vector_size
        self.window_size = window_size 
        self.model = None
        self.training_data = []
        self.threshold = 0
        self.std_vector = np.zeros(vector_size)
    
    def train(self, training_set_folder, dataBalancing=False, save_filename="save_filename"):
        '''
        Will train the Word2Vec model and save it in the file save_filename, the model can be loaded later with the load_model method
        '''

        train_data = getData(training_set_folder)#[:50] #Pour ALLER PLUS VITE LORS DES TEST !!

        if dataBalancing:
            train_data = balanceData(train_data)
            print("{} lines of data kept after balancing".format(len(train_data)))

        for data in train_data:
            data[1] = data[1].split(" ")


        self.training_data += train_data

        train_data = [line[1] for line in self.training_data]
        print(len(train_data))
        
        self.model = Word2Vec(sentences=train_data,
            size=self.vector_size, 
            window=self.window_size, 
            negative=20,
            iter=50,
            seed=1000,
            workers=multiprocessing.cpu_count())

        print("computing correlations")
        self.compute_correlations(self.tokens_to_vect(self.training_data))


    def compute_correlations(self, vects_train_data):
        '''
        Computes the correlations between the values of the vectors and the notes of the reviews from the training set
        '''
        n = len(vects_train_data)
        mean_vector = np.zeros(self.vector_size)
        for line in vects_train_data:
            mean_vector += line[1]/n
        
        for line in vects_train_data:
            self.std_vector += (line[1] - mean_vector) * (1 if (line[0] == 'Positive') else -1)

    def tokens_to_vect(self, data):
        '''
        Transforms the tokenized text from the train and test datasets into vectors using the Word2Vec model 
        '''
        vects = []
        for line in data:
            vect_line = [line[0], np.zeros(self.vector_size)]
            for token in line[1]:
                try:
                    vect_line[1] += self.model[token]
                except:
                    ()
            # Normalize the vector.
            norm = np.linalg.norm(vect_line[1])
            if norm != 0:
                vect_line[1] = vect_line[1]/norm

            vects.append(vect_line)
        return vects

  
    def get_efficiency(self, correlation_test_data):
        '''
        Tries to guess the notes of the test set and returns the efficiency
        '''
        n = len(correlation_test_data)
        mean_vector = np.zeros(self.vector_size)
        for line in correlation_test_data:
            mean_vector += line[1]/n
        
        good = 0
        for line in correlation_test_data:
            score = np.matmul(np.array(self.std_vector), line[1] - np.array(mean_vector))
            if score * (1 if (line[0] == 'Positive') else -1) > self.threshold:
                good += 1
        return good/n



    def showResults(self, testing_set_folder):

        test_data = getData(testing_set_folder)

        for data in test_data:
            data[1] = data[1].split(" ")


        vects_test_data = self.tokens_to_vect(test_data)

        print("\n== TEST RESULTS ==")
        print("  Taux de succès : ............................. {:.3f}".format(self.get_efficiency(vects_test_data)))



if __name__ == "__main__":

    TRAINING_SET_FOLDER_1 = "../../data/data_videos_training_set"
    TESTING_SET_FOLDER_1 = "../../data/data_videos_testing_set"
    #TESTING_SET_FOLDER_1 = TRAINING_SET_FOLDER_1
    
    print("========================")
    print("|        TRAIN         |")
    print("========================")
    model = W2V(20,5)
    model.train(TRAINING_SET_FOLDER_1, dataBalancing=True)


    print("========================")
    print("|        TEST          |")
    print("========================")
    model.showResults(TESTING_SET_FOLDER_1)
