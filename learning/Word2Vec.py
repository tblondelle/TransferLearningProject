from gensim.models.word2vec import Word2Vec
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os 
import multiprocessing
from time import *

DIRECTORY = "C:/Users/Antoine/Documents/Centrale/3A/Transfer_learning/" # The directory must contain the DataLoader script
os.chdir(DIRECTORY)

from data_loader import *

class W2V():
    def __init__(self,vector_size,window_size,threshold_factor,model_train_data,correlation_train_data,correlation_test_data):
        '''
        This class is used to train a word2vec model and use it to determine the note of a review
        First, the word2vec model is trained on the model_train_data corpus
        Then the correlations between the values of the vector and the note of the reviews from the correlation_train_data corpus are computed
        Then we use those correlations to determine the notes of the correlation_test_data corpus
        
        NOTE : The model_train_data corpus should contain reviews from both the train and test corpuses for increased efficiency.
        '''
        self.vector_size = vector_size
        self.window_size = window_size 
        self.model = None
        self.model_train_data = model_train_data
        self.correlation_train_data = correlation_train_data
        self.correlation_test_data = correlation_test_data
        self.means = None
        self.correlations = None
        self.threshold_factor = threshold_factor
        self.threshold = 0
    def train(self,save_filename):
        '''
        Will train the Word2Vec model and save it in the file save_filename, the model can be loaded later with the load_model method
        '''
        train_data = []
        for line in self.model_train_data:
             train_data.append(line[1])
        self.model = Word2Vec(sentences=train_data,
            size=self.vector_size, 
            window=self.window_size, 
            negative=20,
            iter=50,
            seed=1000,
            workers=multiprocessing.cpu_count())
        self.model.save("save_filename")
    def load_model(self,save_filename):
        self.model = Word2Vec.load(save_filename)
    def tokens_to_vect(self):
        '''
        Transforms the tokenized text from the train and test datasets into vectors using the Word2Vec model 
        '''
        vects = []
        for line in self.correlation_train_data:
            vect_line = []
            vect_line.append(line[0])
            vect_line.append(np.zeros(self.vector_size))
            for token in line[1]:
                try:
                    vect_line[1] += self.model[token]
                except:
                    ()
            norm = np.linalg.norm(vect_line[1])
            if norm>0:
                vect_line[1] = vect_line[1]/norm
            vects.append(vect_line)
        self.correlation_train_data = vects
        
        vects = []
        for line in self.correlation_test_data:
            vect_line = []
            vect_line.append(line[0])
            vect_line.append(np.zeros(self.vector_size))
            for token in line[1]:
                try:
                    vect_line[1] += self.model[token]                    
                except:
                    ()
            norm = np.linalg.norm(vect_line[1])
            if norm>0:
                vect_line[1] = vect_line[1]/norm
            vects.append(vect_line)
        self.correlation_test_data = vects
    def compute_correlations(self):
        '''
        Computes the correlations between the values of the vectors and the notes of the reviews from the training set
        '''
        self.means = [0 for i in range(self.vector_size)]
        self.correlations = [0 for i in range(self.vector_size)]
        n = len(self.correlation_train_data)
        for line in self.correlation_train_data:
            for i in range(self.vector_size):
                self.means[i] += line[1][i]/n
        for line in self.correlation_train_data:
            for i in range(self.vector_size):
                self.correlations[i] += (line[1][i]-self.means[i])*(2*(line[0]>3)-1)
    def get_efficiency(self):
        '''
        Tries to guess the notes of the test set and returns the efficiency
        '''
        n = len(self.correlation_test_data)
        self.means = [0 for i in range(self.vector_size)]
        for line in self.correlation_test_data:
            for i in range(self.vector_size):
                self.means[i] += line[1][i]/n
        n_ignored = 0
        n_treated = 0
        good = 0
        for line in self.correlation_test_data:
            n_treated += 1
            score = 0
            for i in range(self.vector_size): 
                score += self.correlations[i] * (line[1][i]-self.means[i])
            if abs(score)<self.threshold:
                n_ignored += 1
            else:
                if score * (2*(line[0]>3)-1) > 0:
                    good += 1
        return("efficiency : "+str(100*good/(n-n_ignored)) + "% \n reviews ignored : "+str(n_ignored) )
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
    def predict(self,review):
        '''
        Returns 0 if the score of the review is too close to 0 (can not determine if positive or negative)
        Returns -1 if negative and 1 if positive
        '''
        review = review.split()
        vect = np.zeros(self.vector_size)
        for token in review:
            try:
                vect += self.model[token]                    
            except:
                ()
        norm = np.linalg.norm(vect)
        if norm>0:
            vect = vect/norm
        score = 0
        for i in range(self.vector_size): 
            score += self.correlations[i] * (vect[i]-self.means[i])
        if abs(score)<self.threshold:
            return(0)
        return(2*(score>0)-1)
            
        
### CLEANING        

loader = DataLoader("text_data") #The text files that will be loaded must be contained in DIRECTORY/text_data

STOPWORDS = stopwords.words('english')
tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

model_train_data = []
correlation_train_data = []
correlation_test_data = []

# Loading the training data
data = loader.load_raw_data("apps.txt")
for line in data:
    line[1].strip().lower()

n = len(data)

# Cleaning the correlations training data
for i, line in enumerate(data):
    if i%int(n/100) == 0:
        print("cleaning train data : ", int(100*i/n),"%")
    tok = tkr.tokenize(line[1])
    tokens = []
    for t in tok:
        if t not in STOPWORDS:
           tokens.append(stemmer.stem(t)) 
    correlation_train_data.append([line[0],tokens])
    
# Loading the test data
data = loader.load_raw_data("cell_phones.txt")
for line in data:
    line[1].strip().lower()

n = len(data)

# Cleaning the test data
for i, line in enumerate(data):
    if i%int(n/100) == 0:
        print("cleaning test data : ", int(100*i/n),"%")
    tok = tkr.tokenize(line[1])
    tokens = []
    for t in tok:
        if t not in STOPWORDS:
           tokens.append(stemmer.stem(t)) 
    correlation_test_data.append([line[0],tokens])

# Creating the Word2Vec training data
model_train_data = correlation_test_data + correlation_train_data

### Création et apprentissage du Word2Vec et calcul des corrélations

w = W2V(20,5,0.8,model_train_data,correlation_train_data,correlation_test_data)

print("training model")
w.train("testw2vclass")

print("tokens_to_vect")
w.tokens_to_vect()

print("computing correlations")
w.compute_correlations()

print("computing threshold")
w.compute_threshold()

print(w.get_efficiency())

