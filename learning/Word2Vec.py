from gensim.models.word2vec import Word2Vec
import numpy as np
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os 
import multiprocessing
from time import *

DIRECTORY = "C:/Users/Antoine/Documents/Centrale/3A/Transfer_learning/"
os.chdir(DIRECTORY)

from data_loader import *

class W2V():
    def __init__(self,vector_size,window_size,model_train_data,correlation_train_data,correlation_test_data):
        self.vector_size = vector_size
        self.window_size = window_size
        self.model = None
        self.model_train_data = model_train_data
        self.correlation_train_data = correlation_train_data
        self.correlation_test_data = correlation_test_data
        self.means = None
        self.correlations = None
    def train(self,save_filename):
        '''
        processed data must be an array of tokenized sentences
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
        n = len(self.correlation_test_data)
        self.means = [0 for i in range(self.vector_size)]
        for line in self.correlation_test_data:
            for i in range(self.vector_size):
                self.means[i] += line[1][i]/n
        good = 0
        for line in self.correlation_test_data:
            score = 0
            for i in range(self.vector_size): 
                score += self.correlations[i] * (line[1][i]-self.means[i])
            if score * (2*(line[0]>3)-1) > 0:
                good += 1
        return(100*good/n)
        


tkr = RegexpTokenizer('[a-zA-Z0-9@]+')
stemmer = LancasterStemmer()

model_train_data = []
correlation_train_data = []
correlation_test_data = []

loader = DataLoader("text_data")

data = loader.load_raw_data("apps.txt")[:10000]
for line in data:
    line[1].strip().lower()

n = len(data)

for i, line in enumerate(data):
    print("cleaning train data : ", int(100*i/n),"%")
    tok = tkr.tokenize(line[1])
    tokens = []
    for t in tok:
        if t not in stopwords.words('english'):
           tokens.append(stemmer.stem(t)) 
    correlation_train_data.append([line[0],tokens])
    
data = loader.load_raw_data("cell_phones.txt")[:10000]
for line in data:
    line[1].strip().lower()

n = len(data)

for i, line in enumerate(data):
    print("cleaning test data : ", int(100*i/n),"%")
    tok = tkr.tokenize(line[1])
    tokens = []
    for t in tok:
        if t not in stopwords.words('english'):
           tokens.append(stemmer.stem(t)) 
    correlation_test_data.append([line[0],tokens])

model_train_data = correlation_test_data + correlation_train_data

w = W2V(20,5,model_train_data,correlation_train_data,correlation_test_data)

print("training model")
w.train("testw2vclass")
print("tokens_to_vect")
w.tokens_to_vect()
print("computing correlations")
w.compute_correlations()
print(w.get_efficiency())

