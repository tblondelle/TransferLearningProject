import os
from gensim.models import KeyedVectors
import numpy as np

DIRECTORY = "C:/Users/Antoine/Documents/Centrale/3A/Transfer_learning"
os.chdir(DIRECTORY)

from data_loader import *

class Word2Vec():
    def __init__(self,input_data_directory,input_data_filenames):
        self.input_data_directory = input_data_directory
        self.input_data_filenames = input_data_filenames
        self.loader = DataLoader(self.input_data_directory)
        print("Loading word2vec model ...")
        self.model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
        
    def text_to_vect(self,text):
        vect = np.array([0 for i in range(300)])
        for word in text.split():
            try:
                v = self.model[word]
                vect = vect+v
            except:
                ()        
        norm = np.linalg.norm(vect)
        if norm>0:
            vect = vect/np.linalg.norm(vect)
        return(vect)
        
    def run(self):
        for filename in self.input_data_filenames:
            print("Loading data from "+filename)
            data = self.loader.load(filename)
            n = len(data)
            result = []
            print("Converting data")
            for data_line in data:
                text = data_line[1]
                vect = self.text_to_vect(text)
                note = int(data_line[0])
                vect = np.insert(vect,0,note)
                result.append(vect)
            print("Saving result")
            np.savetxt("word2vec_"+filename,result)


test = Word2Vec("C:/Users/Antoine/Documents/Centrale/3A/Transfer_learning",["instruments.txt"])
test.run()


            


