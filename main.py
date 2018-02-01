from data.scripts import json_to_text, cleaner, createDatasets
from learning.Classifiers import BaseClassifier


from itertools import islice  # sert à n'ouvrir que les N premières lignes d'un fichier

from sklearn.feature_extraction.text import CountVectorizer  # passe du texte brut à un vecteur
from sklearn.decomposition import TruncatedSVD

import os

# Origin datafile downloaded from jmcauley.ucsd.edu/data/amazon/ in ORIGIN_FOLDER_1
ORIGIN_FOLDER_1 = "../data/data_books"
ORIGIN_FOLDER_2 = "../data/data_video" # eg (we will learn video data from books)

# In this folder are files such that : 
# - every line is of the form "[original ratings]\t[original review]"
STRIPPED_METADATA_FOLDER_1 = "data/data_books_stripped"
STRIPPED_METADATA_FOLDER_2 = "data/data_videos_stripped"

# In this folder are files such that : 
# - every line is of the form "[new ratings]\t[list of relevant words]" with [new ratings] in {"Negative", "Neutral", "Positive"}
CLEANED_DATA_FOLDER_1 = "../data/data_books_cleaned"
CLEANED_DATA_FOLDER_2 = "../data/data_videos_cleaned"

TRAINING_SET_FOLDER_1 = "../data/data_books_training_set"
TESTING_SET_FOLDER_1 = "../data/data_books_testing_set"
TRAINING_SET_FOLDER_2 = "../data/data_videos_training_set"
TESTING_SET_FOLDER_2 = "../data/data_videos_testing_set"

DICT_WORDS = "../data/dict_words/20k.txt"

def stripMetadata(source_folder, target_folder):
    """
    Get files from the folder [source_folder], 
    Perfom the stripping
    Put the results in file (or files) in target_folder
    """
    json_to_text.JsonHandler(source_folder, target_folder).convert()

def simplifyRatingAndKeepRelevantWords(source_folder, target_folder):
    """
    Get files from the folder [source_folder], 
    Simplify the rating and keep relevant words only
    Put the results in file (or files) in target_folder
    """
    cleaner.TextCleaner(source_folder, target_folder).clean()
    
    
def createTrainingSetAndTestSet(source_folder, target_training_set_folder, target_testing_set_folder):
    nb_train, nb_test = createDatasets.createDataset(source_folder, target_training_set_folder, target_testing_set_folder)
    print("{} files written in {}".format(nb_train, target_training_set_folder))
    print("{} files written in {}".format(nb_test, target_testing_set_folder))



def learn(training_set_folder):
    # Opening file
    with open(training_set_folder) as file:
        N = 1000 # number of lines we keeeeep
        head = list(islice(file,N))
    
    X_rawtext,Y_train = [],[]
    for line in head:
        [y,x] = line.split('\t')
        
        y = 1 if y=='Positive' else (0 if y=='Neutral' else -1)
        #y = 1 if y>3 else (0 if y==3 else -1)
        
        if y!= 0:
            X_rawtext.append(x)
            Y_train.append(y)
    
    
    # Tokenization
    with open (DICT_WORDS, 'r') as f:
        Dict = f.read().split('\n')
    
    tokenizer = CountVectorizer()
    tokenizer.fit(Dict)
    
    X_token = tokenizer.transform(X_rawtext)
        
    truncatedsvd = TruncatedSVD(n_components=100)
    truncatedsvd.fit(tokenizer.transform(Dict))
    X_train = truncatedsvd.transform(X_token)
    
    # training
    # this part will change
    Clf = BaseClassifier()
    Clf.train(X_train,Y_train)
    
    return Clf





def transferLearn(old_model, training_set_folder):
    print("/!\ transferLearn not created yet.")
    pass

def showResults(model, testing_set_folder):
    print("/!\ showResults not created yet.")
    pass



if __name__ == "__main__":
    ## Pre-processing for the two datasets.
    print("\n--- PREPROCESSING ---")
    #stripMetadata(ORIGIN_FOLDER_1, STRIPPED_METADATA_FOLDER_1)
    #simplifyRatingAndKeepRelevantWords(STRIPPED_METADATA_FOLDER_1, CLEANED_DATA_FOLDER_1)

    #stripMetadata(ORIGIN_FOLDER_2, STRIPPED_METADATA_FOLDER_2)
    #simplifyRatingAndKeepRelevantWords(STRIPPED_METADATA_FOLDER_2, CLEANED_DATA_FOLDER_2)

    # Learning
    print("\n--- LEARNING FROM DATASET 1 ---")
    createTrainingSetAndTestSet(CLEANED_DATA_FOLDER_1, TRAINING_SET_FOLDER_1, TESTING_SET_FOLDER_1)
    
    DATA_PATHS = os.listdir(TRAINING_SET_FOLDER_1)
    TRAINING_PATH_1 =  TRAINING_SET_FOLDER_1+'/'+DATA_PATHS[0]
    model1 = learn(TRAINING_PATH_1)
    
    showResults(model1, TESTING_SET_FOLDER_1)

    # TransferLearning
    print("\n--- TRANSFER LEARNING FROM DATASET 1 APPLIED TO DATASET 2---")
    createTrainingSetAndTestSet(CLEANED_DATA_FOLDER_2, TRAINING_SET_FOLDER_2, TESTING_SET_FOLDER_2)
    model2 = transferLearn(model1, TRAINING_SET_FOLDER_2) # <---- Difference here!!
    showResults(model2, TESTING_SET_FOLDER_2)


