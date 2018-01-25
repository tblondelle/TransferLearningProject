import data.script.json_to_text as json_to_text





## What means xx_2 and xx_1 ?
# We will learn xx_2 from xx_1 via transferLearning. 


# Origin datafile downloaded from jmcauley.ucsd.edu/data/amazon/ in ORIGIN_FOLDER
ORIGIN_FOLDER_1 = "../data/data_books" # Books ie
ORIGIN_FOLDER_2 = "../data/data_video" # Video (we will learn video data from books)

# In this folder are files such that : 
# - every line is of the form "[original ratings]\t[original review]"
STRIPPED_METADATA_FOLDER_1 = "../data/data_books_stripped"
STRIPPED_METADATA_FOLDER_2 = "../data/data_videos_stripped"

# In this folder are files such that : 
# - every line is of the form "[new ratings]\t[list of relevant words]" with [new ratings] in {0, 1, 2}
CLEANED_DATA_FOLDER_1 = "../data/data_books_cleaned"
CLEANED_DATA_FOLDER_2 = "../data/data_videos_cleaned"

TRAINING_SET_FOLDER_1 = "../data/data_books_training_set"
TESTING_SET_FOLDER_1 = "../data/data_books_testing_set"
TRAINING_SET_FOLDER_2 = "../data/data_videos_training_set"
TESTING_SET_FOLDER_2 = "../data/data_videos_testing_set"



def stripMetadata(source_folder, target_folder):
    """
    Get files from the folder [source_folder], 
    Perfom the stripping
    Put the results in file (or files) in target_folder
    """
    jsonhandler = json_to_text.JsonHandler("C:/Users/Antoine/Documents/Centrale/3A/Transfer_learning","C:/Users/Antoine/Documents/Centrale/3A/Transfer_learning/text_data")
    jsonhandler.convert(["automotive.json"])
    pass

def simplifyRatingAndKeepRelevantWords(source_folder, target_folder):
    """
    Get files from the folder [source_folder], 
    Simplify the rating
    Put the results in file (or files) in target_folder
    """
    pass

def createTrainingSetAndTestSet(source_folder, target_training_set_folder, target_testing_set_folder):
    pass




if __name__ == "__main__":
    ## Pre-processing for the two datasets.
    stripMetadata(ORIGIN_FOLDER_1, STRIPPED_METADATA_FOLDER_1)
    simplifyRating(STRIPPED_METADATA_FOLDER_1, SIMPLIFIED_RATINGS_FOLDER_1)
    keepRelevantWords(SIMPLIFIED_RATINGS_FOLDER_1, CLEANED_DATA_FOLDER_1)

    stripMetadata(ORIGIN_FOLDER_2, STRIPPED_METADATA_FOLDER_2)
    simplifyRating(STRIPPED_METADATA_FOLDER_2, SIMPLIFIED_RATINGS_FOLDER_2)
    keepRelevantWords(SIMPLIFIED_RATINGS_FOLDER_2, CLEANED_DATA_FOLDER_2)


    # Learning
    createTrainingSetAndTestSet(CLEANED_DATA_FOLDER_1, TRAINING_SET_FOLDER_1, TESTING_SET_FOLDER_1)
    model1 = learn(TRAINING_SET_FOLDER_1)
    showResults(model1, TESTING_SET_FOLDER_1)

    # TransferLearning
    createTrainingSetAndTestSet(CLEANED_DATA_FOLDER_2, TRAINING_SET_FOLDER_2, TESTING_SET_FOLDER_2)
    model2 = transferLearn(model1, TRAINING_SET_FOLDER_2) # <---- Difference here!!
    showResults(model2, modelTESTING_SET_FOLDER_2)

