from data.scripts import convertJsonToText, cleanData, createDatasets
from learning import sklearnClassifier
from learning import word2vecClassifier


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



def stripMetadata(source_folder, target_folder):
    """
    Get files from the folder [source_folder], 
    Perfom the stripping
    Put the results in file (or files) in target_folder
    """
    convertJsonToText.JsonHandler(source_folder, target_folder).convert()

def simplifyRatingAndKeepRelevantWords(source_folder, target_folder):
    """
    Get files from the folder [source_folder], 
    Simplify the rating and keep relevant words only
    Put the results in file (or files) in target_folder
    """
    cleanData.TextCleaner(source_folder, target_folder).clean()
    
    
def createTrainingSetAndTestSet(source_folder, target_training_set_folder, target_testing_set_folder):
    nb_train, nb_test = createDatasets.createDataset(source_folder, target_training_set_folder, target_testing_set_folder)
    print("{} files written in {}".format(nb_train, target_training_set_folder))
    print("{} files written in {}".format(nb_test, target_testing_set_folder))



def createModelsAndlearn(training_set_folder):
    print("========================")
    print("|        TRAIN         |")
    print("========================")

    # Create the models and make them learn.
    sklearn_classifier = sklearnClassifier.MetaClassifier(validation_rate=0.1, n_features=150)
    sklearn_classifier.train(TRAINING_SET_FOLDER_1, dataBalancing=True)

    word2vec_classifier = word2vecClassifier.W2V(20,5)
    word2vec_classifier.train(TRAINING_SET_FOLDER_1, dataBalancing=True)

    return [sklearn_classifier, word2vec_classifier]


def transferLearn(old_models, training_set_folder):
    new_models = []

    for old_model in old_models:
        if old_model.name == "sklearn_classifier":
            new_models.append(old_model)
        if old_model.name == "word2vec_classifier":
            old_model.train(training_set_folder, dataBalancing=True)
            new_models.append(old_model)

    return new_models



def showResults(models, testing_set_folder):
    print("========================")
    print("|        TEST          |")
    print("========================")

    for model in models:
        print("\n\n" + model.name)
        model.showResults(testing_set_folder)


def main(origin_folder_1=ORIGIN_FOLDER_1, origin_folder_2=ORIGIN_FOLDER_2, 
    stripped_metatdata_folder_1=STRIPPED_METADATA_FOLDER_1, stripped_metatdata_folder_2=STRIPPED_METADATA_FOLDER_2,
    cleaned_data_folder_1=CLEANED_DATA_FOLDER_1, cleaned_data_folder_2=CLEANED_DATA_FOLDER_2,
    training_set_folder_1=TRAINING_SET_FOLDER_1, training_set_folder_2=TRAINING_SET_FOLDER_2,
    testing_set_folder_1=TESTING_SET_FOLDER_1, testing_set_folder_2=TESTING_SET_FOLDER_2):

    print("\n################################")
    print("##                            ##")
    print("##       PREPROCESSING        ##")
    print("##                            ##")
    print("################################")
    #stripMetadata(origin_folder_1, stripped_metatdata_folder_1)
    #simplifyRatingAndKeepRelevantWords(stripped_metatdata_folder_1, cleaned_data_folder_1)

    #stripMetadata(origin_folder_2, stripped_metatdata_folder_2)
    #simplifyRatingAndKeepRelevantWords(stripped_metatdata_folder_2, cleaned_data_folder_2)

    
    print("\n################################")
    print("##                            ##")
    print("##   LEARNING FROM DATASET 1  ##")
    print("##                            ##")
    print("################################")
    #createTrainingSetAndTestSet(cleaned_data_folder_1, training_set_folder_1, testing_set_folder_1)
    
    model1 = createModelsAndlearn(training_set_folder_1)
    showResults(model1, testing_set_folder_1)

    

    print("\n################################")
    print("##                            ##")
    print("##  TRANSFER LEARNING (1->2)  ##")
    print("##                            ##")
    print("################################")
    createTrainingSetAndTestSet(cleaned_data_folder_2, training_set_folder_2, testing_set_folder_2)
    model2 = transferLearn(model1, training_set_folder_2) # <---- Difference here!!
    showResults(model2, testing_set_folder_2)


if __name__ == "__main__":
    main()
    


