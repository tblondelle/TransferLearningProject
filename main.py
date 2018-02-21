import os

from data.scripts import json_to_text, cleaner, createDatasets
import learning.Classifiers as clf


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
    print("========================")
    print("|        TRAIN         |")
    print("========================")

    return clf.learn(training_set_folder, dataBalancing=True)



def transferLearn(old_model, training_set_folder):
    new_model = old_model
    # Change the identity function to something more complex.
    return new_model



def showResults(model, testing_set_folder):
    print("========================")
    print("|        TEST          |")
    print("========================")
    clf.showResults(model, testing_set_folder)


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
    createTrainingSetAndTestSet(cleaned_data_folder_1, training_set_folder_1, testing_set_folder_1)
    
    model1 = learn(training_set_folder_1)
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
    


