from data.scripts import convertJsonToText, cleanData, createDatasets
from learning import sklearnClassifier, word2vecClassifier
from random import sample, seed
import numpy as np
import sklearn.metrics as skm

seed(8080)

# Origin datafile downloaded from jmcauley.ucsd.edu/data/amazon/ in ORIGIN_FOLDER_1
ORIGIN_FOLDER_1 = "../data/data_books"
ORIGIN_FOLDER_2 = "../data/data_videos" # eg (we will learn video data from books)

# In this folder are files such that : 
# - every line is of the form "[original ratings]\t[original review]"
STRIPPED_METADATA_FOLDER_1 = ORIGIN_FOLDER_1 + "_stripped"
STRIPPED_METADATA_FOLDER_2 = ORIGIN_FOLDER_2 + "_stripped"

# In this folder are files such that : 
# - every line is of the form "[new ratings]\t[list of relevant words]" with [new ratings] in {"Negative", "Neutral", "Positive"}
CLEANED_DATA_FOLDER_1 = ORIGIN_FOLDER_1 + "_cleaned"
CLEANED_DATA_FOLDER_2 = ORIGIN_FOLDER_2 + "_cleaned"

TRAINING_SET_FOLDER_1 = ORIGIN_FOLDER_1 + "_training_set"
TESTING_SET_FOLDER_1 = ORIGIN_FOLDER_1 + "_testing_set"
TRAINING_SET_FOLDER_2 = ORIGIN_FOLDER_2 + "_training_set"
TESTING_SET_FOLDER_2 = ORIGIN_FOLDER_2 + "_testing_set"



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



def createModelsAndLearn(training_set_folder):

    print("\n##\n## Retrieving training data\n##")
    data = sklearnClassifier.getData(training_set_folder)[:5000]
    data = sklearnClassifier.balanceData(data)
    print("{} lines of data".format(len(data)))
    labels, X = zip(*data)
    Y = sklearnClassifier.binariseLabels(labels)    
    print("Ratio of positive reviews: {:.3f}".format(np.mean(Y)))


    models = []

    # Create the models and make them learn.
    print("\n##\n## MetaClassifier Sklearn\n##")
    sklearn_classifier = sklearnClassifier.MetaClassifier(validation_rate=0.1, n_features=150)
    sklearn_classifier.train(X, Y)
    models.append(sklearn_classifier)

    print("\n##\n## Word2Vec Classifier\n##")
    word2vec_classifier = word2vecClassifier.W2V(20,5)
    word2vec_classifier.train(X, Y)
    models.append(word2vec_classifier)
    
    return models


def transferLearn(old_models, training_set_folder):

    print("\n##\n## Retrieving training data\n##")
    data = sklearnClassifier.getData(training_set_folder)[:5000]
    data = sklearnClassifier.balanceData(data)
    print("{} lines of data".format(len(data)))
    labels, X = zip(*data)
    Y = sklearnClassifier.binariseLabels(labels)    
    print("Ratio of positive reviews: {:.3f}".format(np.mean(Y)))



    new_models = []

    for old_model in old_models:
        if old_model.name == "sklearn_classifier":
            print("\n##\n## Adding old sklearn_classifier...\n##")
            new_models.append(old_model)
        elif old_model.name == "word2vec_classifier":
            print("\n##\n## Training old word2vec_classifier...\n##")
            old_model.train(X, Y)
            new_models.append(old_model)

    return new_models



def testModels(models, testing_set_folder):

    def getSampleResults(y_pred, y_true):
        index_sample = sample(range(len(y_true)), k=10)
        return [int(Y_pred[i]) for i in index_sample], [int(Y[i]) for i in index_sample]


    print("\n##\n## Retrieving testing data\n##")
    data = sklearnClassifier.getData(testing_set_folder)
    print("{} lines of data".format(len(data)))
    labels, X = zip(*data)
    Y = sklearnClassifier.binariseLabels(labels)


    
    for model in models:
        print("\n##\n## Testing {}\n##".format(model.name))
        Y_pred, success_rate = model.test(X, Y)

        sample_pred, sample_true = getSampleResults(Y_pred, Y)

        print("Number of predictions : {}".format(len(Y_pred)))
        print("Predictions sample:        {}".format(sample_pred))
        print("True classes of sample:    {}".format(sample_true))
        print("Ratio of positive reviews (test data):   {:.3f}".format(np.mean([Y])))
        print("Ratio of positive reviews (predictions): {:.3f}".format(np.mean(Y_pred)))
        print("Precision score: {:.3f}".format(skm.precision_score(Y, Y_pred))) # tp / (tp + fp)
        print("Recall score: {:.3f}".format(skm.recall_score(Y, Y_pred))) # tp / (tp + fn)
        print("Accuracy score : ................................................. {:.3f}".format(success_rate))



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
    
    model1 = createModelsAndLearn(training_set_folder_1)
    testModels(model1, testing_set_folder_1)

    

    print("\n################################")
    print("##                            ##")
    print("##  TRANSFER LEARNING (1->2)  ##")
    print("##                            ##")
    print("################################")
    #createTrainingSetAndTestSet(cleaned_data_folder_2, training_set_folder_2, testing_set_folder_2)
    model2 = transferLearn(model1, training_set_folder_2) # <---- Difference here!!
    testModels(model2, testing_set_folder_2)


if __name__ == "__main__":
    main()