import main

PATH_TO_DATA = "../data/"
CATEGORIES = ["data_books",
"data_videos",
"data_electronics",
"data_clothes",
"data_movies",
"data_cds",
"data_cellphones",
"data_health",
"data_kindle",
"data_digitalmusic",
"data_musicinstruments",
"data_sport",
"data_toys",
"data_baby",
"data_apps", 
"data_office", 
"data_tools"]


for categorie1 in CATEGORIES:

    ORIGIN_FOLDER_1 = PATH_TO_DATA + categorie1

    print("\n################################")
    print("##                            ##")
    print("##  Create dataset            ##    " + ORIGIN_FOLDER_1)
    print("##                            ##")
    print("################################")



    stripped_metadata_folder_1 = PATH_TO_DATA + categorie1 + "_stripped"
    CLEANED_DATA_FOLDER_1 = PATH_TO_DATA + categorie1 + "_cleaned"
    TRAINING_SET_FOLDER_1 = PATH_TO_DATA + categorie1 + "_training_set"
    TESTING_SET_FOLDER_1 = PATH_TO_DATA + categorie1 + "_testing_set"
    
    
    main.stripMetadata(ORIGIN_FOLDER_1, stripped_metadata_folder_1)
    main.simplifyRatingAndKeepRelevantWords(stripped_metadata_folder_1, CLEANED_DATA_FOLDER_1)    
    main.createTrainingSetAndTestSet(CLEANED_DATA_FOLDER_1, TRAINING_SET_FOLDER_1, TESTING_SET_FOLDER_1)
    
