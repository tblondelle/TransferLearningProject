"""
Ce fichier sert à faire tourner l'algorithme de transfer learning sur tous les ensembles deux à deux. (pas optimisé).
"""


import main

"data_books"
"data_videos"
"data_electronics"
"data_clothes"
"data_movies"
"data_cds"
"data_cellphones"
"data_health"
"data_kindle"
"data_digitalmusic"
"data_musicinstruments"
"data_sport"
"data_toys"

CATEGORIES = ["data_baby", "data_app", "data_office", "data_tools"]



for i, categorie1 in enumerate(CATEGORIES):
    for categorie2 in CATEGORIES[i+1:]:
        
        
        # Origin datafile downloaded from jmcauley.ucsd.edu/data/amazon/ in ORIGIN_FOLDER_1
        ORIGIN_FOLDER_1 = "../data/" + categorie1
        ORIGIN_FOLDER_2 = "../data/" + categorie2

        # In this folder are files such that : 
        # - every line is of the form "[original ratings]\t[original review]"
        STRIPPED_METADATA_FOLDER_1 = "../data/" + categorie1 + "_stripped"
        STRIPPED_METADATA_FOLDER_2 = "../data/" + categorie2 + "_stripped"

        # In this folder are files such that : 
        # - every line is of the form "[new ratings]\t[list of relevant words]" with [new ratings] in {"Negative", "Neutral", "Positive"}
        CLEANED_DATA_FOLDER_1 = "../data/" + categorie1 + "_cleaned"
        CLEANED_DATA_FOLDER_2 = "../data/" + categorie2 + "_cleaned"

        TRAINING_SET_FOLDER_1 = "../data/" + categorie1 + "_training_set"
        TESTING_SET_FOLDER_1 = "../data/" + categorie1 + "_testing_set"
        TRAINING_SET_FOLDER_2 = "../data/" + categorie2 + "_training_set"
        TESTING_SET_FOLDER_2 = "../data/" + categorie2 + "_testing_set"

        """main.main(origin_folder_1=ORIGIN_FOLDER_1, origin_folder_2=ORIGIN_FOLDER_2, 
            stripped_metatdata_folder_1=STRIPPED_METADATA_FOLDER_1, stripped_metatdata_folder_2=STRIPPED_METADATA_FOLDER_2,
            cleaned_data_folder_1=CLEANED_DATA_FOLDER_1, cleaned_data_folder_2=CLEANED_DATA_FOLDER_2,
            training_set_folder_1=TRAINING_SET_FOLDER_1, training_set_folder_2=TRAINING_SET_FOLDER_2,
            testing_set_folder_1=TESTING_SET_FOLDER_1, testing_set_folder_2=TESTING_SET_FOLDER_2)
        """
        print(ORIGIN_FOLDER_1)
        #print(ORIGIN_FOLDER_2)

    # Origin datafile downloaded from jmcauley.ucsd.edu/data/amazon/ in ORIGIN_FOLDER_1
    ORIGIN_FOLDER_1 = "../data/" + categorie1

    # In this folder are files such that : 
    # - every line is of the form "[original ratings]\t[original review]"
    STRIPPED_METADATA_FOLDER_1 = "../data/" + categorie1 + "_stripped"

    # In this folder are files such that : 
    # - every line is of the form "[new ratings]\t[list of relevant words]" with [new ratings] in {"Negative", "Neutral", "Positive"}
    CLEANED_DATA_FOLDER_1 = "../data/" + categorie1 + "_cleaned"

    TRAINING_SET_FOLDER_1 = "../data/" + categorie1 + "_training_set"
    TESTING_SET_FOLDER_1 = "../data/" + categorie1 + "_testing_set"


    main.stripMetadata(ORIGIN_FOLDER_1, STRIPPED_METADATA_FOLDER_1)
    main.simplifyRatingAndKeepRelevantWords(STRIPPED_METADATA_FOLDER_1, CLEANED_DATA_FOLDER_1)

