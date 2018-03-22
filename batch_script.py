"""
Ce fichier sert à faire tourner l'algorithme de transfer learning sur tous les ensembles deux à deux. (pas trop optimisé).
"""


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
    print("##   LEARNING FROM DATASET 1  ##    " + ORIGIN_FOLDER_1)
    print("##                            ##")
    print("################################")
    
    
    TRAINING_SET_FOLDER_1 = PATH_TO_DATA + categorie1 + "_training_set"
    TESTING_SET_FOLDER_1 = PATH_TO_DATA + categorie1 + "_testing_set"
    
    
    model1 = main.createModelsAndLearn(TRAINING_SET_FOLDER_1)
    main.testModels(model1, TESTING_SET_FOLDER_1)

    for categorie2 in CATEGORIES:
            
        ORIGIN_FOLDER_2 = PATH_TO_DATA + categorie2
        TRAINING_SET_FOLDER_2 = PATH_TO_DATA + categorie2 + "_training_set"
        TESTING_SET_FOLDER_2 = PATH_TO_DATA + categorie2 + "_testing_set"

        
        print("\n################################")
        print("##                            ##")
        print("##  TRANSFER LEARNING (1->2)  ##    " + ORIGIN_FOLDER_1 + " --> " + ORIGIN_FOLDER_2)
        print("##                            ##")
        print("################################")

        model2 = main.transferLearn(model1, TRAINING_SET_FOLDER_2) # <---- Difference here!!
        main.testModels(model2, TESTING_SET_FOLDER_2)
        
