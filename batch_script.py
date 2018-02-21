"""
Ce fichier sert à faire tourner l'algorithme de transfer learning sur tous les ensembles deux à deux. (pas optimisé).
"""


import main

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
"data_apps", "data_office", "data_tools"]



for categorie1 in CATEGORIES[0:1]:


    ORIGIN_FOLDER_1 = "../data/" + categorie1

    print("\n################################")
    print("##                            ##")
    print("##   LEARNING FROM DATASET 1  ##")
    print("## " + ORIGIN_FOLDER_1 + " ##")
    print("##                            ##")
    print("################################")
    
    
    
    CLEANED_DATA_FOLDER_1 = "../data/" + categorie1 + "_cleaned"
    TRAINING_SET_FOLDER_1 = "../data/" + categorie1 + "_training_set"
    TESTING_SET_FOLDER_1 = "../data/" + categorie1 + "_testing_set"
            
    main.createTrainingSetAndTestSet(CLEANED_DATA_FOLDER_1, TRAINING_SET_FOLDER_1, TESTING_SET_FOLDER_1)
    model1 = main.learn(TRAINING_SET_FOLDER_1)
    main.showResults(model1, TESTING_SET_FOLDER_1)
    
    
    
    for categorie2 in CATEGORIES:
      
            
        ORIGIN_FOLDER_2 = "../data/" + categorie2
        CLEANED_DATA_FOLDER_2 = "../data/" + categorie2 + "_cleaned"
        TRAINING_SET_FOLDER_2 = "../data/" + categorie2 + "_training_set"
        TESTING_SET_FOLDER_2 = "../data/" + categorie2 + "_testing_set"


        
        
        
        print("\n################################")
        print("##                            ##")
        print("##  TRANSFER LEARNING (1->2)  ##")
        print("##" + ORIGIN_FOLDER_1 + " --> " + ORIGIN_FOLDER_2 + "##")
        print("##                            ##")
        print("################################")
        main.createTrainingSetAndTestSet(CLEANED_DATA_FOLDER_2, TRAINING_SET_FOLDER_2, TESTING_SET_FOLDER_2)
        model2 = main.transferLearn(model1, TRAINING_SET_FOLDER_2) # <---- Difference here!!
        main.showResults(model2, TESTING_SET_FOLDER_2)
        
        
