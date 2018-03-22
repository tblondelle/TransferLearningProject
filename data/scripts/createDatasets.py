# -*- coding: utf-8 -*-

import os

PROPORTION_TRAIN = 0.9 # in ]0, 1[


def createDataset(source_location, target_training_set_folder, target_testing_set_folder):

    list_files = os.listdir(source_location)
    
    index_train_start = 0
    index_train_stop = int(len(list_files) *  PROPORTION_TRAIN) - 1
    index_test_start = index_train_stop + 1
    index_test_stop = len(list_files)
   
 
    for filename in list_files[index_train_start : index_train_stop]:
        try:
            os.makedirs(os.path.dirname(target_training_set_folder + '/' + filename))
            os.makedirs(os.path.dirname(source_location + '/' + filename))
        except OSError as exc:
            ()
                    
        with open(source_location + '/' + filename, 'r') as f:
            with open(target_training_set_folder + '/' + filename, 'w') as g:  
                g.write(f.read())
            
    for filename in list_files[index_test_start : index_test_stop]:
        try:
            os.makedirs(os.path.dirname(target_testing_set_folder + '/' + filename))
            os.makedirs(os.path.dirname(source_location + '/' + filename))
        except OSError as exc:
            ()
            
        with open(source_location + '/' + filename, 'r') as f:
            with open(target_testing_set_folder + '/' + filename, 'w') as g:
                g.write(f.read())
          
    files_written_for_train = index_train_stop - index_train_start
    files_written_for_test = index_test_stop - index_test_start  
    return (files_written_for_train, files_written_for_test)
            

if __name__ == "__main__":
    createDataset("../../../data/data_books_cleaned", "../../../data/data_books_training_set", "../../../data/data_books_testing_set")
