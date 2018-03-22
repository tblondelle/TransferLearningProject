# -*- coding: utf-8 -*-

import os

PROPORTION_TRAIN = 0.9 # in ]0, 1[


def createDataset(source_location, target_training_set_folder, target_testing_set_folder):

    list_files = os.listdir(source_location)
    
    all_lines = []
    
    for filename in list_files:
        
        # Crée le dossier de destination s'il n'existe pas.
        try:
            os.makedirs(os.path.dirname(target_training_set_folder + '/' + filename))
            os.makedirs(os.path.dirname(source_location + '/' + filename))
        except OSError as exc:
            ()
                    
        with open(source_location + '/' + filename, 'r') as f:
            with open(target_training_set_folder + '/' + filename, 'w') as g:  
                all_lines.append(f.read())
    

    index_sep = int(len(all_lines)*PROPORTION_TRAIN)
    
    with open(target_training_set_folder + '/train' , 'w') as g:
        g.write(all_lines[:index_sep])
        
    with open(target_testing_set_folder + '/test', 'w') as g:
        g.write(all_lines[index_sep:])
    
    
    return (len(all_lines[:index_sep]), len(all_lines[index_sep:]))
            

if __name__ == "__main__":
    createDataset("../../../data/data_tools_cleaned", "../../../data/data_tools_training_set", "../../../data/data_tools_testing_set")
