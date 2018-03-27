# -*- coding: utf-8 -*-

import os

PROPORTION_TRAIN = 0.9 # in ]0, 1[


def createDataset(source_location, target_training_set_folder, target_testing_set_folder):

    # Crée le dossier de destination s'il n'existe pas.
    if not os.path.exists(target_training_set_folder):
        os.makedirs(target_training_set_folder)
        
    if not os.path.exists(target_testing_set_folder):
        os.makedirs(target_testing_set_folder)
       

    all_lines = []
    for filename in os.listdir(source_location):             
        with open(source_location + '/' + filename, 'r') as f:
            all_lines += f.readlines()
    
    index_sep = int(len(all_lines)*PROPORTION_TRAIN)
            
    with open(target_training_set_folder + '/train' , 'w') as g:
        for line in all_lines[:index_sep]: 
            g.write(line)
        
    with open(target_testing_set_folder + '/test', 'w') as g:
        for line in all_lines[index_sep:]: 
            g.write(line)
    
    return (len(all_lines[:index_sep]), len(all_lines[index_sep:]))
            

if __name__ == "__main__":
    print(createDataset("../../../data/data_a", "../../../data/data_a_training_set", "../../../data/data_a_testing_set"))
