# -*- coding: utf-8 -*-

import json
import uuid
import os

SOURCE_LOCATION = "../Data/data_books" # Where datafiles are.
TARGET_LOCATION = "../Data/data_books_processed" # Where processed datafiles will be.
datafiles = ["books_ac.txt", "books_ad.txt", "books_ae.txt"] # List of files (one json per line).

id_target_folder = str(uuid.uuid4())
if not os.path.exists(TARGET_LOCATION + "-" + id_target_folder):
    os.makedirs(TARGET_LOCATION + "-" + id_target_folder)

for datafile in datafiles:
    
    # Read the file.
    with open(SOURCE_LOCATION + "/" + datafile, "r") as f:
        file_content = f.read()
    
    # Split it so that every element of the array is a json string.
    file_content = file_content.split("\n")[:-1]
    
    # Process the json string and write the relevant info in the new file.
    for line in file_content:
        data = json.loads(line)
        
        with open(TARGET_LOCATION + "-" + id_target_folder + "/" + datafile, "a") as f:
            #print( "{}\t{}...".format(data["overall"], data["reviewText"][:30]))
            f.write(str(data["overall"]) + "\t" + data["reviewText"] + "\n")
    
    print("New file written at {}/{}".format(TARGET_LOCATION + "-" + id_target_folder, datafile))
    