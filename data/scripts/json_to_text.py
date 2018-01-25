# -*- coding: utf-8 -*-

import json
import os

class JsonHandler():
    def __init__(self,SOURCE_LOCATION,TARGET_LOCATION):
        self.source = SOURCE_LOCATION #Where datafiles are.
        self.target = TARGET_LOCATION #Where processed datafiles will be.
        if SOURCE_LOCATION == TARGET_LOCATION:
            print("TARGET_LOCATION must differ from SOURCE_LOCATION")
    def convert(self,datafiles):
        list_files = os.listdir(SOURCE_LOCATION)
        for datafile in list_files:
    
            # Read the file.
            with open(self.source + "/" + datafile, "r") as f:
                file_content = f.read()
    
            # Split it so that every element of the array is a json string.
            file_content = file_content.split("\n")[:-1]
    
            # Process the json string and write the relevant info in the new file.
            output_path = self.target+ "/" + datafile.split(".")[0]+".txt"
            try:
                os.makedirs(os.path.dirname(output_path))
            except OSError as exc:
                ()
                
            f = open(output_path, "w")
            for line in file_content:
                data = json.loads(line)
                f.write(str(int(data["overall"])) + " " + data["reviewText"] + "\n")
            f.close()
    
            print("New file written at {}".format(output_path))


### EXAMPLE


#jsonhandler = JsonHandler("../Data/data_books_processed-3b836e62-d25c-4cdb-a417-6af240b0cf9b")
#jsonhandler.convert(["automotive.json"])
    