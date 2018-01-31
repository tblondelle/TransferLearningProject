# -*- coding: utf-8 -*-


import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import enchant
import os

#define spell check dictionnaries
DUS = enchant.Dict("en_US")
DUK = enchant.Dict("en_UK")

#Define sentiment based on rating
negative = [1,2]
neutral = [3,4]
positive = [5]

# set the stopwords and punctuation, in a list so you can adjust
stop_words=list(set(stopwords.words('english')))
punctuation = list(set(string.punctuation))
# Adjusts sets of stop words and punctuation
punctuation = punctuation + ["''"] + ["``"] #punctuation adjustment 
stop_words = stop_words + ["n't"] #stopwords adjustment, tokenize separate didn't in did and n't

# Define simple function to check if a string is a number 
def is_number(s):
    try:
        int(s)
        return(True)
    except ValueError:
        return(False)



class TextCleaner():
    def __init__(self, SOURCE_LOCATION, TARGET_LOCATION):
        self.source = SOURCE_LOCATION #Where datafiles are.
        self.target = TARGET_LOCATION #Where processed datafiles will be.
        if SOURCE_LOCATION == TARGET_LOCATION:
            print("TARGET_LOCATION must differ from SOURCE_LOCATION")
            
    def clean(self, separator="\t"):
        list_files = os.listdir(self.source)
        for datafile in list_files:
            data = []
            input_path = self.source + "/" + datafile
            r = open(input_path,"r")
            for line in r:
                data.append([int(line[0]),line[2:-1]])
            output_path = self.target + "/" + datafile.split(".")[0]+".txt"
            try:
                os.makedirs(os.path.dirname(output_path))
            except OSError as exc:
                ()
            f=open(output_path,"w")
            for i in range(len(data)):
                # Define sentiment associated with review based on mark
                sentiment = ""
                if data[i][0] in negative:
                    sentiment = "Negative"
                elif data[i][0] in neutral:
                    sentiment = "Neutral"
                else:
                    sentiment = "Positive"
                #Tokenize, clean of stop words and simple punctuation
                current_text = (data[i][1]).lower()
                tokens = word_tokenize(current_text)
                #remove stop words
                filtered = [w for w in tokens if w not in stop_words]
                #remove punctuation
                filtered = [w for w in filtered if w not in punctuation]
                #remove numbers
                filtered = [w for w in filtered if not is_number(w)]
                #prepare string 
                filtered_review = ""
                for e in filtered:
                    #check if the words exists in the dictionnary
                    if DUS.check(e) or DUK.check(e):
                        filtered_review += e + " " 
                    else:
                    #correct the word with the dictionnary suggestion
                        corrected = DUK.suggest(e)
                        #if there isn't a correction drop the word
                        if len(corrected)==0:
                            pass
                        else:
                            filtered_review += corrected[0] + " "
                f.write(sentiment + separator + filtered_review + "\n")
            
            print("New file written at {}".format(output_path))
            
    
