# -*- coding: utf-8 -*-


import string
from nltk.corpus import stopwords
#from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import os


#Define sentiment based on rating
negative = [1,2,3]
neutral = []
positive = [4,5]

#stemmer = SnowballStemmer("english", ignore_stopwords=True)
lmtzr = WordNetLemmatizer()
# set the stopwords and punctuation, in a list so you can adjust
stop_words=list(set(stopwords.words('english')))
punctuation = list(set(string.punctuation))
# Adjusts sets of stop words and punctuation
punctuation = punctuation + ["''"] + ["``"] #punctuation adjustment 
stop_words = stop_words

stop_words2 = []

for e in stop_words:
    try:
        if e[-3:] == "n't" or e[-1] =="n":
            pass
        else:
            stop_words2 += [e]
    except:
        print("word isn't long enough")

stop_words = stop_words2

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
        count = 0
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
                print(exc)
                pass
            f=open(output_path,"w",encoding="utf-8")
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
                #remove punctuation
                for e in current_text:
                    if e in punctuation:
                        current_text = current_text.replace(e,"")
                #tokens = word_tokenize(current_text)
                tokens = current_text.split(" ")
                #remove stop words
                filtered = [w for w in tokens if w not in stop_words]
                #remove numbers
                filtered = [w for w in filtered if not is_number(w)]
                #prepare string 
                filtered_review = ""
                for e in filtered:
                    filtered_review += lmtzr.lemmatize(e) + " "
                f.write(sentiment + separator + filtered_review + "\n")
                count += 1
                if count % 1000 == 0:
                    print(count,"lignes écrites")
            
            print("New file written at {}".format(output_path))
            
    
