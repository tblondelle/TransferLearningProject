import json_to_text as jtt
import data_loader as dl
import operator
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

#Define paths and parameters
negative = [1,2]
neutral = [3,4]
positive = [5]
filetype="video"
datapath = "D:/ECL/S9/Projet info/TransferLearningProject/data/" #Input the path to the "data directory here"
raw = "data_raw" #Input the name of the source directory here 
txt = "data_txt" #Input the name of the target directory here 
clean = "data_cleaned" #Input the name of the target directory after cleaning


#load data
Json = jtt.JsonHandler(datapath+raw,datapath+txt)
Json.convert()
Loader = dl.DataLoader(datapath+txt)
data = Loader.load(filetype)
# Use only a small portion of the data

# set the stopwords and punctuation, in a list so you can adjust
stop_words=list(set(stopwords.words('english')))
punctuation = list(set(string.punctuation))
punctuation = punctuation + ["''"] + ["``"] #punctuation adjustment 

#Open file
f = open(datapath+clean+"/"+filetype+".txt", "w")

for i in range(len(data)):
    # Define sentiment associated with review based on mark
    sentiment = ""
    if data[i][0] in negative:
        sentiment = "Negative"
    elif data[i][0] in neutral:
        sentiment = "Neutral"
    else:
        sentiment = "Positive"
    #Tokenize, clean of stop words and punctuation
    current_text = (data[i][1]).lower()
    tokens = word_tokenize(current_text)
    filtered = [w for w in tokens if w not in stop_words]
    filtered = [w for w in filtered if w not in punctuation]
    #prepare string 
    filtered_review = ""
    for e in filtered:
        filtered_review += e + " " 
    f.write(sentiment + " " + filtered_review + "\n")

print("New file written at {}".format(datapath+clean))
    
    

#def occurences(tokens):
#        occs={}
#        for element in tokens:
#            for word in element:
#                w = word.lower()
#                if w in occs:
#                    occs[w] += 1
#                else:
#                    occs[w] = 1
#        # Sort by number of occurences in dictionnary 
#        occs = sorted(occs.items(),key=operator.itemgetter(1))
#        return(occs)
#        
#
#def Mostoccs(occs,k):
#    words = [occs[k][0] for k in range(len(occs))] 
#    return(words[(len(words)-1-k):(len(words)-1)])
        