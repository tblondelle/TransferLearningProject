import os

# Modifier le dir si nécessaire
os.chdir("C:/Users/Antoine/Documents/Centrale/3A/Transfer_learning")

#Remplacer par les noms des json à traiter (sans le .json)
JSONS = ["apps","automotive","babies","beauty","cell_phones","food","garden","health","instruments","music","office_products","pet_supplies","tools","toys","video_games","videos"]

for JSON in JSONS:
    print(JSON)
    JSON += ".json"
    json_doc = open(JSON,"r")
    try:
        os.remove(JSON.split(".")[0]+".txt","w")
    except:
        ()
    text_doc = open(JSON.split(".")[0]+".txt","w")
    for line in json_doc:
        parts = line.split('"')
        cleaned_parts = []
        to_concatenate = 0
        for i in range(len(parts)):
            if to_concatenate:
                to_concatenate = 0
                cleaned_parts[-1] += parts[i][:-1]
            else:
                cleaned_parts.append(parts[i])
            if parts[i]:
                if parts[i][-1] == "\\":
                    to_concatenate = 1
        valid = 1
        try:
            note = int(cleaned_parts[20][2])
        except:
            valid = 0
        text_review = cleaned_parts[17]
        if valid:
            text_doc.write(str(note)+" "+text_review +'\n')
        
    

