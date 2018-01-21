# Data

## 1. Données brutes
On peut télécharger les données des reviews Amazon à l'URL suivante : http://jmcauley.ucsd.edu/data/amazon/. On conseillera l'utilisation des "petits" subsets, qui sont déjà catégorisés (prendre la version 5-core).

### 1.1 Informations sur les fichiers du site
- Le "petit" subset des **books** 5-core (8,898,041 reviews) pèse 3.00Go et est compressé 2.9 fois (le fichier décompressé pèse donc 8.80Go)
- Par interpolation, on peut en calculer le poids moyen compressé d'**une** review : 362.29 octets et son poids décompressé : 1062.93 octets. (peut être utile pour prévoir l'espace disque nécessaire pour le stockage d'un gros fichier à télécharger et à décompresser)

### 1.2 Informations sur le format
Le fichier décompressé contient un certain nombre de "sample review" (1 sample review = 1 array json) séparés par des retours à la ligne.

Exemple de Sample review : 
```javascript
{"reviewerID": "A2SUAM1J3GNN3B","asin": "0000013714",  "reviewerName": "J. McDonald",  "helpful": [2, 3],  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",  "overall": 5.0,  "summary": "Heavenly Highway Hymns",  "unixReviewTime": 1252800000,  "reviewTime": "09 13, 2009"}
```


## 2. Pré-traitement des données
On peut alors utiliser le script `json_to_text.py` pour les transformer en fichiers texte ne contenant que les informations nécessaires. (lire les commentaires du script avant de l'exécuter).

Le script `data_loader.py` fournit une classe `DataLoader` dont la méthode `load` renvoie un tableau contenant les données contenues dans un fichier. (lire les commentaires du script pour avoir des exemples d'utilisation)  

## 3. Données transformées

Les données retournées par le script sont consignées dans des fichiers texte de la manière suivante : 

* Chaque ligne correspond à une review.
* Le premier caractère de la ligne est la note associée (entier entre 1 et 5),
* Le second caractère de la ligne est un espace,
* Les caractères suivants correspondent au commentaire.

Exemple d'une ligne :
```
5 Not much to write about here, but it does exactly what it's supposed to. filters out the pop sounds. now my recordings are much more crisp. it is one of the lowest prices pop filters on amazon so might as well buy it, they honestly work the same despite their pricing,
```

Voir aussi le [fichier exemple](./instruments.txt) dans ce dossier.

