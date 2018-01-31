# Data

## Sommaire
- [1. Données brutes](#1-donn%C3%A9es-brutes)
- [2. Pré-traitement des données](#2-pr%C3%A9-traitement-des-donn%C3%A9es)
- [3. Données transformées](#3-donn%C3%A9es-transform%C3%A9es)
- [4. Données nettoyées](#4-donn%C3%A9es-nettoy%C3%A9es)
- [Installation de packages](#installation-de-packages)

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
NB: Ce script n'est plus utilisé par les autres programmes et peut sûrement être retiré. 

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


## 4. Données nettoyées

Après pré-traitement des données, il convient de transformer les commentaires de manière à pouvoir stocker de manière efficace les données tout en gardant les informations importantes. 

Pour ce faire, le script `cleaner.py` fournit une classe `text_cleaner` dont la méthode `cleaner` écrit un fichier texte contenant les données précédentes nettoyées dans le répertoire cible. Ce script fait appel à certains packages de Python qui imposent une installation préalable, voir les notes importantes à la fin de ce document. 

Plus précisément, on effectue les opérations suivantes : 

### 4.1 Traitement de la note 

La note est un entier entre 1 et 5 (compris). Afin de mieux pouvoir traiter le ton d'un commentaire et le sentiment qui lui est associé, le script `cleaner.py` transforme la note en un mot "Neutral", "Positive" ou "Negative". Pour le moment, l'association est faite de la manière suivante : 

- 1 ou 2 : "Negative"
- 3 ou 4 : "Neutral"
- 5 : "Positive"

Cette association peut être modifiée en modifiant le script `cleaner.py` de manière très simple. Elle peut même à terme passer en argument de la classe `text_cleaner`.

### 4.2 Traitement du commentaire 

Dans un premier temps, on crée des tokens avec le texte afin de pouvoir en faciliter le traitement.

A partir de ces tokens, on supprime : 
- les "stop words", comprendre les mots qui sont couramment utilisés dans la langue anglaise et ne porte pas beaucoup de sens, comme par exemple "it", "him", "are", etc...
- la ponctuation
- les nombres 

On effectue également un traitement sur les fautes d'orthographes à l'aide du package [Enchant](http://pythonhosted.org/pyenchant/tutorial.html) de Python. On vérifie la présence dans le dictionnaire anglais UK ou anglais US d'un mot avant de le conserver, et on prend la meilleur suggestion de mot disponible si le mot n'existe pas. 

Ce traitement prend environ 12 secondes par millier de reviews.

## Installations de packages

Le script `cleaner.py` utilise plusieurs packages importants à installer avant de pouvoir l'utiliser : 

### NLTK

NLTK, pour Natural Langage Tool Kit, permet d'assurer la tokenization du texte ainsi que de fournir la liste de "stop words".

Les informations pour installer le package sont disponibles sur le [site officiel de NLTK](http://www.nltk.org/install.html).
Après avoir installé nltk, lancer les commandes en console python : 

```python
import nltk
nltk.download("all")
```

Ces commandes devraient permettre l'installation des données de nltk correctement. A noter que `nltk.download()` lance une interface GUI qui semble bugger sur certains packages de nltk. 

### Enchant 

Enchant or PyEnchant nous permet de réaliser la correction des fautes d'orthographe au sein du texte. 

Les informations contenant l'installation du package sont contenues sur le [site officiel de PyEnchant](http://pythonhosted.org/pyenchant/tutorial.html), créé par Ryan Kelly. 

A noter que Enchant ne fonctionne que sur une version de Python 32 bits (une version de Python 32 bits peut cependant être installée sur une machine 64 bits sans problème)


