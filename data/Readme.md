# DATA

Les données sont consignées dans des fichiers texte de la manière suivante : 

* Chaque ligne correspond à une review.
* Le premier caractère de la ligne est la note associée et le reste correspond au commentaire.

On peut télécharger les données au format json à l'url suivante : http://jmcauley.ucsd.edu/data/amazon/
(Il faut télécharger les versions 5-core pour chacune des catégories).
On peut alors utiliser le script `json_to_text.py` pour les transformer en fichiers texte ne contenant que les informations nécessaires. (lire les commentaires du script avant de l'exécuter)

Le script `data_loader.py` fournit une classe `DataLoader` dont la méthode load renvoie un tableau contenant les données contenues dans un fichier. (lire les commentaires du script pour avoir des exemples d'utlilisation)  
