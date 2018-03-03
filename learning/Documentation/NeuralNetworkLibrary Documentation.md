# NeuralNetworkLibrary

Ce script permet de générer et d'entraîner un réseau de neurones. La classe principale est la classe Network,
les autres classes ne sont pas sensées être manipulées par l'utilisateur

## Méthodes 

### Le constructeur :

* __arguments :__ 
  * __dimensions :__ Une liste d'entiers contenant les dimensions de chaque couche du réseau (la couche d'input comprise), 
  par exemple pour générer un réseau prenant en entrée un tableau de taille 7, ayant une couche intermédiaire de 10 
  neurones et une couche de sortie de 2 neurones, dimensions = [7,10,2]
  
* __output :__
  * Une instance de la classe Network
  
### train : 

* __arguments :__ 
  * __input :__ un tableau de réels dont la taille doit correspondre à la taille de la couche d'input du réseau
  * __expected_output :__ un tableau de réels contenant la sortie que l'on attend du réseau
  * __alpha :__ La vitesse à laquelle les poids du réseau sont modifiés lors de l'apprentissage, doit être petit par rapport à 1 
  
* __output :__
  * __None__, cette méthode modifie les poids du réseau par backpropagation mais ne retourne rien.
  
 Note : La classe Network a un attribut LAMBDA qui correspond à la pente de la sigmoïde utilisée pour calculer les sorties des neurones
 , ALPHA est set à 1 par défaut, mais peut être changé manuellement (même entre deux apprentissages mais du coup ça gâche un peu l'apprentissage déjà effectué)
