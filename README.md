# Implémentation d’un réseau de neurones pour la reconnaissance de chiffres manuscrits

La reconnaissance des chiffres manuscrits est un problème classique, pour l'aborder on se base souvent sur le machine learning. Notre projet consiste à la résolution du jeu de données MNIST. Il s’agit d’une base de données regroupant des images de chiffres manuscrits de petite taille (28x28) en noir et blanc.

Le but est d’entraîner un programme (dans notre cas un réseau de neurones) à classifier chaque image dans la bonne catégorie (un chiffre de 0 à 9). Ce réseau de neurones que nous allons implémenter de zéro en C, ce qui nécessitera de se familiariser avec la théorie dans un premier temps. Ensuite améliorer sa performance jusqu’à avoir de bons résultats.

Le projet se déroulera de la manière suivante:


### Pour le premier semestre:

  * Recherches et familiarisation sur l’aspect théorique (structure d’un réseau de neurones, différentiation automatique, descente de gradient, …).

  * Implémentation d’un système de différentation automatique, d’un premier réseau de neurones, et d’un algorithme de descente de gradient.

  * Test de l’ensemble sur le jeu de donnée MNIST.

### Pour le deuxieme semestre:

  * Mesures de performances de la version séquentielle.

  * Exploration, puis implémentation de pistes de parallélisation (parallélisation de mini-batches par exemple).

  * Mesures de performances et itérations sur la version parallèle.

  * Si possible, amélioration du taux de précision (en jouant sur des critères comme le choix de la fonction d’activation et le dimensionnement du réseau).


En résumé, l’objectif du premier semestre sera de développer une version minimale, et celui du deuxième d’améliorer le temps d’entraînement et la précision.



## Compilation

Pour lancer le programme `main.c,` il faut le compiler avec `Makefile` avec la commande:

      $ make
      
## Exécution

On doit lancer l'exécutable avec OpenMP :

  * Avec une seule couche (1 Hidden Layer) :
  
        $ OMP_NUM_THREADS=[NUM THREADS] ./exe [TrainSize] [HiddenNodes] [TestOffset] [TestSize]
       
  * Avec 2 couches (2 Hiddens Layer) :
  
        $ OMP_NUM_THREADS=[NUM THREADS] ./2Hidden [TrainSize] [HiddenNodes] [TestOffset] [TestSize]
  

