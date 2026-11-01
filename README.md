Prédiction du départ des clients (Customer Churn)

Projet de machine learning : prédire si un client d'un opérateur télécom va quitter l'entreprise, à partir de ses caractéristiques et de son utilisation du service.

Notebook : churn_projet.ipynb (à ouvrir avec Google Colab)

1. Problème

Le churn est le départ d'un client. Garder un client coûte moins cher qu'en trouver un nouveau : si l'entreprise sait à l'avance quels clients risquent de partir, elle peut agir (offre, appel, réduction).

C'est une classification binaire : la cible Churn vaut Yes (parti) ou No (resté).

2. Données
Jeu de données : Telco Customer Churn (source : Kaggle, licence : à compléter après vérification sur la page du dataset).
7043 clients, 21 colonnes (profil, services souscrits, contrat et facturation).
Classes déséquilibrées : 73,5 % restés, 26,5 % partis.
Une accuracy de 73,5 % s'obtient en répondant toujours « resté » : c'est la baseline à dépasser.
3. Préparation des données
Colonne TotalCharges convertie en nombres ; 11 valeurs vides (clients avec tenure = 0, sans paiement cumulé) remplacées par 0.
Colonne customerID supprimée (identifiant unique, sans valeur prédictive).
Modalités « No internet service » et « No phone service » fusionnées avec « No ».
Cible encodée : Yes = 1, No = 0.
Variables catégorielles encodées en one-hot (drop_first=True) : 23 colonnes.
Séparation 80 % entraînement / 20 % test (random_state=42, stratify=y).
Mise à l'échelle (StandardScaler) calculée sur l'entraînement uniquement, pour éviter toute fuite de données.
4. Algorithmes

Régression logistique : calcule un score par somme pondérée des caractéristiques, le convertit en probabilité de départ avec la fonction sigmoïde, puis décide avec un seuil (0,5 par défaut). Les poids sont appris en minimisant la perte logarithmique.

Random Forest : 100 arbres de décision, chacun construit sur un échantillon aléatoire de clients et de caractéristiques ; la prédiction est le vote de l'ensemble des arbres.

5. Résultats (jeu de test, classe « Parti »)
Modèle	Accuracy	Précision	Rappel	F1
Baseline (toujours « resté »)	0,735	-	0	-
Régression logistique (seuil 0,5)	0,81	0,66	0,56	0,61
Random Forest	0,79	0,64	0,49	0,56
Régression logistique, seuil 0,3	-	0,52	0,75	0,62
Régression logistique, class_weight="balanced"	0,74	0,51	0,78	0,61

Modèle retenu : la régression logistique. Elle est meilleure que le Random Forest sur toutes les mesures, rate moins de départs (163 contre 189) et reste facile à interpréter.

Facteurs associés au départ (poids du modèle) : une ancienneté élevée (-1,25) et un contrat de 2 ans (-0,59) sont associés à moins de départs ; la fibre optique (+0,74) est associée à plus de départs. Ce sont des associations, pas des causes.

6. Pourquoi l'accuracy ne suffit pas

Avec 73,5 % de clients restés, un modèle peut avoir une accuracy correcte tout en ratant beaucoup de départs. Pour l'entreprise, rater un départ (faux négatif) coûte plus cher qu'une fausse alerte (faux positif) : le rappel de la classe « Parti » est donc la mesure principale.

7. Améliorations testées
Baisser le seuil à 0,3 : le rappel passe de 0,56 à 0,75, la précision de 0,66 à 0,52.
class_weight="balanced" : le rappel passe à 0,78, la précision à 0,51, l'accuracy baisse à 0,74.
Les deux méthodes ont un effet très proche : elles rendent le modèle plus prêt à prédire « parti », au prix de plus de fausses alertes.
8. Analyse des erreurs

Les départs manqués (163) ont un risque moyen de 28 %. Ils sont plus anciens (24,7 mois en moyenne contre 10,1) et paient moins (64,9 contre 78,8) que les départs bien repérés : ils ne ressemblent pas au profil « classique » du départ.

9. Limites
Le jeu de données ne contient pas les raisons du départ (plainte, concurrence, déménagement).
Le modèle repère encore au mieux 56 % des départs avec le seuil par défaut.
Le seuil a été comparé sur le jeu de test ; pour être rigoureux, il faudrait un jeu de validation séparé.
Un seul découpage train/test a été utilisé : une validation croisée donnerait une estimation plus fiable.
Le choix du seuil dépend du coût réel d'une fausse alerte par rapport à un départ manqué, qui n'est pas connu ici.
Le modèle est entraîné sur un seul opérateur : il ne se généralise pas à une autre entreprise.
10. Reproduire le projet
Télécharger le dataset (voir section 2) et le placer dans le dossier data/.
Ouvrir churn_projet.ipynb dans Google Colab.
Exécuter toutes les cellules (Exécution → Tout exécuter).

Outils : Python, pandas, NumPy, Matplotlib, scikit-learn, Google Colab, GitHub.
