# Foobar

Foobar is a Python library for dealing with word pluralization.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install foobar
```

## Usage

```python
import foobar

# returns 'words'
foobar.pluralize('word')

# returns 'geese'
foobar.pluralize('goose')

# returns 'phenomenon'
foobar.singularize('phenomena')
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
## Dataset
[Lien vers le dossier Google drive contenant le dataset](https://drive.google.com/drive/folders/1DiroemRDE350kfqPi169KtrLlRWJbLRn?usp=drive_link)
# Description des données, méthodes, sources et objectif 
## Type de données utilisées
Le projet utilise des données historiques de matchs de tennis masculin de l’ATP, couvrant la période de 2000 à 2024. Chaque fichier annuel (atp_matches_YYYY.csv) contient les résultats de tous les matchs joués dans les tournois ATP cette année-là.
Ces fichiers incluent :

Des informations sur le tournoi (nom, surface, niveau, date),
Des informations sur les joueurs (nom, âge, taille, main dominante, classement),
Des statistiques de match (aces, double fautes, points de service, etc.),
Des informations sur le résultat du match (gagnant, perdant, score).

- Variables de base :
tourney_id : Identifiant des tournois. Type object.

tourney_name : Nom du tournoi. Type object.

surface : Surface du court (par exemple, "Hard", "Clay", etc.). Type object.

draw_size : Taille du tableau (nombre de joueurs dans le tournoi). Type int64.

Statistiques : La taille du tableau varie entre 2 et 128 joueurs, avec une moyenne de 55.34.

tourney_level : Niveau du tournoi (par exemple, "A" pour ATP, "B" pour Challenger, etc.). Type object.

tourney_date : Date du tournoi au format int64, avec des valeurs dans les années 2000 à 2020.

match_num : Numéro du match. Type int64.

Statistiques : Le numéro du match varie entre 1 et 1701 avec une moyenne de 102.27.

- Variables sur les joueurs :
winner_id : ID du gagnant. Type int64.

winner_seed : Classement (seed) du gagnant. Type float64.

Statistiques : Les valeurs varient entre 1 et 35, avec une moyenne de 7.40.

winner_entry : Statut d'entrée du gagnant (par exemple, "Q" pour qualifié). Type object.

winner_name : Nom du gagnant. Type object.

winner_hand : Main dominante du gagnant (par exemple, "R" pour droitier). Type object.

winner_ht : Taille du gagnant en cm. Type float64.

Statistiques : Les tailles vont de 3 cm (probablement une erreur) à 211 cm, avec une moyenne de 186.27.

winner_age : Âge du gagnant. Type float64.

Statistiques : L'âge des gagnants varie de 14.9 à 44.6 ans, avec une moyenne de 26.29.

- Variables sur le perdant :
loser_id : ID du perdant. Type int64.

loser_seed : Classement du perdant. Type float64.

Statistiques : Le classement varie entre 1 et 35, avec une moyenne de 8.92.

loser_entry : Statut d'entrée du perdant. Type object.

loser_name : Nom du perdant. Type object.

loser_hand : Main dominante du perdant. Type object.

loser_ht : Taille du perdant en cm. Type float64.

Statistiques : Les tailles vont de 3 cm à 211 cm, avec une moyenne de 185.67.

loser_age : Âge du perdant. Type float64.

Statistiques : L'âge des perdants varie de 14.9 à 44.6 ans, avec une moyenne de 26.29.

### 2. Statistiques sur les performances du match
w_ace, w_df, w_svpt, etc. : Ce sont des statistiques concernant les services du gagnant (aces, doubles fautes, points de service, etc.).

Par exemple, le nombre moyen de aces pour le gagnant est de 3.96 avec un maximum de 39.

l_ace, l_df, l_svpt, etc. : Ce sont des statistiques concernant les services du perdant.

Par exemple, le nombre moyen de aces pour le perdant est de 2.83 avec un maximum de 38.

### 3. Rankings des joueurs
winner_rank et loser_rank : Classement des joueurs avant le match.

Le classement des gagnants varie entre 1 et 2101, avec une moyenne de 79.64 pour les gagnants.

Le classement des perdants varie entre 1 et 2159, avec une moyenne de 117.70.

winner_rank_points et loser_rank_points : Points de classement des joueurs avant le match.

Les points de classement des gagnants varient entre 1 et 16950, avec une moyenne de 1608.44.

Les points de classement des perdants varient entre 1 et 16950, avec une moyenne de 977.41.

## Sources de données
Les données proviennent du dépôt GitHub maintenu par Jeff Sackmann, une référence dans les datasets open source du tennis professionnel :
➡️ https://github.com/JeffSackmann/tennis_atp

Les fichiers CSV sont récupérés automatiquement par scraping, puis concaténés en un seul DataFrame multi-années.

## Méthodes employées
1) Web scraping automatique des fichiers .csv via requests + os
2) Fusion et nettoyage des données via pandas
3) Analyse exploratoire (EDA) : stats descriptives sur les surfaces, performances par âge, classement, etc.
4) Prétraitement : sélection de variables pertinentes, gestion des valeurs manquantes, encodage
5) Création d’un jeu d’entraînement pour prédiction :
- Chaque ligne = un match
- Variables explicatives : âge, classement, surface, points ATP, ratio ace/double fautes
- Variable cible : victoire ou non d’un joueur

## Objectif du projet
L'objectif global est de prédire l’issue d’un match de tennis en fonction des caractéristiques des deux joueurs (âge, taille, classement, statistiques précédentes) et du contexte du match (surface, niveau du tournoi, etc.).
Le projet vise à explorer l’applicabilité de modèles de machine learning pour modéliser la victoire d’un joueur à partir de données historiques.

# Méthodes de machine learning sélectionnées et justifications
## Modèles testés
1) Random Forest Classifier
Avantages : robuste aux valeurs aberrantes, gère bien les variables catégorielles, bonne performance sans trop d’optimisation.
Raison du choix : permet d’identifier les variables les plus importantes via l’attribut feature_importances_, ce qui est utile pour interpréter les résultats.
Limite : risque d’overfitting observé lors de l’évaluation, notamment si l’hyperparamètre max_depth n’est pas bien contrôlé.
2) Logistic Regression 
Avantages : baseline interprétable, rapide à entraîner.
Raison : bonne référence pour comparer les performances des modèles plus complexes.
3) Gradient Boosting / XGBoost 
Avantages : souvent plus performant que Random Forest sur des datasets structurés.
Raison : gérer les corrélations entre variables et ajuster les erreurs précédentes.

## Évaluation des performances
- accuracy_score : prédiction correcte du gagnant
- confusion_matrix : savoir quand le modèle se trompe
- feature_importances_ : pour savoir quelles variables expliquent le plus la victoire (ex. classement, surface, âge, service)
