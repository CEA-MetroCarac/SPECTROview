# **Analyse multivariée (MVA) de données spectroscopiques Raman**

Ce document présente une vue technique des fonctionnalités d'Analyse Multivariée (MVA) implémentées dans `SPECTROview`.

## **Architecture**

La fonctionnalité MVA suit le patron architectural MVVM (Model-View-ViewModel) :

- **Model** (`spectroview/model/analysis/m_mva.py`) : Contient les mathématiques fondamentales des techniques MVA, à savoir l'Analyse en Composantes Principales (PCA) et la Factorisation par Matrices Non Négatives (NMF). Utilise exclusivement les bibliothèques numériques standard (`numpy`, `scipy`) sans dépendance supplémentaire telle que `scikit-learn`.
- **ViewModel** (`spectroview/viewmodel/analysis/vm_mva.py`) : Orchestre le flux de données. Extrait les données prétraitées des spectres actifs, appelle le Model pour calculer les décompositions, met les résultats en cache et calcule les erreurs de reconstruction par spectre.
- **View** (`spectroview/view/components/lower_tabs/` : `v_mva.py`, `v_mva_plots.py`) : Fournit l'interface utilisateur avec panneau d'onglets graphiques (`Summary`, `Scree Plot`, `Loadings`, `Heatmap`, `Scores`, `Residuals`), contrôles de paramètres interactifs et barres d'outils `matplotlib` sur chaque tracé.

## **Implémentation algorithmique**

L'implémentation de l'ACP (PCA) dans `SPECTROview` repose sur la décomposition en valeurs singulières (SVD) de données centrées sur la moyenne.

### **Enrichissements de visualisation**

`SPECTROview` propose une interface complète à onglets graphiques pour explorer les résultats :

- **Summary** : Tableau de bord 2×2 affichant Scree Plot (PCA uniquement), les 3 premiers Loadings et les projections principales de Scores
- **Scree Plot** (PCA uniquement) : Histogramme de variance par composante + courbe cumulée
- **Loadings** : Superposition des profils spectraux avec remplissage sous courbe pour plus de lisibilité
- **Heatmap** : Carte spatiale 2D des scores de la composante choisie lorsque les noms de spectres portent des coordonnées `(x, y)`
- **Scores** : Nuage de points 2D avec sélection interactive des axes
- **Residuals** : Erreur de reconstruction par spectre pour évaluer la qualité du modèle

## **Méthodes implémentées**

### **1. Analyse en Composantes Principales (PCA)**

L'ACP est une méthode de réduction de dimensionnalité transformant les spectres dans un nouveau repère orthogonal de sorte que la plus grande variance de projection se situe sur le premier axe (première composante principale).

- **Détails d'implémentation** : Basée sur la décomposition en valeurs singulières (`numpy.linalg.svd`) sur données optionnellement centrées sur la moyenne.
- **Paramètres** :
  - *Components* : Nombre de composantes principales à retenir (2–50).
  - *Mean centering* : Indique s'il faut soustraire le spectre moyen avant la SVD (activé par défaut, pratique standard en ACP spectrale).
- **Visualisations** :
  - *Summary* : Tableau de bord synthétique 2×2 combinant Scree, Loadings et Scores.
  - *Scree Plot* : Pourcentage de variance expliquée par chaque composante principale et variance cumulée.
  - *Loadings* : Contribution spectrale de chaque composante avec le pourcentage de variance expliquée en légende.
  - *Heatmap* : Carte spatiale 2D des scores pour une composante choisie lorsque `(x, y)` figurent dans les noms de spectres.
  - *Scores* : Nuage 2D des spectres projetés sur les composantes principales avec choix interactif des axes.
  - *Residuals* : Erreur de reconstruction L2 par spectre pour juger de la fidélité du modèle.

### **2. Factorisation par Matrices Non Négatives (NMF)**

La NMF factorise la matrice de données en deux matrices à termes positifs ou nuls (classiquement W et H), particulièrement adaptée à la spectroscopie car elle produit des composantes additives positives correspondant souvent aux spectres purs des espèces présentes (*endmembers*).

- **Détails d'implémentation** : Règles de mise à jour multiplicatives de Lee & Seung, minimisant itérativement la norme de Frobenius de l'erreur de reconstruction.
- **Paramètres** :
  - *Components* : Nombre de composantes positives à extraire.
  - *Max Iterations* : Nombre maximal d'itérations de mise à jour multiplicatives avant arrêt.
  - *Tolerance* : Seuil de convergence déclenchant l'arrêt anticipé si la variation relative d'erreur de reconstruction passe en dessous.
  - *Random Seed* : Graine du générateur pseudo-aléatoire initialisant les matrices afin d'assurer la reproductibilité (42 par défaut).
- **Visualisations** :
  - *Summary* : Vue d'ensemble des Loadings et projections principales NMF.
  - *Loadings* : Composantes spectrales non négatives (*endmembers*).
  - *Heatmap* : Carte d'abondance spatiale 2D pour une composante NMF choisie lorsque les noms portent `(x, y)`.
  - *Scores* : Nuage 2D des abondances/concentrations des composantes.
  - *Residuals* : Erreur de reconstruction L2 par spectre.

> **Pourquoi la NMF n'a-t-elle pas de Scree Plot ?**  
> Contrairement à la PCA, qui ordonne strictement des composantes orthogonales par variance maximale expliquée, la NMF identifie des composantes additives non orthogonales (*endmembers*) qui se combinent. Comme les composantes NMF ne sont ni orthogonales ni ordonnées séquentiellement par variance, aucun pourcentage isolé de « variance expliquée » ne peut être attribué individuellement à chaque composante. Pour évaluer un modèle NMF, appuyez-vous sur **Residuals** pour la qualité de reconstruction et sur **Loadings** pour valider le sens physique des spectres extraits.

## **Interface utilisateur**

La vue MVA adopte une disposition à deux panneaux :

### **Panneau gauche (Contrôles)**

- **Sélection de méthode** : Boutons radio basculant entre `PCA` et `NMF`.
- **Groupes de paramètres** : Boîtes distinctes pour PCA et NMF n'affichant que les réglages pertinents.
- **Bouton Run** : Exécute l'analyse.
- **Section Export** : Transfère les scores vers l'espace `Graphs`.

### **Panneau droit (Onglets de tracés)**

Chaque tracé occupe son propre onglet dans un `QTabWidget` :

- **Summary** : Grille combinée (Scree Plot [PCA], 3 premiers Loadings et projections de Scores).
- **Scree Plot** (PCA uniquement) : Variance expliquée par composante.
- **Loadings** : Profils spectraux des composantes.
- **Heatmap** : Carte spatiale 2D des scores (nécessite `(x, y)` dans les noms de spectres).
- **Scores** : Nuage 2D interactif avec choix d'axes.
- **Residuals** : Qualité de reconstruction par spectre.

Chaque onglet inclut une `NavigationToolbar` `matplotlib` pour zoomer, déplacer et enregistrer.

## **Pipeline de données**

1. **Sélection** : La MVA opère exclusivement sur les spectres *actifs* (cochés) dans l'espace `SPECTROview`.
2. **Prétraitement** : `VMMVA` lit directement `SpectraStore` (`md.Y` si prétraité, sinon `md.Y0`), exploitant ainsi les données ayant déjà reçu les prétraitements utilisateur (`baseline`, recadrage, correction X).
3. **Interpolation** : Si les spectres actifs possèdent des axes de nombres d'onde distincts, ils sont automatiquement interpolés sur une grille commune (celle du premier spectre actif) pour former une matrice homogène.
4. **Export** : Les résultats (`scores` pour PCA, matrice `W` pour NMF) peuvent être exportés directement vers l'espace `Graphs` pour comparaison et visualisation avancée.

## **Citations et références**

Si ces fonctionnalités contribuent à une publication, merci de citer les articles décrivant les algorithmes implémentés :

[1] **Algorithme PCA** : Wold, S., Esbensen, K., and Geladi, P., "Principal component analysis." *Chemometrics and intelligent laboratory systems* 2.1-3 (1987): 37-52.

[2] **Algorithme NMF** : Lee, Daniel D., and H. Sebastian Seung, "Learning the parts of objects by non-negative matrix factorization." *Nature* 401.6755 (1999): 788-791.
