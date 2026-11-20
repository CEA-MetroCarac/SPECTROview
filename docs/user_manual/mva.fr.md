## **Analyse multivariée (MVA)**

**SPECTROview** intègre des outils d'Analyse Multivariée (MVA), facilement accessibles depuis l'onglet dédié **MVA** disponible aussi bien dans l'espace `Spectra` que dans l'espace `Maps`.

> **Remarque** : Les algorithmes MVA nécessitent au minimum 2 spectres actifs pour s'exécuter.

### **1. Algorithmes pris en charge**

#### **1.1. Analyse en Composantes Principales — ACP (`PCA`)**

Procédure statistique qui transforme orthogonalement et réduit la dimensionnalité de jeux de données spectraux complexes, révélant les principaux axes de variance. L'implémentation repose sur la décomposition en valeurs singulières (SVD).

  - **Paramètres** :
    - *Components* (2–50) : Nombre de composantes principales à conserver.
    - *Mean centering* (activé/désactivé) : Indique s'il faut soustraire le spectre moyen avant l'analyse. Activé par défaut.
  - **Visualisations disponibles** : Tableau de bord récapitulatif (`Summary`), diagramme d'éboulis (`Scree Plot` — variance expliquée), poids spectraux des composantes principales (`Loadings`), carte thermique spatiale 2D d'une composante (`Heatmap`), nuages de points des scores (`Scores`) et erreur de reconstruction résiduelle (`Residuals`).

#### **1.2. Factorisation par Matrices Non Négatives (`NMF`)**

Algorithme de décomposition puissant qui factorise les données spectrales en composantes additives distinctes et strictement positives, correspondant souvent à des espèces chimiques pures (*endmembers*).

  - **Paramètres** :
    - *Components* (2–50) : Nombre de composantes non négatives à extraire.
    - *Max Iterations* (50–5000) : Nombre maximal d'itérations de mise à jour.
    - *Tolerance* : Seuil de convergence pour l'arrêt anticipé.
    - *Random Seed* : Graine aléatoire garantissant des résultats reproductibles.
  - **Visualisations disponibles** : Tableau de bord récapitulatif (`Summary`), poids spectraux des composantes (`Loadings`), carte thermique spatiale 2D (`Heatmap`), nuages de points des scores (`Scores`) et erreur de reconstruction résiduelle (`Residuals`).

> **Pourquoi la NMF n'a-t-elle pas de Scree Plot ?**  
> Contrairement à l'ACP (`PCA`), qui ordonne strictement des composantes orthogonales selon le maximum de variance qu'elles expliquent, la NMF recherche des composantes additives non orthogonales (*endmembers*) qui se combinent entre elles. Comme les composantes NMF ne sont ni orthogonales ni ordonnées séquentiellement par variance, il est impossible d'attribuer un pourcentage isolé de « variance expliquée » à chaque composante individuelle. Pour évaluer un modèle NMF, appuyez-vous sur le graphique **Residuals** afin de juger de la qualité de reconstruction et sur **Loadings** pour vérifier que les composantes correspondent à des spectres physiquement cohérents.

------

### **2. Interface utilisateur**

L'onglet MVA repose sur une **disposition à deux panneaux** :

#### **2.1. Panneau gauche — Contrôles**

- **Sélection de la méthode** : Choisissez entre `PCA` et `NMF` à l'aide des boutons radio.
- **Contrôles des paramètres** : Tous les paramètres réglables sont accessibles sous forme d'éléments graphiques interactifs (sélecteurs numériques, cases à cocher). Seuls les paramètres propres à la méthode sélectionnée sont affichés.
- **Bouton d'exécution** : Cliquez sur « ▶ Run Analysis » pour lancer l'analyse avec les paramètres courants.
- **Export** : Envoyez les résultats (scores) vers l'espace `Graphs` pour une analyse plus poussée.

#### **2.2. Panneau droit — Onglets de graphiques**

Les résultats sont présentés dans plusieurs onglets, chacun disposant de sa propre barre d'outils `matplotlib` interactive (zoom, déplacement, sauvegarde) :

| Onglet | PCA | NMF | Description |
|--------|-----|-----|-------------|
| **Summary** | ✅ | ✅ | Tableau de bord combiné affichant simultanément Scree, Loadings et Scores |
| **Scree Plot** | ✅ | - | Histogramme de la variance expliquée par composante avec courbe cumulée |
| **Loadings** | ✅ | ✅ | Profils spectraux des composantes superposés sur l'axe des nombres d'onde |
| **Heatmap** | ✅ | ✅ | Carte spatiale 2D des scores pour une composante choisie *(lorsque des coordonnées spatiales `(x, y)` figurent dans les noms de spectres)* |
| **Scores** | ✅ | ✅ | Nuage de points 2D avec choix des composantes pour les axes X et Y |
| **Residuals** | ✅ | ✅ | Erreur de reconstruction par spectre pour évaluer la qualité du modèle |

### **3. Conseils**

- Utilisez le **Scree Plot** pour déterminer le nombre optimal de composantes (repérez le « coude » sur la courbe de variance cumulée).
- Le graphique **Loadings** révèle quelles signatures spectrales contribuent le plus à chaque composante.
- L'onglet **Heatmap** visualise les zones spatiales où le score d'une composante donnée se concentre selon les coordonnées `(x, y)` de la carte.
- Le graphique **Scores** met en évidence les regroupements et les spectres atypiques de votre jeu de données.
- L'onglet **Residuals** permet d'identifier les spectres mal reconstruits par le modèle.
- Chaque onglet graphique intègre une **barre d'outils `matplotlib`** pour zoomer, se déplacer et enregistrer les figures.


-----

### **4. Citations et références**

Si vous utilisez ces fonctionnalités MVA pour obtenir des résultats publiés, nous vous remercions de bien vouloir citer les articles de référence décrivant les algorithmes sous-jacents :

- **Algorithme PCA** : Wold, S., Esbensen, K., and Geladi, P., "Principal component analysis." *Chemometrics and intelligent laboratory systems* 2.1-3 (1987): 37-52.
- **Algorithme NMF** : Lee, Daniel D., and H. Sebastian Seung, "Learning the parts of objects by non-negative matrix factorization." *Nature* 401.6755 (1999): 788-791.
