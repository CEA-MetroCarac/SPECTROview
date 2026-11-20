## **Formats de données pris en charge**

Des exemples de tous les types de données pris en charge sont disponibles dans le dossier [`/examples`](https://github.com/CEA-MetroCarac/SPECTROview/tree/main/examples) du dépôt GitHub. Vous pouvez télécharger ces fichiers pour vous familiariser avec les formats acceptés, la structure des données et prendre en main **SPECTROview** sur des jeux de données réels.

| Format | Extension | Chargeur | Type de données |
|---|---|---|---|
| Texte délimité | `.txt`, `.csv` | Détection automatique du séparateur et de l'en-tête | Spectres individuels, cartes 2D, cartographies de plaquettes (*wafer*) et cartes résolues en temps |
| Renishaw WiRE | `.wdf` | Lecteur binaire intégré | Spectres individuels, séries temporelles, profils linéaires et cartes 2D |
| HORIBA LabSpec 6 | `.l6s`, `.l6m` | Lecteur binaire intégré | Spectres individuels (`.l6s`) et cartes hyperspectrales 2D (`.l6m`) |
| HORIBA LabSpec 5 | `.ngs`, `.ngc` | Lecteur binaire intégré | Spectres individuels (`.ngs`) et cartes hyperspectrales 2D compressées (`.ngc`) |
| SPC | `.spc` | Lecteur binaire intégré | Spectres individuels et cartes 2D |
| TRPL | `.dat` | Analyseur intégré | Déclins de photoluminescence résolue en temps |
| Données tabulaires | `.xlsx`, `.xls`, `.csv` | `Pandas` & `Openpyxl` | DataFrames pour le tracé graphique |
| Modèle d'ajustement | `.json` | Fit Model Builder | Modèles de pics et configuration de ligne de base |

### **1. Formats `.wdf` (Renishaw), `.spc`, `.l6s` / `.l6m` (HORIBA LabSpec 6) et `.ngs` / `.ngc` (HORIBA LabSpec 5)**

**SPECTROview** prend nativement en charge les formats binaires de spectromètres sans nécessiter de logiciel constructeur externe :

- **`.wdf`** — Données enregistrées avec le logiciel WiRE sur des instruments Renishaw (spectres discrets, séries temporelles ou de répétabilité, profils linéaires 1D et cartes hyperspectrales 2D).
- **`.l6s` et `.l6m`** — Spectres binaires natifs (`.l6s`) et images/cartes spectrales 2D (`.l6m`), compressés ou non compressés, enregistrés avec le logiciel HORIBA LabSpec 6.
- **`.ngs` et `.ngc`** — Spectres binaires natifs (`.ngs`) et images/cartes spectrales 2D compressées (`.ngc`) enregistrés avec le logiciel HORIBA LabSpec 5.
- **`.spc`** — Données enregistrées avec les logiciels LabSpec 6 / T64000 / GRAMS sur des instruments HORIBA et Thermo (spectres Raman/PL discrets et cartes hyperspectrales 2D).

Les spectres discrets comme les jeux de données hyperspectraux sont automatiquement orientés vers le bon espace dès leur ouverture ou leur glisser-déposer dans la fenêtre principale, sans conversion préalable. **SPECTROview** extrait et importe également les **métadonnées** instrumentales (longueur d'onde d'excitation, réseau, objectif, puissance laser, temps d'exposition, accumulations, taille de fente/trou confocal, unités spectrales, pas spatiaux, horodatage d'acquisition ainsi que d'autres paramètres comme le modèle et la température du détecteur, le mode SWIFT ou le rapport signal sur bruit S/N).

### **2. Données spectroscopiques 1D (`.txt`, `.csv`)**

**SPECTROview** accepte les données spectroscopiques 1D aux formats TXT ou CSV. Les fichiers doivent contenir **deux colonnes numériques** séparées par des points-virgules (`;`), des virgules (`,`), des tabulations ou des espaces. Les lignes de commentaire initiales (`#`) et les en-têtes de colonnes sont facultatifs et détectés automatiquement sans perdre le premier point sur les fichiers sans en-tête :

- **Colonne 1** : Valeurs de l'axe X (par exemple, déplacement Raman en cm⁻¹, longueur d'onde en nm, 2θ en degrés). Les axes X décroissants sont automatiquement réordonnés par ordre croissant.
- **Colonne 2** : Valeurs d'intensité correspondantes (u.a.).

### **3. Données hyperspectrales (`.txt`, `.csv`)**

**SPECTROview** prend en charge les données hyperspectrales (cartes 2D de platine motorisée, cartographies de plaquettes *wafer* et cartes *wafer* multi-balayages résolues en temps) aux formats TXT ou CSV, structurées comme suit :

| X    | Y    | R1 | R2 | R3 | ... | Rn |
|------|------|----|----|----|-----|----|
| **X1** | **Y1** | i1 | i1 | i1 | ... | i1 |
| **X2** | **Y2** | i2 | i2 | i2 | ... | i2 |
| **...** | **...** | ... | ... | ... | ... | ... |
| **Xn** | **Yn** | in | in | in | ... | in |

- La **première ligne** (R1 → Rn) définit les valeurs de l'axe X (par exemple, déplacement Raman en cm⁻¹) partagées entre tous les spectres.
- Les **deux premières colonnes** définissent les coordonnées spatiales X et Y de chaque spectre au sein de la carte.
- Les **colonnes suivantes** contiennent les valeurs d'intensité correspondantes pour chaque spectre (I1 → In), de la deuxième ligne jusqu'à la dernière.

> **Remarque** : Les fichiers de cartes natifs Renishaw `.wdf` peuvent être chargés directement dans **SPECTROview** sans conversion. Les cartes 2D exportées au format texte long depuis Renishaw WiRE (colonnes `#X`, `#Y`, `#Wave`, `#Intensity`) peuvent être converties au format matriciel large `.txt` à l'aide de `spectroview.api.io.convert_renishaw_map(input_path, output_path)`.

### **4. Tableaux de données (Excel ou CSV)**

Les classeurs Excel (`.xlsx`, `.xls`) contenant une ou plusieurs feuilles, ainsi que les fichiers CSV, peuvent être importés directement dans l'espace **`Graphs`** pour être visualisés et tracés immédiatement.

### **5. Formats natifs **SPECTROview****

Pour vous permettre d'enregistrer vos analyses et de les reprendre ultérieurement, **SPECTROview** utilise des extensions de fichiers spécifiques à chaque espace de travail :

- **Espace `Maps`** → `.maps`
- **Espace `Spectra`** → `.spectra`
- **Espace `Graphs`** → `.graphs`
- **Modèles d'ajustement** → `.json` (glisser-déposer ou bouton **Load** dans les espaces `Spectra` et `Maps`)
