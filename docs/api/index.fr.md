# API

L'API Python de SPECTROview vous permet d'automatiser vos flux d'analyse et d'interagir par programme avec les moteurs de traitement spectroscopique principaux de SPECTROview. Elle est totalement découplée de l'interface graphique Qt (aucun `QApplication` requis), ce qui la rend particulièrement adaptée aux notebooks Jupyter, aux scripts par lots et aux pipelines d'intégration continue.

Tous les modules sont exposés via le paquet `spectroview.api`.

## Modules de l'API

- **[`spectroview.api.workspace`](spectra.md#workspaces-à-état-persistant)** — `SpectraWorkspace` / `MapsWorkspace` : sessions à état persistant permettant de charger, prétraiter, ajuster, collecter les résultats et enregistrer/recharger au sein d'un même objet. Les fichiers lus et écrits utilisent *exactement le même format* que le bouton « Save work » de l'interface graphique — un espace créé intégralement par script s'ouvre directement dans l'application, et inversement.
- **[`spectroview.api.io`](spectra.md#chargement-des-données)** — Lecteurs de fichiers (WDF, SPC, L6S, L6M, NGS, NGC, CSV, TXT, DAT), import de jeux de données, export des résultats et convertisseur Renishaw sans interface graphique.
- **[`spectroview.api.preprocessing`](spectra.md#prétraitement)** — Fonctions sans état opérant directement sur les tableaux : soustraction de ligne de base, recadrage et normalisation.
- **[`spectroview.api.fitting`](spectra.md#ajustement-par-lots)** — Interface vers le moteur Vectorized Batch Fit (VBF) : construction de modèles, ajustement par lots et gestion CRUD des modèles d'ajustement.
- **[`spectroview.api.analysis`](2dmap.md#analyse-multivariée-mva)** — Analyse multivariée : ACP (PCA) et NMF.
- **[`spectroview.api.graphs`](graphs.md)** — Graphiques prêts pour la publication reproduisant au pixel près le rendu de l'interface graphique, pour les 9 styles de tracé pris en charge, ainsi que la gestion des modèles de tracé.
- **[`spectroview.api.calculators`](calculators.md)** — Calculateurs physiques pour l'optique et les conversions d'unités.
- **`spectroview.api.settings`** — Accès en lecture/écriture sans interface graphique à la configuration persistante (paramètres d'ajustement par défaut, dossier des modèles, dernier répertoire utilisé).
- **`spectroview.api.exceptions`** — La hiérarchie `SpectroviewError` levée dans l'ensemble du paquet (`LoadError`, `FitModelError`, `FitError`, `WorkspaceError`, `TemplateError`).

## Explorer l'API

- [Espace Spectra](spectra.md) : Charger des spectres discrets, les prétraiter et exécuter le moteur d'ajustement par lots — aussi bien via la classe à état persistant `SpectraWorkspace` que via les fonctions bas niveau sur tableaux.
- [Espace Maps et analyse 2D](2dmap.md) : Charger des cartes hyperspectrales ou *wafer*, ajuster chaque pixel, générer des cartes thermiques et profils linéaires, et exécuter une ACP/NMF.
- [Visualisation de données (Graphs)](graphs.md) : Reproduire par script les tracés natifs de SPECTROview pour tous les styles de graphiques, ainsi que des recettes réutilisables.
- [Calculateurs rapides](calculators.md) : Convertisseurs optiques et d'unités spectroscopiques par programme.
- [Étendre SPECTROview](extending.md) : Les points d'extension réels basés sur les fichiers (méthodes de ligne de base personnalisées, dossiers de modèles d'ajustement et de graphiques) — il n'existe pas de système de plugins, et cette page explique les alternatives disponibles.

## Gestion des erreurs

Chaque fonction de `spectroview.api` lève une sous-classe de `spectroview.api.exceptions.SpectroviewError` en cas d'échec — jamais un `KeyError` ou `ValueError` brut issu de la couche modèle sous-jacente, et jamais une boîte de dialogue Qt (qui bloquerait un script sans interface). Interceptez `SpectroviewError` de manière générale, ou un sous-type précis (`LoadError`, `FitModelError`, `FitError`, `WorkspaceError`, `TemplateError`) :

```python
from spectroview.api import io
from spectroview.api.exceptions import LoadError

try:
    data = io.load_spectra("missing_file.wdf")
except LoadError as e:
    print(f"Could not load spectra: {e}")
```
