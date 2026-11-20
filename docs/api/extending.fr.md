# Étendre SPECTROview

SPECTROview ne possède pas de système de plugins ou de points d'entrée dynamiques — il n'existe pas de registre pour injecter du code personnalisé dans l'application. À la place, l'extensibilité repose sur deux mécanismes fondés sur les fichiers et directement exposés par l'API. Si vous cherchez comment ajouter un élément personnalisé à SPECTROview, l'une de ces deux approches répond presque toujours au besoin.

---

## Méthodes de ligne de base personnalisées

L'évaluation de ligne de base (`spectroview.api.preprocessing.subtract_baseline`, ainsi que `set_baseline` sur chaque espace de travail) aiguille selon le nom de méthode passé dans `config["mode"]`. Au-delà des modes intégrés `None`, `"Linear"`, `"Polynomial"` et `"arpls"`, **n'importe quelle méthode `Baseline` de [pybaselines](https://pybaselines.readthedocs.io/) fonctionne directement**, sans étape d'enregistrement :

```python
from spectroview.api import preprocessing

# "asls", "airpls", "modpoly", "snip", ... n'importe quel nom de méthode pybaselines.Baseline
Y_corrected, Y_baseline = preprocessing.subtract_baseline(x, Y, {"mode": "asls", "lam": 1e6})
```

Si `pybaselines` ajoute une nouvelle méthode dans une version future, elle devient immédiatement disponible dans SPECTROview sans modification de code de part et d'autre.

---

## Modèles d'ajustement et modèles de graphiques

Les modèles d'ajustement comme les configurations de graphiques sont de simples fichiers JSON placés dans un dossier que vous contrôlez — SPECTROview (interface ou API) se contente de scanner ce dossier. Ainsi, n'importe quel outil externe, script ou dépôt versionné de modèles peut les alimenter :

```python
from spectroview.api import fitting, graphs, settings

# Les modèles d'ajustement résident dans le dossier de travail SPECTROview. Pointez-le vers un
# emplacement partagé — set_working_folder() crée les sous-dossiers
# fit_model/plot_recipe/plot_style/ai_chat_history/template et config.json :
settings.set_working_folder("./shared_spectroview")
# ...tout fichier *.json déposé dans fit_model (via cette API, l'interface ou écrit à la main)
# devient immédiatement visible :
names = fitting.list_fit_model_templates(settings.get_fit_model_folder())

# Les recettes de tracé fonctionnent de la même façon via un dossier dédié :
graphs.save_plot_recipe("./shared_spectroview/plot_recipe", "QC Dashboard", configs=[...])
```

La structure JSON d'un modèle d'ajustement est `{"0": {"peak_labels": [...], "peak_models": {...}, "baseline": {...}, "range_min": ..., "range_max": ..., "fit_params": {...}}}` — générez-en un directement si vous produisez des modèles depuis un système externe, ou créez-en un avec `fitting.build_fit_model()` / `fitting.save_fit_model_template()` pour inspecter le format produit.

---

## Profils de pics personnalisés

Le registre vectorisé des profils de pics (`spectroview.fit_engine.models.BATCHED_MODELS`) est un simple dictionnaire Python associant un nom de profil au triplet `(eval_fn, jacobian_fn, param_names)`. Il ne fait pas partie de l'API publique garantie stable entre versions, mais si vous avez besoin d'une forme de pic non incluse dans SPECTROview, c'est dans ce registre — ainsi que dans `spectroview.fit_engine.scalar_models.PEAK_MODEL_REGISTRY` pour le chemin scalaire de repli — qu'il faudra l'ajouter. Cela nécessite une modification du code source et non une extension à l'exécution ; il n'existe pas actuellement de moyen d'enregistrer un nouveau profil de pic depuis un script sans modifier le paquet installé.
