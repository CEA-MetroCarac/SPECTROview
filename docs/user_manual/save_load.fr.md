## **Sauvegarde et chargement d'un espace de travail**

**SPECTROview** s'appuie sur des formats de fichiers natifs dédiés pour vous permettre d'interrompre, de sauvegarder et de reprendre vos sessions d'analyse sans perdre ni vos configurations ni vos résultats d'ajustement.

### **1. Sauvegarder votre travail**

Pour enregistrer votre avancement, cliquez sur le bouton **Save** dans la barre d'outils principale de l'application :

<div align="center">
  <img src="../user_manual_images/Save_Load/save_load_clear.png" alt="Boutons Save, Load et Clear" width="120"><br>
</div>

L'application génère une extension de fichier spécifique en fonction de l'espace de travail actif :

| Espace actif | Extension native | Contenu enregistré |
|---|---|---|
| **`Spectra`** | `.spectra` | Tous les spectres 1D chargés, les configurations personnalisées de `baseline`, les modèles d'ajustement mathématiques et l'ensemble des résultats d'ajustement. |
| **`Maps`** | `.maps` | Les données brutes de la carte 2D ou *wafer*, les métadonnées instrumentales, tous les spectres associés et tous les résultats d'ajustement point par point. |
| **`Graphs`** | `.graphs` | Tous les jeux de données chargés, les configurations graphiques personnalisées et les annotations dessinées. |

### **2. Charger votre travail**

Pour reprendre une session précédente, cliquez simplement sur le bouton **Open** dans la barre d'outils principale et sélectionnez un fichier `.spectra`, `.maps` ou `.graphs` précédemment sauvegardé. **SPECTROview** détecte automatiquement le format, reconstruit exactement votre jeu de données tel que vous l'aviez laissé et ouvre instantanément l'onglet d'espace de travail correspondant.
