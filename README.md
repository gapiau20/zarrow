# Zarrow

Construire des **cohortes cliniques** (MIMIC-IV, MIMIC-III…) à partir d'une config YAML, les stocker au format **[Zarr](https://zarr.dev)**, puis les utiliser avec pandas, scikit-learn ou PyTorch.

```
config YAML ──► téléchargement PhysioNet ──► lecture par blocs + filtres ──► store Zarr ──► ML
```

## Installation

Python ≥ 3.10.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt   # + requirements-dev.txt pour pytest
```

PyTorch s'installe en version CPU. Pour le GPU, décommentez la ligne `--extra-index-url` de `requirements.txt`.

### Identifiants PhysioNet (MIMIC-IV complet)

Les démos (`mimic-iv-demo`, `mimiciii-demo`) sont publiques. Pour MIMIC-IV complet, il faut un compte PhysioNet **credentialed** qui a signé l'accord d'utilisation (DUA) du projet. Fournissez vos identifiants de l'une des deux façons suivantes (jamais dans le dépôt) :

```powershell
$env:PHYSIONET_USERNAME = "user"; $env:PHYSIONET_PASSWORD = "mot_de_passe"
```

ou via un fichier `~/.netrc` (sous Windows : `%USERPROFILE%\_netrc`) :

```
machine physionet.org
login user
password mot_de_passe
```

Testez l'accès sur un petit fichier avant de lancer une grosse cohorte :

```python
from modules.download import download_physionet_file
download_physionet_file("mimiciv", "3.1", "hosp/patients.csv.gz", "data/mimiciv-tmp")
```

Un fichier déjà présent dans `tmp_dir` n'est pas retéléchargé. Vous pouvez donc aussi y déposer les tables vous-même.

## Démarrage rapide

```python
from modules.physionet_cohort import MIMICPatientCohort
from modules.cohort import get_schema_from_config
from modules.zarr_tools import ZarrLoader

schema = get_schema_from_config("config/mimiciv_demo.yaml")
cohort = MIMICPatientCohort("data/tmp", "subject_id", schema)
cohort.build_and_save("data/demo.zarr")                  # télécharge, filtre, écrit

df = ZarrLoader("data/demo.zarr").load_group("admission", as_df=True)
```

Exemple complet (mortalité à 28 jours après un infarctus, régression logistique et analyse par sous-groupes) :

```bash
python scripts/mortality28d.py     # config : config/mimic_iv_infarction.yaml
```

Le script ne reconstruit la cohorte que si `data/cohort.zarr` n'existe pas. Supprimez ce dossier pour la reconstruire.

## Définir une cohorte

```yaml
dataset:
  name: mimiciv                 # projet PhysioNet
  version: '3.1'                # version du projet
  diagnoses:                    # un « groupe » = une table = un groupe Zarr
    file: hosp/diagnoses_icd.csv.gz
    inclusion: true             # ses filtres définissent qui est dans la cohorte
    read_options: {encoding: utf-8}   # optionnel, kwargs pandas
    columns: [subject_id, hadm_id, icd_code]
    processor: {name: EHRDataFrameProcessor}
    filters:
      - name: ICDFilter
        parameters: {diag_column: icd_code, icd_codes: ['I21*', '410*']}
```

Règles à connaître :
- **Chaque groupe contient la colonne d'index** (ex. `subject_id`).
- **Les filtres s'appliquent avant la sélection de `columns`** : ils peuvent donc porter sur une colonne non conservée.
- **`inclusion: true`** restreint *tous* les groupes à l'intersection des patients des groupes d'inclusion. Sans ce drapeau, un filtre ne sélectionne que les lignes de son propre groupe (par exemple, quels examens de laboratoire garder).

| Filtre | Paramètres |
|---|---|
| `AgeFilter` | `age_column`, `age_min`, `age_max` (bornes incluses) |
| `SexFilter` | `sex_column`, `sex` |
| `ICDFilter` | `diag_column`, `icd_codes` (le suffixe `*` signifie « préfixe ») |
| `PatientFilter` | `patient_col`, `subject_ids` |
| `LabEventFilter` / `ProcedureFilter` / `MedicationFilter` / `CharteventFilter` | `<x>_column`, `<x>s` (liste de valeurs) |

**Ajouter un filtre ou un processeur** : il suffit d'hériter de `BaseFilter` (méthode `apply(df)`) ou de `DatabaseProcessor` (méthode `process(df)`). La classe est enregistrée automatiquement et utilisable par son nom dans le YAML.

## Store Zarr produit

```
cohort.zarr/
├── subject_id          # index : identifiants uniques triés
├── admission/          # un groupe par groupe de la config
│   ├── subject_id      # une colonne = un array
│   └── admittime       # texte en UTF-8 ; valeur manquante = ''
└── labs/ ...
```

Les tables sont « longues » (une ligne par enregistrement, pas par patient) : faites les jointures sur `subject_id` / `hadm_id`. Pour PyTorch, `MultimodalDataset(zarr_path, "subject_id", groups)` renvoie un item par patient, avec les colonnes numériques triées par nom (voir `dataset.columns`). Le nombre de lignes varie d'un patient à l'autre, donc il faut un `collate_fn` avec padding.

## Organisation du code

| Fichier | Rôle |
|---|---|
| [modules/cohort.py](modules/cohort.py) | `BaseCohort` / `TabularCohort` : pipeline, lecture tabulaire avec détection d'encodage |
| [modules/db_filters.py](modules/db_filters.py), [modules/db_processors.py](modules/db_processors.py) | Filtres et processeurs, registres automatiques |
| [modules/physionet_cohort.py](modules/physionet_cohort.py) | Cohortes PhysioNet (`MIMICPatientCohort`) |
| [modules/download.py](modules/download.py) | Téléchargement PhysioNet authentifié, en streaming, avec reprise |
| [modules/zarr_tools.py](modules/zarr_tools.py) | `ZarrWriter` / `ZarrLoader` |
| [modules/torch_loader.py](modules/torch_loader.py) | `MultimodalDataset` |
| [modules/utils.py](modules/utils.py) | Utilitaires pour les codes CIM (ICD) |
| [modules/features.py](modules/features.py), [modules/multimodal.py](modules/multimodal.py), [main.py](main.py) | Squelettes pour les futures modalités |
| [config/](config/) | Cohortes : `mimiciv_demo`, `mimiciii_demo`, `mimic_iv_infarction` |
| `data/` | Données d'entrée (ignorées par git) |

## Tests

```bash
python -m pytest -q      # 44 tests
```

## Points ouverts

- **Identifiants valides non testés** : l'authentification PhysioNet n'a été validée qu'en cas de refus (403). Le téléchargement public du démo, lui, fonctionne.
- **À valider cliniquement** : les codes d'infarctus (`I21*`, `I22*`, `410*`) et les `itemid` des examens de laboratoire de `mimic_iv_infarction.yaml` (à vérifier dans `d_labitems`).
- **Stores Zarr existants** : ceux écrits avant le correctif du filtrage (drapeau `inclusion`) contiennent tous les patients. Il faut les reconstruire.
- **Fichiers temporaires** : `remove_tmp_dir()` n'est pas appelé automatiquement.
- **Dates** : elles sont stockées comme texte et doivent être converties à la lecture.
- **Index** : `ZarrLoader.load_index()` lit le premier array trouvé à la racine.

## Branches et feuille de route

`master` = code fonctionnel, `dev` = développement (voir [contributing.md](contributing.md)). Prochaines étapes : séries temporelles de laboratoire, ECG, waveforms, MIMIC-III tabulaire, puis IMPROVE, registres d'infarctus et VitalDB.

## Référence

Johnson, A.E.W. et al. *MIMIC-IV, a freely accessible electronic health record dataset.* Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

Licence : voir [LICENSE](LICENSE).
