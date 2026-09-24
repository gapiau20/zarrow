# Zarrow

Framework Python modulaire pour **construire des cohortes cliniques** à partir de bases de données médicales (MIMIC-IV, MIMIC-III…), les **stocker au format [Zarr](https://zarr.dev)** et les exploiter pour du **benchmarking de modèles de machine learning**.

La définition d'une cohorte est entièrement **déclarative** : un fichier YAML décrit les tables à lire, les colonnes à garder, et les filtres à appliquer (âge, codes CIM/ICD, examens biologiques…). Zarrow télécharge les fichiers nécessaires, les traite par blocs (chunks), puis écrit le résultat dans un store Zarr prêt à l'emploi pour pandas, scikit-learn ou PyTorch.

> Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. *MIMIC-IV, a freely accessible electronic health record dataset.* Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

---

## Sommaire

1. [Pipeline](#pipeline)
2. [Arborescence](#arborescence)
3. [Installation](#installation)
4. [Démarrage rapide](#démarrage-rapide)
5. [Format du fichier de configuration](#format-du-fichier-de-configuration)
6. [Référence des modules](#référence-des-modules)
7. [Format du store Zarr produit](#format-du-store-zarr-produit)
8. [Script d'exemple : mortalité à 28 jours](#script-dexemple--mortalité-à-28-jours)
9. [Tests](#tests)
10. [Problèmes identifiés et correctifs appliqués](#problèmes-identifiés-et-correctifs-appliqués)
11. [Feuille de route](#feuille-de-route)
12. [Résumé scientifique](#résumé-scientifique-abstract)

---

## Pipeline

```
   config YAML (dataset, groupes, colonnes, filtres, processeur)
                          │
                          ▼
   get_schema_from_config()  ──►  schema (dict)
                          │
                          ▼
   BaseCohort.__init__ : pour chaque groupe
       ├─ instancie les filtres   (FILTER_REGISTRY)
       └─ instancie le processeur (PROCESSOR_REGISTRY)
                          │
                          ▼
   build_cohort() : pour chaque groupe
       ├─ download_file()   (PhysioNet via wfdb, ou copie locale)
       └─ process_chunks()  lecture CSV par blocs
                            → filtres → sélection des colonnes
                          │
                          ▼
   build_and_save() : ZarrWriter
       ├─ un groupe Zarr par groupe de la config (une colonne = un array)
       └─ un array d'index à la racine (ex. subject_id)
                          │
                          ▼
   ZarrLoader / MultimodalDataset  →  pandas / scikit-learn / PyTorch
```

## Arborescence

| Chemin | Rôle |
|---|---|
| [modules/cohort.py](modules/cohort.py) | Cœur du framework : lecture des fichiers tabulaires, chargement du schéma YAML, classes `BaseCohort` et `TabularCohort` |
| [modules/db_filters.py](modules/db_filters.py) | Filtres composables (`AgeFilter`, `SexFilter`, `ICDFilter`, `LabEventFilter`…) et leur registre automatique |
| [modules/db_processors.py](modules/db_processors.py) | Processeurs appliqués à chaque bloc (`EHRDataFrameProcessor`) et leur registre automatique |
| [modules/physionet_cohort.py](modules/physionet_cohort.py) | Cohortes PhysioNet/MIMIC : téléchargement des tables avec `wfdb.io.dl_files` |
| [modules/improve_cohort.py](modules/improve_cohort.py) | Cohortes à partir de fichiers locaux (registre IMPROVE) : copie du fichier dans le répertoire temporaire |
| [modules/zarr_tools.py](modules/zarr_tools.py) | `ZarrWriter` (DataFrame → Zarr) et `ZarrLoader` (Zarr → dict / DataFrame) |
| [modules/torch_loader.py](modules/torch_loader.py) | `MultimodalDataset`, un `torch.utils.data.Dataset` lisant un store Zarr |
| [modules/features.py](modules/features.py) | Classe de base `FeatureExtractor` pour les futures modalités (ECG, CXR, waveforms, texte). Squelette seulement |
| [modules/utils.py](modules/utils.py) | Utilitaires CIM : détection de la version ICD-9/ICD-10, normalisation des codes |
| [modules/download.py](modules/download.py), [modules/multimodal.py](modules/multimodal.py) | Modules vides (docstring seule), prévus pour la suite |
| [scripts/mortality28d.py](scripts/mortality28d.py) | Exemple de bout en bout : cohorte d'infarctus du myocarde sous MIMIC-IV, prédiction de la mortalité à 28 jours, analyse par sous-groupes |
| [config/](config/) | Configurations de cohortes (`mimiciv_demo.yaml`, `mimiciii_demo.yaml`, `mimic_iv_infarction.yaml`…) |
| [tests/](tests/) | Tests unitaires pytest, un fichier par module |
| [main.py](main.py) | Vide (point d'entrée à venir) |
| `data/` | Données d'entrée, ignorées par git (`.gitignore`) |

## Installation

Python ≥ 3.10 est requis (syntaxe `list[str]`, opérateur walrus). Le code utilise l'API **Zarr v3** (`create_array`, chaînes `dtype=str`) : `zarr>=3.1`.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt        # exécution
pip install -r requirements-dev.txt    # + pytest, ipykernel
```

Par défaut, PyTorch est installé dans sa version CPU. Pour le GPU, décommentez la ligne `--extra-index-url https://download.pytorch.org/whl/cu126` de `requirements.txt`.

Versions validées dans l'environnement de développement : `zarr 3.2.0`, `pandas 3.0.2`, `numpy 2.4.4`.

## Démarrage rapide

```python
from modules.physionet_cohort import MIMICPatientCohort
from modules.cohort import get_schema_from_config
from modules.zarr_tools import ZarrLoader

# 1. Charger la définition de la cohorte
schema = get_schema_from_config("config/mimiciv_demo.yaml")

# 2. Construire la cohorte (téléchargement + filtres) et l'écrire en Zarr
cohort = MIMICPatientCohort(tmp_dir="data/tmp", zarr_index_name="subject_id", schema=schema)
cohort.build_and_save("data/mimiciv_demo.zarr")

# 3. Relire un groupe
loader = ZarrLoader("data/mimiciv_demo.zarr")
df_adm = loader.load_group("admission", as_df=True)
```

`cohort.load(groups, zarr_path)` combine les étapes 2 et 3 : il construit le store s'il n'existe pas, puis renvoie `{groupe: dict de np.ndarray}`.

## Format du fichier de configuration

```yaml
dataset:
  name: mimic-iv-demo            # identifiant PhysioNet passé à wfdb.dl_files
  <groupe>:                      # nom libre, devient un groupe Zarr
    file: hosp/patients.csv.gz   # chemin relatif à tmp_dir (et au dépôt PhysioNet)
    inclusion: true              # optionnel : ses filtres restreignent TOUS les groupes
    read_options:                # optionnel : kwargs du lecteur pandas
      encoding: utf-8            #   (évite la détection d'encodage sur les gros fichiers)
      usecols: [subject_id, anchor_age]
    columns:                     # colonnes conservées après filtrage
      - subject_id
      - anchor_age
    processor:
      name: EHRDataFrameProcessor   # nom d'une sous-classe de DatabaseProcessor
    filters:                     # optionnel, appliqués dans l'ordre
      - name: AgeFilter          # nom d'une sous-classe de BaseFilter
        parameters:              # kwargs du constructeur
          age_column: anchor_age
          age_min: 18
          age_max: 99
```

Chaque groupe **doit** contenir la colonne d'index (`zarr_index_name`, ex. `subject_id`). Un filtre peut porter sur une colonne absente de `columns`, car les filtres sont appliqués **avant** la sélection des colonnes.

**Critères d'inclusion ou sélection de lignes.** Par défaut, les filtres d'un groupe ne sélectionnent que les lignes de ce groupe : par exemple, `LabEventFilter` choisit quels examens garder. Avec `inclusion: true`, les patients retenus par ce groupe définissent la cohorte : tous les groupes sont restreints à l'intersection des patients des groupes d'inclusion.

### Filtres disponibles ([modules/db_filters.py](modules/db_filters.py))

| Filtre | Paramètres | Effet |
|---|---|---|
| `PatientFilter` | `patient_col`, `subject_ids` | Garde une liste explicite de patients |
| `AgeFilter` | `age_column`, `age_min`, `age_max` | Bornes d'âge inclusives |
| `SexFilter` | `sex_column`, `sex` | Égalité stricte |
| `ICDFilter` | `diag_column`, `icd_codes` | Codes exacts, ou préfixe si le code se termine par `*` (ex. `I21*`) |
| `EventFilter` | `event_column`, `events` | `isin` générique |
| `LabEventFilter` | `lab_event_column`, `lab_events` | `EventFilter` sur les `itemid` de laboratoire |
| `ProcedureFilter` | `procedure_event_column`, `procedure_events` | `EventFilter` sur les procédures |
| `MedicationFilter` | `medication_event_column`, `medication_events` | `EventFilter` sur les médicaments |
| `CharteventFilter` | `chartevent_column`, `chartevents` | `EventFilter` sur les chartevents |

### Ajouter un filtre ou un processeur

Les registres sont remplis automatiquement par `__init_subclass__`. Il suffit donc de définir la sous-classe dans un module importé avant la construction de la cohorte :

```python
from modules.db_filters import BaseFilter

class LengthOfStayFilter(BaseFilter):
    def __init__(self, min_days):
        self.min_days = min_days
    def apply(self, df):
        los = (pd.to_datetime(df.dischtime) - pd.to_datetime(df.admittime)).dt.days
        return df[los >= self.min_days]
```

Le filtre devient alors utilisable dans le YAML sous `name: LengthOfStayFilter`.

## Référence des modules

### `modules/cohort.py`
- `detect_encoding(path)` renvoie le premier encodage (utf-8, cp1252, latin1) capable de décoder tout le fichier, lu en flux, `.gz` compris.
- `read_tabular_file(path, **kwargs)` lit un CSV (y compris `.csv.gz`, avec l'encodage détecté si aucun n'est fourni), un fichier Excel, Parquet ou JSON. Avec `chunksize`, il renvoie toujours un itérable de DataFrames.
- `get_schema_from_tabular_file(path)` et `build_default_schema_from_tabular_file(path, yaml)` génèrent un squelette de schéma (toutes les colonnes, sans filtre) à partir d'un fichier.
- `get_schema_from_config(path)` renvoie la section `dataset` du YAML.
- `BaseCohort(tmp_dir, zarr_index_name, schema, chunk_size=100_000)` instancie un processeur et ses filtres par groupe.
  - `build_cohort()` → `{groupe: DataFrame}`, restreint aux groupes `inclusion: true`.
  - `build_and_save(zarr_path)` écrit chaque groupe, puis l'index (identifiants uniques et triés, tous groupes confondus).
  - `load(groups, zarr_path)` construit le store si besoin, puis le lit.
  - `remove_tmp_dir()` supprime le répertoire temporaire. Il n'est pas appelé automatiquement (TODO).
  - `download_file(file)` est abstraite et doit être implémentée par les sous-classes.
- `TabularCohort.process_chunks(path, processor, **read_options)` lit le fichier par blocs de `chunk_size` lignes, applique le processeur à chacun, puis concatène. Si aucune ligne n'est retenue, il renvoie un DataFrame vide.

### `modules/db_processors.py`
- `DatabaseProcessor(zarr_index, columns, filters)` est la classe de base. Elle fournit `add_filters()` et la méthode abstraite `process(df) -> (df, index)`.
- `EHRDataFrameProcessor` applique les filtres, puis garde `columns` (toutes les colonnes si `columns` vaut `None`).
- `PatientPreprocessor` est un squelette sans implémentation.

### `modules/physionet_cohort.py`
- `PhysioNetPatientCohort.download_file(file)` télécharge `file` depuis le dépôt PhysioNet `schema['name']` vers `tmp_dir` (`wfdb.io.dl_files`, `keep_subdirs=True`), sauf si `tmp_dir/file` existe déjà.
- `MIMICPatientCohort` est un alias sémantique.

### `modules/improve_cohort.py`
- `IMPROVECohort.download_file(file)` ne télécharge rien : il vérifie que le fichier local existe et le copie dans `tmp_dir`.

### `modules/zarr_tools.py`
- `ZarrWriter(path, index_id)` ouvre le store en mode `w`, ce qui **écrase** tout store existant.
  - `write_dataframe(df, group)` : une colonne devient un array. Le texte est stocké en chaînes UTF-8 de longueur variable (valeur manquante → `''`), `int64` en `int32` si les valeurs tiennent, `float64` en `float32`. Les chunks font au plus 1024 éléments.
  - `write_index(ids)` écrit l'array d'index à la racine.
- `ZarrLoader(path)` fournit `load_group(group, as_df)` et `load_index()`, qui lit le premier array trouvé à la racine. Les anciens stores en bytes `S` sont décodés en UTF-8.

### `modules/torch_loader.py`
- `MultimodalDataset(zarr_path, index_name, groups)` renvoie un item par patient de l'index : `{groupe: tensor float32 (lignes × colonnes numériques)}`. Le nombre de lignes varie d'un item à l'autre, donc un `collate_fn` avec padding est nécessaire pour le `DataLoader`.

### `modules/utils.py`
- `find_icd_version(code)` renvoie 9 si le code commence par un chiffre, 10 sinon.
- `normalize_icd_code(code)` insère le point des codes ICD-9 numériques (`41071` → `410.71`). Cette fonction n'est utilisée nulle part dans le pipeline.

## Format du store Zarr produit

```
cohort.zarr/
├── subject_id            # array d'index (racine)
├── patient/
│   └── subject_id
├── admission/
│   ├── subject_id
│   ├── hadm_id
│   ├── admittime         # chaînes UTF-8
│   └── ...
├── demographics/ ...
├── diagnoses/ ...
└── labs/ ...
```

Chaque groupe est une table « longue » : une ligne par enregistrement, et non par patient. Les jointures se font sur `subject_id` et `hadm_id`.

## Script d'exemple : mortalité à 28 jours

[scripts/mortality28d.py](scripts/mortality28d.py), avec la configuration [config/mimic_iv_infarction.yaml](config/mimic_iv_infarction.yaml) :

1. Construit la cohorte MIMIC-IV si `data/cohort.zarr` n'existe pas : adultes de 18 à 99 ans, diagnostics d'infarctus `I21*`, `I22*` ou `410*`, 6 examens biologiques.
2. Garde les admissions avec un diagnostic d'infarctus, puis joint la démographie et les données sociales.
3. Définit la cible `mortality_28d = dod − admittime ≤ 28 jours`, sur la première admission pour infarctus de chaque patient.
4. Découpe en train/test (70/30), stratifié sur sexe × origine ethnique simplifiée × événement.
5. Entraîne une régression logistique sur l'âge, le sexe et le statut marital, puis compare l'AUC du modèle global à celle de modèles entraînés par sous-groupe.

```bash
python scripts/mortality28d.py
```

> MIMIC-IV (version complète) est une base à **accès restreint** (credentialed) sur PhysioNet. `wfdb.dl_files` ne gère pas l'authentification ; vérifiez ce point dans votre environnement. En pratique, il est plus sûr de télécharger les tables manuellement (`wget --user … -r`) dans `tmp_dir` : `download_file` ignorera alors le téléchargement.

## Tests

```bash
python -m pytest -q
```

État actuel : **37 tests réussis**, dont des tests de non-régression pour chaque correctif ci-dessous.

---

## Problèmes identifiés et correctifs appliqués

Chaque bug ci-dessous a été **reproduit** avec un script avant correction. **Tous les correctifs décrits ici sont appliqués** et couverts par des tests (`pytest`). Cette section est conservée comme journal des changements.

⚠️ Changement de format : les stores Zarr existants ont été écrits avec l'ancien filtrage, qui ne restreignait pas la cohorte. Il faut les **reconstruire**, par exemple en supprimant `data/cohort.zarr`.

### Vue d'ensemble

| # | Gravité | Fichier | Problème (✅ corrigé) |
|---|---|---|---|
| 1 | 🔴 Critique | [modules/cohort.py](modules/cohort.py#L137-L150) | Les filtres ne s'appliquent qu'à leur groupe : la cohorte n'est pas restreinte |
| 2 | 🔴 Critique | [scripts/mortality28d.py](scripts/mortality28d.py#L45-L59) | Le jeu de modélisation contient tous les patients (pas seulement les infarctus), et `groupby().first()` mélange les lignes |
| 3 | 🔴 Critique | [modules/cohort.py](modules/cohort.py#L15-L33) | Le repli d'encodage est inopérant en lecture par chunks, et les formats non-CSV plantent |
| 4 | 🔴 Critique | [modules/zarr_tools.py](modules/zarr_tools.py#L9-L34) | Les caractères non-ASCII plantent, NaN devient `"nan"`, les entiers débordent en int32, un groupe vide plante |
| 5 | 🟠 Majeur | [modules/cohort.py](modules/cohort.py#L162) | `build_and_save` exige un groupe nommé `patient` |
| 6 | 🟠 Majeur | [modules/db_processors.py](modules/db_processors.py#L16) | Défaut mutable `filters=[]` partagé entre instances |
| 7 | 🟠 Majeur | [modules/cohort.py](modules/cohort.py#L184) | `pd.concat([])` plante si un filtre ne garde aucune ligne |
| 8 | 🟠 Majeur | [modules/physionet_cohort.py](modules/physionet_cohort.py#L32) | Un fichier présent dans le répertoire courant n'est ni téléchargé ni lu |
| 9 | 🟠 Majeur | [config/mimic_iv_infarction.yaml](config/mimic_iv_infarction.yaml#L205) | Les infarctus codés en ICD-9 sont exclus |
| 10 | 🟡 Mineur | [tests/test_db_processors.py](tests/test_db_processors.py#L24) | Le test attend que la classe de base soit enregistrée |
| 11 | 🟡 Mineur | [modules/torch_loader.py](modules/torch_loader.py) | `MultimodalDataset` est incompatible avec le store produit |
| 12 | 🟡 Mineur | [modules/db_processors.py](modules/db_processors.py#L48) | `columns: None` provoque une `KeyError` |
| 13 | 🟡 Mineur | [modules/cohort.py](modules/cohort.py#L105) | Un `processor` absent lève une `AttributeError` au lieu d'un message clair |
| 14 | 🟡 Mineur | [modules/cohort.py](modules/cohort.py#L166-L168) | Préfixe `f` manquant dans deux `print` |
| 15 | 🟡 Mineur | [scripts/mortality28d.py](scripts/mortality28d.py) | Import inutilisé, docstring erronée, reconstruction à chaque exécution, catégories inconnues, chemins Windows |
| 16 | ⚪ Perf | [modules/cohort.py](modules/cohort.py#L78) | `chunk_size=64` et lecture de toutes les colonnes |
| 17 | ⚪ Qualité | [modules/utils.py](modules/utils.py#L3-L13) | Codes ICD-9 `E`/`V` classés en ICD-10, chaîne vide |
| 18 | ⚪ Qualité | [requirements.txt](requirements.txt) | Versions non fixées, dépendances inutilisées |

---

### 1. Les filtres ne s'appliquent qu'à leur groupe : la cohorte n'est pas restreinte

**Constat.** Chaque groupe est filtré indépendamment. Dans `mimic_iv_infarction.yaml`, `ICDFilter(I21*)` ne filtre que le groupe `diagnoses`, et `AgeFilter` que `demographics`. Les groupes `patient`, `admission` et `social` contiennent donc **tous** les patients de MIMIC-IV, et l'index Zarr aussi. Le nom « cohorte » est trompeur : aucun critère d'inclusion n'est réellement appliqué.

Tous les filtres ne sont pas pour autant des critères d'inclusion : `LabEventFilter` sélectionne des lignes (quels examens garder) et ne décide pas qui entre dans la cohorte. Le correctif ajoute donc un drapeau explicite `inclusion: true` au niveau du groupe.

**Correctif dans [modules/cohort.py](modules/cohort.py).** À la fin de `BaseCohort.__init__`, avant `return` :

```python
        # groups whose filters define who belongs to the cohort (intersection of their subjects)
        self.inclusion_groups=[k for k, v in self.schema.items() if k!='name' and v.get('inclusion', False)]
```

Dans `build_cohort`, remplacer le `return cohort` final par :

```python
        if self.inclusion_groups:
            ids=set.intersection(*(set(cohort[g][self.zarr_index]) for g in self.inclusion_groups))
            cohort={g: df[df[self.zarr_index].isin(ids)] for g, df in cohort.items()}
        return cohort
```

**Correctif dans [config/mimic_iv_infarction.yaml](config/mimic_iv_infarction.yaml).** Ajouter `inclusion: true` aux groupes `demographics` et `diagnoses` :

```yaml
  demographics:
    file: hosp/patients.csv.gz
    inclusion: true
    ...
  diagnoses:
    file: hosp/diagnoses_icd.csv.gz
    inclusion: true
    ...
```

Test associé : `test_inclusion_groups_restrict_every_group` dans `tests/test_cohort.py`.

### 2. `mortality28d.py` : cohorte non restreinte aux infarctus, et lignes mélangées

**Constat.**
- La jointure `df_adm.merge(df_diag, how='left')` ([ligne 50](scripts/mortality28d.py#L50)) garde toutes les admissions, y compris celles sans diagnostic d'infarctus. Le modèle est donc entraîné sur l'ensemble de MIMIC-IV, conséquence directe du bug 1.
- `df.groupby('subject_id').first()` ([ligne 59](scripts/mortality28d.py#L59)) renvoie, **colonne par colonne**, la première valeur **non nulle**. Il peut donc combiner l'`admittime` d'une admission avec le `marital_status` d'une autre. Reproduit : `{t:1, x:None}` + `{t:2, x:'B'}` → `{t:1, x:'B'}`.
- La fusion avec les diagnostics produit une ligne par code CIM. Ce n'est pas faux après déduplication, mais c'est inutile.

**Correctif dans [scripts/mortality28d.py](scripts/mortality28d.py#L37-L59).** Remplacer les lignes 37 à 59 par :

```python
loader=ZarrLoader(zarr_path)
df_adm = pd.DataFrame(loader.load_group("admission"))
df_demo = pd.DataFrame(loader.load_group("demographics"))
df_diag = pd.DataFrame(loader.load_group("diagnoses"))
df_social = pd.DataFrame(loader.load_group("social"))

# keep only admissions with a myocardial infarction diagnosis (diagnoses group is already ICD-filtered)
mi_adm = df_diag[['subject_id', 'hadm_id']].drop_duplicates()
df = df_adm.merge(mi_adm, on=['subject_id', 'hadm_id'], how='inner')
df = df.merge(df_demo, on='subject_id', how='inner')          # inner: applies the age filter
df = df.merge(df_social, on=['subject_id', 'hadm_id'], how='left')

df["admittime"] = pd.to_datetime(df["admittime"])
df["dod"] = pd.to_datetime(df["dod"], errors="coerce")
df["mortality_28d"] = (
    (df["dod"].notna()) &
    ((df["dod"] - df["admittime"]).dt.days <= 28)
).astype(int)

# index admission = first MI admission of each patient (whole row, no column mixing)
df = df.sort_values(by=['subject_id', 'admittime']).drop_duplicates('subject_id', keep='first')
```

Une fois le bug 1 corrigé, les jointures `inner` restent correctes et deviennent redondantes, sans risque.

### 3. Repli d'encodage inopérant en lecture par chunks ; formats non-CSV incompatibles

**Constat.** `process_chunks` appelle toujours `read_tabular_file(path, chunksize=...)`.
- Avec `chunksize`, `pd.read_csv` renvoie un itérateur. Le `UnicodeDecodeError` est levé **pendant l'itération**, hors du `try`, donc le repli cp1252/latin1 ne s'exécute jamais. Reproduit : un CSV cp1252 contenant `Müller` lève `UnicodeDecodeError`.
- `pd.read_excel`, `read_parquet` et `read_json` n'acceptent pas `chunksize`. Reproduit : `TypeError: read_excel() got an unexpected keyword argument 'chunksize'`.
- La détection `'.gz' in filepath` est fragile (un répertoire `x.gz/` suffit à la tromper) et échoue si `filepath` est un `Path`.

**Correctif dans [modules/cohort.py](modules/cohort.py#L15-L33).** Ajouter `import codecs` et `import gzip` en tête du fichier, puis remplacer `read_tabular_file` par :

```python
def detect_encoding(filepath:str, encodings=("utf-8", "cp1252", "latin1"), block_size:int=1<<20)->str:
    '''Return the first encoding able to decode the whole file (streamed, gzip aware).'''
    opener = gzip.open if filepath.lower().endswith('.gz') else open
    for enc in encodings:
        decoder = codecs.getincrementaldecoder(enc)()
        try:
            with opener(filepath, 'rb') as f:
                while block := f.read(block_size):
                    decoder.decode(block)
                decoder.decode(b'', final=True)
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode file {filepath} with tried encodings.")

def read_tabular_file(filepath:str,**kwargs):
    """Read a file using pandas based on its extension.
    With chunksize, always returns an iterable of DataFrames (like pd.read_csv)."""
    filepath = str(filepath)
    name = filepath.lower()
    ext = os.path.splitext(name[:-3] if name.endswith('.gz') else name)[-1]

    if ext == ".csv":
        # pass encoding explicitly (e.g. from the YAML) to skip the detection pass on large files
        kwargs.setdefault("encoding", detect_encoding(filepath))
        return pd.read_csv(filepath, **kwargs)

    chunksize = kwargs.pop("chunksize", None)
    if ext in [".xls", ".xlsx"]:
        df = pd.read_excel(filepath, **kwargs)
    elif ext == ".parquet":
        df = pd.read_parquet(filepath, **kwargs)
    elif ext == ".json":
        df = pd.read_json(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
    return [df] if chunksize else df
```

La détection lit le fichier une fois en entier. Pour `labevents.csv.gz` (plusieurs Go), mieux vaut passer `encoding="utf-8"` explicitement, par exemple via une clé `read_options` du YAML transmise par `process_chunks` (voir le bug 16).

### 4. `ZarrWriter` / `ZarrLoader` : Unicode, valeurs manquantes, débordement, groupe vide

**Constat** (tous reproduits) :
- `data.astype("S")` encode en ASCII. Une chaîne non-ASCII (`Müller`, `é`, `ß`) lève `UnicodeEncodeError` et fait échouer toute la sauvegarde.
- Les `None` et `NaN` des colonnes texte deviennent la chaîne `"nan"`, et ne sont plus détectés comme valeurs manquantes à la relecture.
- `int64 → int32` sans contrôle : `3 000 000 000` est relu `-1 294 967 296`.
- Un DataFrame vide donne `chunks=(0,)` → `ValueError: chunk edge length must be >= 1`. Même problème dans `write_index`.
- Côté lecture, `arr.astype(str)` sur des bytes décode en ASCII, ce qui plante sur toute donnée UTF-8.
- Les bytes `S` n'ont pas de spécification Zarr v3 (`UnstableSpecificationWarning`).

**Correctif dans [modules/zarr_tools.py](modules/zarr_tools.py#L9-L42).** Utiliser le type chaîne UTF-8 à longueur variable de Zarr v3 (`dtype=str`, spécifié et sans avertissement) :

```python
    def write_dataframe(self, df, group_name):
        group = self.store.require_group(group_name)
        chunks = (max(1, min(len(df), 1024)),)
        for col in df.columns:
            data = df[col].to_numpy(copy=True)

            # strings -> variable-length UTF-8 (Zarr v3 spec), missing values -> ''
            if data.dtype == object:
                data = pd.Series(data).fillna("").astype(str).to_numpy(dtype=object)
                arr = group.create_array(name=col, shape=data.shape, dtype=str, chunks=chunks, overwrite=True)
                arr[:] = data
                continue

            # int64/float64 -> int32/float32, only when values fit
            if data.dtype.kind == "i" and (len(data) == 0 or (
                    data.min() >= np.iinfo(np.int32).min and data.max() <= np.iinfo(np.int32).max)):
                data = data.astype("int32")
            elif data.dtype.kind == "f":
                data = data.astype("float32")

            group.create_array(name=col, data=data, chunks=chunks, overwrite=True)

    def write_index(self, subject_ids:np.array):
        self.store.create_array(
            name=self.index_id,
            data=subject_ids,
            chunks=(max(1, min(len(subject_ids), 1024)),),
            overwrite=True
        )
```

Dans `ZarrLoader.load_group` ([lignes 59-61](modules/zarr_tools.py#L59-L61)) :

```python
            if arr.dtype.kind == "S":      # stores written by the previous version (utf-8 bytes)
                arr = np.char.decode(arr, "utf-8")
            elif arr.dtype.kind == "T":    # numpy StringDType -> object for pandas
                arr = arr.astype(object)
```

Le `del group[col]` devient inutile grâce à `overwrite=True`. Les stores déjà écrits restent lisibles via la branche `"S"`.

### 5. `build_and_save` exige un groupe nommé `patient`

**Constat.** `writer.write_index(cohort['patient'][...])` ([ligne 162](modules/cohort.py#L162)). Une configuration sans groupe `patient` lève `KeyError: 'patient'` après tout le traitement, c'est-à-dire au dernier moment. De plus, l'index ne reflète pas les filtres (bug 1).

**Correctif dans [modules/cohort.py](modules/cohort.py#L162)** :

```python
        ids = pd.concat([df[self.zarr_index] for df in cohort.values()]).drop_duplicates().sort_values()
        writer.write_index(ids.to_numpy())
```

### 6. Défaut mutable `filters=[]` partagé entre toutes les instances

**Constat.** `DatabaseProcessor.__init__(..., filters=[])` et `EHRDataFrameProcessor.__init__(..., filters=[])`. `add_filters()` fait un `extend` sur cette liste partagée. Reproduit : après `EHRDataFrameProcessor('id').add_filters([SexFilter(...)])`, une **nouvelle** instance `EHRDataFrameProcessor('id')` possède déjà ce filtre.

**Correctif dans [modules/db_processors.py](modules/db_processors.py#L16-L19)** :

```python
    def __init__(self, zarr_index:str,columns:list[str]=None, filters:list[BaseFilter]=None):
        self.zarr_index=zarr_index
        self.columns=columns
        self.filters=list(filters) if filters else []
```

Et [ligne 37](modules/db_processors.py#L37) : `filters:list[BaseFilter]=None`.

### 7. `process_chunks` plante si aucune ligne ne passe les filtres

**Constat.** `pd.concat([])` → `ValueError: No objects to concatenate` ([ligne 184](modules/cohort.py#L184)). C'est fréquent sur les jeux démo avec des filtres restrictifs.

**Correctif dans [modules/cohort.py](modules/cohort.py#L184)** :

```python
        if not cohort:
            return pd.DataFrame(columns=processor.columns)
        cohort=pd.concat(cohort,axis=0)
```

### 8. `PhysioNetPatientCohort.download_file` teste le mauvais chemin

**Constat.** Le téléchargement est ignoré si `file` existe **relativement au répertoire courant** ([ligne 32](modules/physionet_cohort.py#L32)), mais `process_chunks` lit ensuite `tmp_dir/file`, d'où un `FileNotFoundError`. `IMPROVECohort` contourne le problème en copiant le fichier, ce que la version PhysioNet ne fait pas.

**Correctif dans [modules/physionet_cohort.py](modules/physionet_cohort.py#L25-L36)** :

```python
    def download_file(self,file):
        '''Download the necessary csv files to build the clinical cohort'''
        os.makedirs(self.tmp_dir,exist_ok=True)
        if os.path.exists(os.path.join(self.tmp_dir, file)):
            print(f"{file} already present in {self.tmp_dir}. Skipping download.")
            return
        print("Downloading files from PhysioNet...")
        dl_files(self.db,self.tmp_dir,[file],keep_subdirs=True)
```

`tests/test_physionet_cohort.py` passe un chemin absolu : `os.path.join(tmp_dir, chemin_absolu)` renvoie ce chemin absolu, donc le test reste valide.

Nettoyer aussi les imports inutilisés (`files`, `shutil`, `Path`, `pd`, `DatabaseProcessor`, `normalize_icd_code`). Conserver `get_schema_from_config`, que `mortality28d.py` importe depuis ce module.

### 9. Infarctus codés en ICD-9 exclus de la cohorte

**Constat.** MIMIC-IV couvre 2008-2019. Les diagnostics y sont codés en ICD-9 jusqu'au passage à ICD-10 (fin 2015), et `diagnoses_icd.icd_version` vaut 9 ou 10. Le filtre `['I21*']` ne capture que les codes ICD-10. Une part importante des infarctus (code ICD-9 `410.x`, stocké sans point : `410xx`) est donc perdue.

**Correctif dans [config/mimic_iv_infarction.yaml](config/mimic_iv_infarction.yaml#L205)** :

```yaml
          icd_codes: ['I21*', 'I22*', '410*']   # ICD-10 acute + subsequent MI, ICD-9 AMI
```

Le préfixe `410*` n'est pas ambigu : aucun code ICD-10 ne commence par un chiffre. À valider cliniquement : `I22*` (récidive) et le 5e caractère ICD-9 (`4107x2` = épisode ultérieur de soins).

### 10. Test du registre de processeurs en échec

**Constat.** `__init_subclass__` n'enregistre que les sous-classes, ce qui est le comportement voulu, cohérent avec `FILTER_REGISTRY` (qui n'inclut pas `BaseFilter`). Le test se trompe donc en attendant `"DatabaseProcessor"` dans le registre.

**Correctif dans [tests/test_db_processors.py](tests/test_db_processors.py#L22-L25)** :

```python
def test_register_database_processors_returns_processor_types():
    registry = register_database_processors()
    assert "DatabaseProcessor" not in registry  # abstract base is not registered
    assert "EHRDataFrameProcessor" in registry
```

### 11. `MultimodalDataset` est incompatible avec le store produit par `ZarrWriter`

**Constat.** `MultimodalDataset` suppose un sous-groupe par échantillon. Or `ZarrWriter` écrit un groupe **par table**, plus un array d'index à la racine. Reproduit sur un store réel :
- `dataset[0]` renvoie la table `patient` **entière**, et non un patient ;
- `dataset[1]` tombe sur l'array d'index et lève `AttributeError: 'Array' object has no attribute 'array_keys'`.

Par ailleurs, `torch.tensor` échoue sur les colonnes de chaînes.

**Correctif proposé dans [modules/torch_loader.py](modules/torch_loader.py)** : un item par patient de l'index, contenant les colonnes numériques de chaque groupe demandé.

```python
'''
Dataset and dataloader utilities for pytorch.
'''
import numpy as np
import torch
from torch.utils.data import Dataset

from modules.zarr_tools import ZarrLoader

class MultimodalDataset(Dataset):
    '''
    One item per subject of the zarr index.
    Returns, for each requested group, the numeric columns of that subject's rows as a float tensor.
    '''
    def __init__(self, zarr_path:str, index_name:str, groups:list[str]):
        loader = ZarrLoader(zarr_path)
        self.ids = loader.store[index_name][:]
        self.tables = {g: dict(tuple(loader.load_group(g, as_df=True).groupby(index_name))) for g in groups}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sid = self.ids[idx]
        out = {}
        for g, rows_by_id in self.tables.items():
            rows = rows_by_id.get(sid)
            values = rows.select_dtypes("number").to_numpy() if rows is not None else np.empty((0, 0))
            out[g] = torch.tensor(values, dtype=torch.float32)
        return out
```

Les items ont un nombre de lignes variable, donc il faut un `collate_fn` (padding) pour le `DataLoader`. `tests/test_torch_loader.py` est à réécrire en conséquence, par exemple sur un store créé via `ZarrWriter`.

### 12. `EHRDataFrameProcessor.process` avec `columns=None`

**Constat.** La signature autorise `columns=None`, mais `df[None]` lève une `KeyError`.

**Correctif dans [modules/db_processors.py](modules/db_processors.py#L48)** :

```python
        return (df if self.columns is None else df[self.columns]), self.zarr_index
```

### 13. Message d'erreur trompeur si `processor` manque dans le YAML

**Constat.** `group_info.get('processor', None).get('name', None)` lève `AttributeError: 'NoneType' object has no attribute 'get'` au lieu du `ValueError` prévu juste en dessous.

**Correctif dans [modules/cohort.py](modules/cohort.py#L105)** :

```python
            processor_cls_name=(group_info.get('processor') or {}).get('name')
```

### 14. Préfixe `f` manquant dans `load()`

**Correctif dans [modules/cohort.py](modules/cohort.py#L166-L168)** :

```python
            print(f'Building cohort at: {zarr_path}')
        ...
        print(f'Loading cohort from {zarr_path}')
```

### 15. `scripts/mortality28d.py` : autres points

| Ligne | Problème | Correctif |
|---|---|---|
| [3-4](scripts/mortality28d.py#L3-L4) | La docstring cite les codes `I21*, E785, R570` et la « troponine, itemid 50912 ». La config n'utilise que `I21*`, et 50912 correspond à la créatinine dans `d_labitems` (troponine T : 51003). Les labs ne sont d'ailleurs pas utilisés par le modèle | Aligner la docstring sur la config ; vérifier les `itemid` de [la config](config/mimic_iv_infarction.yaml#L220) dans `d_labitems` |
| [13](scripts/mortality28d.py#L13) | `import icdlookup` n'est pas utilisé et impose une dépendance | Supprimer |
| [19-22](scripts/mortality28d.py#L19-L22) | La cohorte est reconstruite (et `labevents` relu) à chaque exécution | `if not os.path.exists(zarr_path): cohort.build_and_save(zarr_path)` |
| [19, 22, 26](scripts/mortality28d.py#L19) | Chemins codés en dur avec `\\` (Windows uniquement) | `Path('data') / 'mimiciv-tmp'`, etc. |
| [91](scripts/mortality28d.py#L91) | `OneHotEncoder(drop='first')` plante si une modalité n'apparaît que dans le test | `OneHotEncoder(drop='first', handle_unknown='ignore')` |
| [67-72](scripts/mortality28d.py#L67-L72) | La stratification porte sur sexe × origine, pas sur la cible, alors que l'événement est rare | Stratifier sur `strata + '_' + mortality_28d` |

### 16. Performances : `chunk_size=64` et lecture de toutes les colonnes

**Constat.** Avec `chunk_size=64`, `labevents.csv.gz` (plus de 100 millions de lignes) est découpé en plus d'un million de blocs, avec un surcoût pandas sur chacun. Toutes les colonnes sont parsées alors que seules `columns` et les colonnes filtrées sont utiles. Un même fichier (`admissions`, `patients`) est aussi relu pour chaque groupe qui le référence.

**Correctif dans [modules/cohort.py](modules/cohort.py#L78)** : passer le défaut à `chunk_size:int=100_000`. Puis, dans `build_cohort`, transmettre des options de lecture optionnelles depuis le YAML :

```python
            group=self.process_chunks(os.path.join(self.tmp_dir, dataset_csv_file), self.processors[p],
                                      **self.schema[p].get('read_options', {}))
```

avec, dans `process_chunks` :

```python
    def process_chunks(self,file_path:str,processor:DatabaseProcessor,**read_options)->pd.DataFrame:
        ...
        for chunk in read_tabular_file(file_path,chunksize=self.chunk_size,**read_options):
```

Exemple de YAML : `read_options: {encoding: utf-8, usecols: [subject_id, hadm_id, itemid, charttime, value]}`.

### 17. `modules/utils.py` : classification ICD fragile

**Constat.** Les codes ICD-9 supplémentaires `E800-E999` et `V01-V91` commencent par une lettre et sont classés en ICD-10. `find_icd_version('')` lève `IndexError`. L'annotation de retour dit `str` alors que la fonction renvoie un `int`. Ces fonctions ne sont pas utilisées par le pipeline.

**Correctif.** Pour MIMIC, préférer la colonne `icd_version`, fiable. Côté code ([lignes 3-13](modules/utils.py#L3-L13)) :

```python
def find_icd_version(code:str)->int:
    """
    Guess the ICD version (9 or 10) of a code from its format.
    Ambiguous for ICD-9 E/V codes: prefer the dataset's icd_version column when available.
    """
    code = str(code).strip().upper()
    if not code:
        raise ValueError("Empty ICD code.")
    return 9 if code[0].isdigit() else 10
```

### 18. `requirements.txt`

- Aucune version n'est fixée, alors que le code dépend de l'API **Zarr v3** (`create_array`, `dtype=str`). Il faut au minimum `zarr>=3.1`, `pandas>=2.2` et `numpy>=2`.
- `icdlookup` et `icdmap` ne sont utilisés nulle part, hormis l'import mort du bug 15.
- `ipykernel` et `pytest` relèvent du développement : déplacés dans `requirements-dev.txt`.
- L'index CUDA imposé (`cu126`) empêche une installation CPU simple : la ligne est désormais commentée.

### Autres points (hors bugs)

- `remove_tmp_dir()` est commenté dans `build_cohort` ([ligne 149](modules/cohort.py#L149), TODO) : les fichiers temporaires s'accumulent.
- `ZarrLoader.load_index()` suppose que l'index est le **premier** array racine. Il vaudrait mieux lire `self.store[index_id]`.
- Les données texte sont stockées telles quelles : les dates restent des chaînes, à convertir à la lecture.
- Modules vides ou squelettes : `main.py`, `modules/download.py`, `modules/multimodal.py`, les méthodes de `modules/features.py` et `PatientPreprocessor`.
- `tests/test_download.py`, `tests/test_multimodal.py` et `tests/test_features.py` ne testent que l'existence des modules.

---

## Feuille de route

Reprise de [contributing.md](contributing.md). Le code fonctionnel va sur `master`, le développement sur `dev`.

- **MIMIC-IV** : création de cohortes ✅ ; séries temporelles biologiques ; exemple jouet de mortalité à 28 jours après infarctus ✅ ; modalité ECG ; modalité waveforms ; extension à MIMIC-III (tabulaire).
- **Autres sources** : IMPROVE, registres d'infarctus, VitalDB.

## Résumé scientifique (abstract)

**Zarrow: a Modular Framework for Clinical Cohort Construction and Machine Learning Benchmarking on Multimodal medical datasets.**

*Introduction.* AI model development in clinical research is hindered by a lack of reproducibility, heterogeneous data formats, and a lack of standardized cohort construction pipelines. Public datasets like MIMIC provide a valuable resource. Yet, the conversion of the raw relational tables into analysis ready datasets results in a case by case from scratch error-prone and time consuming pipeline redevelopment. The extension of the datasets to integrate new modalities like waveforms, ECG, X-ray, or text notes is also non-trivial.

*Methods.* We present a modular Python framework to ease clinical cohort construction and machine learning benchmarking. The framework allows for flexible cohort definition through composable filtering operation. It allows scalable processing and downloading through chunk data loading, and efficient storage through Zarr format for large scale datasets. The framework allows for multiple data modality including static tabular data and time-series. A demonstration usecase is implemented on MIMIC-IV to construct a cohort of patients with Acute Myocardial Infarction based on ICD codes and selected laboratory measurements and sociodemographic features.

*Results.* The framework enables end-to-end cohort building, from raw-data downloading to machine learning ready datasets in a reproducible manner. As a proof of concept, we derive a cohort of patients from MIMIC-IV and define a 28-day mortality endpoint for an incident myocardial infarction based on admission and death timestamp. The resulting dataset integrates heterogeneous clinical variables and can be readily used for downstream machine learning or deep learning tasks.

*Conclusion.* The work provides a modular extensible framework to standardize clinical data processing and machine learning workflow. By reducing the implementation overhead, the framework eases faster and more reliable implementation of machine learning models in healthcare. Future work aims at extending the framework to multimodal integration and representation learning approaches.

## Licence

Voir [LICENSE](LICENSE).
