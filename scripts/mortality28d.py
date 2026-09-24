'''
A dummy example of how to use the MIMICIVPatient cohort. 
We aim to predict the 28-day mortality of patients admitted to the hospital for an acute myocardial infarction
(ICD codes I21*, I22*, 410*, see config/mimic_iv_infarction.yaml) based on their demographics.
Lab values are extracted into the cohort (labs group) but not used by the model yet.
The obtained cohort is then used to develop and assess a machine learning model for mortality prediction.
'''
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from modules.physionet_cohort import MIMICPatientCohort,get_schema_from_config
import pandas as pd

# --- Build cohort with filters ---
from pathlib import Path

tmp_dir=str(Path('data') / 'mimiciv-tmp')
zarr_path=str(Path('data') / 'cohort.zarr')
if not os.path.exists(zarr_path):
    schema=get_schema_from_config(Path('config/mimic_iv_infarction.yaml'))
    cohort=MIMICPatientCohort(tmp_dir,'subject_id',schema=schema)
    cohort.build_and_save(zarr_path)

# --- Load cohort and prepare dataset for modeling ---
from modules.zarr_tools import ZarrLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.base import clone

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
# race simplified pour stratification et sous-groupes
df["race_simple"] = df["race"].astype(str).apply(lambda x: "WHITE" if "WHITE" in x.upper() else "NON_WHITE")

# --- Train/test split par patient, stratifié ---
# strata also include the (rare) outcome so that train and test keep the same event rate
df['strata'] = df["gender"].astype(str) + "_" + df["race_simple"] + "_" + df["mortality_28d"].astype(str)
patients = df[['subject_id', 'strata']].drop_duplicates()

train_patients, test_patients = train_test_split(
    patients['subject_id'],
    test_size=0.3,
    random_state=42,
    stratify=patients['strata']
)

df_train = df[df['subject_id'].isin(train_patients)].copy()
df_test = df[df['subject_id'].isin(test_patients)].copy()

# --- Features et target ---
features = ['anchor_age', 'gender', 'marital_status']  # race_simple exclue du modèle
target = 'mortality_28d'

X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]

# --- Pipeline avec ColumnTransformer ---
categorical_features = ['gender', 'marital_status']
numeric_features = ['anchor_age']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ('num', 'passthrough', numeric_features)
])

clf = Pipeline([
    ('pre', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

# --- Entraînement global ---
clf_global = clone(clf)
clf_global.fit(X_train, y_train)

y_pred_global = clf_global.predict_proba(X_test)[:,1]
auc_global = roc_auc_score(y_test, y_pred_global)
print("AUC global:", auc_global)

# --- Analyse par sous-groupes ---
groups = [("M", "WHITE"), ("M", "NON_WHITE"), ("F", "WHITE"), ("F", "NON_WHITE")]
results = []

for sex, race in groups:
    train_mask = (df_train["gender"] == sex) & (df_train["race_simple"] == race)
    test_mask = (df_test["gender"] == sex) & (df_test["race_simple"] == race)
    
    if train_mask.sum() < 30 or test_mask.sum() < 20:
        continue
    
    clf_sub = clone(clf)
    clf_sub.fit(X_train[train_mask], y_train[train_mask])
    
    y_pred_sub = clf_sub.predict_proba(X_test[test_mask])[:,1]
    auc_sub = roc_auc_score(y_test[test_mask], y_pred_sub)
    
    # AUC du modèle global sur le même sous-groupe
    y_pred_global_sub = y_pred_global[test_mask]
    auc_global_sub = roc_auc_score(y_test[test_mask], y_pred_global_sub)
    
    results.append({
        "sex": sex,
        "race": race,
        "auc_subgroup": auc_sub,
        "auc_global": auc_global_sub,
        "delta": auc_sub - auc_global_sub
    })

df_results = pd.DataFrame(results)
print(df_results)

print('dataset shapes:')
print(X_train.shape, y_train.value_counts())
print(X_test.shape, y_test.value_counts())