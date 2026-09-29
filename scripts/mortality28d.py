'''
A dummy example of how to use the MIMICIVPatient cohort.
We aim to predict the 28-day mortality of patients admitted to the hospital for an acute myocardial infarction
(ICD codes I21*, I22*, 410*, see config/mimic_iv_infarction.yaml) based on their demographics and laboratory values.
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.base import clone

loader=ZarrLoader(zarr_path)
df_adm = pd.DataFrame(loader.load_group("admission"))
df_demo = pd.DataFrame(loader.load_group("demographics"))
df_diag = pd.DataFrame(loader.load_group("diagnoses"))
df_social = pd.DataFrame(loader.load_group("social"))
df_labs = pd.DataFrame(loader.load_group("labs"))

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
# race simplified for stratification and subgroup analysis
df["race_simple"] = df["race"].astype(str).apply(lambda x: "WHITE" if "WHITE" in x.upper() else "NON_WHITE")

# --- Laboratory features: mean value per lab item, for the indexed (first MI) admission ---
# LabEventFilter in the config already restricts df_labs to the monitored itemids.
# Use 'valuenum' (MIMIC-IV's pre-parsed numeric field), not 'value': the latter is free text
# and is largely non-numeric/masked ("___") for several of these itemids (NTproBNP...).
df_labs["value_num"] = pd.to_numeric(df_labs["valuenum"], errors="coerce")
lab_agg = (
    df_labs.groupby(["subject_id", "hadm_id", "itemid"])["value_num"]
    .mean()
    .unstack("itemid")
)
lab_agg.columns = [f"lab_{int(itemid)}" for itemid in lab_agg.columns]
lab_features = list(lab_agg.columns)
df = df.merge(lab_agg.reset_index(), on=["subject_id", "hadm_id"], how="left")

# --- Train/test split by patient, stratified ---
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

# --- Features and target ---
categorical_features = ['gender', 'marital_status']
numeric_features = ['anchor_age'] + lab_features       # race_simple excluded from the model
features = categorical_features + numeric_features
target = 'mortality_28d'

X_train = df_train[features]
y_train = df_train[target]
X_test = df_test[features]
y_test = df_test[target]

# --- Pipeline with ColumnTransformer ---
# labs: frequently missing (test not ordered) -> median imputation + missingness indicator,
# then standardization (useful for LogisticRegression's default L2 regularization)
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
    ('num', Pipeline([
        ('impute', SimpleImputer(strategy='median', add_indicator=True)),
        ('scale', StandardScaler()),
    ]), numeric_features),
])

clf = Pipeline([
    ('pre', preprocessor),
    ('model', LogisticRegression(max_iter=1000))
])

# --- 5-fold cross-validation (whole cohort, one row per patient) ---
# More robust than a single 70/30 split: reports mean +/- std over folds instead of one number.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
X_all, y_all = df[features], df[target]
cv_scores = cross_validate(clf, X_all, y_all, cv=cv, scoring=['roc_auc', 'average_precision'])
auc_cv_mean, auc_cv_std = cv_scores['test_roc_auc'].mean(), cv_scores['test_roc_auc'].std()
ap_cv_mean, ap_cv_std = cv_scores['test_average_precision'].mean(), cv_scores['test_average_precision'].std()
print(f"5-fold cross-validated performance ({len(df)} patients, mean ± std over folds):")
print(f"  AUC = {auc_cv_mean:.3f} ± {auc_cv_std:.3f}   AUPRC = {ap_cv_mean:.3f} ± {ap_cv_std:.3f}")

# --- Held-out 70/30 split (below): for the subgroup/interpretability analysis ---
clf_global = clone(clf)
clf_global.fit(X_train, y_train)

y_pred_global = clf_global.predict_proba(X_test)[:, 1]
auc_global = roc_auc_score(y_test, y_pred_global)
auprc_global = average_precision_score(y_test, y_pred_global)
brier_global = brier_score_loss(y_test, y_pred_global)

lab_coverage = df[lab_features].notna().mean().mul(100).round(1)
print(f"\nCohort: {len(df)} patients (1st MI admission), {len(lab_features)} laboratory tests aggregated.")
print(f"Lab coverage on the index admission (% of patients with a value):\n{lab_coverage.to_string()}")
print(f"\nHeld-out 30% test set: AUC = {auc_global:.4f}  |  AUPRC = {auprc_global:.4f}  |  Brier score = {brier_global:.4f}")

# --- Most predictive factors (global model coefficients, on standardized variables) ---
feature_names = clf_global.named_steps['pre'].get_feature_names_out()
coefs = pd.Series(clf_global.named_steps['model'].coef_[0], index=feature_names)
top_features = coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10)
print("\nTop 10 factors associated with 28-day mortality (logistic regression coefficients):")
print(top_features.round(3).to_string())

# --- Subgroup analysis ---
groups = [("M", "WHITE"), ("M", "NON_WHITE"), ("F", "WHITE"), ("F", "NON_WHITE")]
results = []

for sex, race in groups:
    train_mask = (df_train["gender"] == sex) & (df_train["race_simple"] == race)
    test_mask = (df_test["gender"] == sex) & (df_test["race_simple"] == race)

    if train_mask.sum() < 30 or test_mask.sum() < 20:
        continue

    clf_sub = clone(clf)
    clf_sub.fit(X_train[train_mask], y_train[train_mask])

    y_pred_sub = clf_sub.predict_proba(X_test[test_mask])[:, 1]
    auc_sub = roc_auc_score(y_test[test_mask], y_pred_sub)

    # AUC of the global model on the same subgroup
    y_pred_global_sub = y_pred_global[test_mask]
    auc_global_sub = roc_auc_score(y_test[test_mask], y_pred_global_sub)

    results.append({
        "sex": sex,
        "race": race,
        "n_test": int(test_mask.sum()),
        "auc_subgroup": auc_sub,
        "auc_global": auc_global_sub,
        "delta": auc_sub - auc_global_sub
    })

df_results = pd.DataFrame(results).round(4)
print("\nAUC by subgroup (model retrained on the subgroup vs. global model evaluated on the subgroup):")
print(df_results.to_string(index=False))

print('\nDataset sizes:')
print(f"train: {X_train.shape}, events: {y_train.value_counts().to_dict()}")
print(f"test : {X_test.shape}, events: {y_test.value_counts().to_dict()}")
