'''
28-day mortality after an acute myocardial infarction, with and without the presenting ECG.

Same cohort, label and tabular features (demographics, first laboratory values within the window) as
scripts/mortality28d.py.
Adds simple features computed from the first 12-lead ECG of the index admission (MIMIC-IV-ECG, window
[-12 h, +24 h] around admission, see the 'modalities' section of config/mimic_iv_infarction.yaml):
heart rate, RR variability and, per lead, the QRS amplitude and the ST deviation of the median beat
(ST elevation/depression is the ECG signature of an infarction).
The model is a logistic regression, compared with and without the ECG features on the same 5 folds:
it trains in seconds on a laptop CPU.
'''
import argparse
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modules.features import first_in_window
from modules.multimodal import add_modality, decode_records, get_modalities_from_config
from modules.physionet_cohort import MIMICPatientCohort, get_schema_from_config
from modules.zarr_tools import ZarrLoader

# --- Build the cohort, or add the ECGs to a cohort built by scripts/mortality28d.py ---
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--config', default='config/mimic_iv_infarction.yaml')
parser.add_argument('--zarr', default=str(Path('data') / 'cohort.zarr'), help='cohort store, built if missing')
parser.add_argument('--tmp-dir', default=str(Path('data') / 'mimiciv-tmp'), help='downloaded MIMIC-IV tables')
parser.add_argument('--plot-subject', type=int, default=None,
                    help='subject_id of the ECG shown in figures/ecg_example.svg (default: first clean ECG)')
args = parser.parse_args()
config, zarr_path, tmp_dir = Path(args.config), args.zarr, args.tmp_dir
modalities = get_modalities_from_config(config)
if not os.path.exists(zarr_path):
    cohort = MIMICPatientCohort(tmp_dir, 'subject_id', schema=get_schema_from_config(config), modalities=modalities)
    cohort.build_and_save(zarr_path)
elif 'ecg' not in zarr.open_group(zarr_path, mode='r'):
    add_modality(zarr_path, 'subject_id', 'ecg', modalities['ecg'])

# --- Index admission, label and laboratory features (as in scripts/mortality28d.py) ---
loader = ZarrLoader(zarr_path)
df_adm = loader.load_group("admission", as_df=True)
df_demo = loader.load_group("demographics", as_df=True)
df_diag = loader.load_group("diagnoses", as_df=True)
df_social = loader.load_group("social", as_df=True)
df_labs = loader.load_group("labs", as_df=True)

mi_adm = df_diag[['subject_id', 'hadm_id']].drop_duplicates()
df = df_adm.merge(mi_adm, on=['subject_id', 'hadm_id'], how='inner')
df = df.merge(df_demo, on='subject_id', how='inner')
df = df.merge(df_social, on=['subject_id', 'hadm_id'], how='left')

df["admittime"] = pd.to_datetime(df["admittime"])
df["dod"] = pd.to_datetime(df["dod"], errors="coerce")
# dod is a date (midnight): count calendar days from the admission day, so a death on the day of admission is day 0
days_to_death = (df["dod"] - df["admittime"].dt.normalize()).dt.days
df["mortality_28d"] = (df["dod"].notna() & days_to_death.between(0, 28)).astype(int)
df = df.sort_values(by=['subject_id', 'admittime']).drop_duplicates('subject_id', keep='first')

# first value of each lab within the ECG window around the index admission (matched by subject and time)
window_h = modalities['ecg']['window']['hours']
df_labs["value_num"] = pd.to_numeric(df_labs["valuenum"], errors="coerce")
lab_first = first_in_window(df_labs, df[["subject_id"]].assign(anchor_time=df["admittime"]),
                            "subject_id", "charttime", "itemid", "value_num", window_h)
lab_first.columns = [f"lab_{int(itemid)}" for itemid in lab_first.columns]
lab_features = list(lab_first.columns)
df = df.merge(lab_first.reset_index(), on="subject_id", how="left")

# --- ECG features: first valid ECG matched to the index admission ---
def ecg_features(signal:np.ndarray, fs:int, leads:list[str])->dict:
    '''
    Simple, interpretable features of a 10 s 12-lead ECG (in mV), from its median beat.
    Beats are detected on the summed slope energy of all leads; fiducial points are approximate
    (fixed offsets from the detected beat, no wave delineation). Returns {} if fewer than 3 beats.
    '''
    signal = np.nan_to_num(signal - np.nanmedian(signal, axis=0))   # missing samples -> baseline
    b, a = butter(2, [0.5, 40], btype='band', fs=fs)
    x = filtfilt(b, a, signal, axis=0)
    energy = np.sum(np.diff(x, axis=0) ** 2, axis=1)
    # threshold from the median of the per-second maxima: robust to a single large artifact
    per_second_max = energy[:len(energy) // fs * fs].reshape(-1, fs).max(axis=1)
    beats, _ = find_peaks(energy, distance=int(0.25 * fs), height=0.3 * np.median(per_second_max))
    pre, post = int(0.25 * fs), int(0.40 * fs)
    beats = beats[(beats >= pre) & (beats < len(x) - post)]
    if len(beats) < 3:
        return {}
    rr = np.diff(beats) / fs
    median_beat = np.median(np.stack([x[i - pre:i + post] for i in beats]), axis=0)
    at = lambda ms: pre + int(ms * fs / 1000)            # sample of the median beat, ms from the beat
    baseline = median_beat[at(-100):at(-60)].mean(axis=0)               # PR segment
    st = median_beat[at(80):at(120)].mean(axis=0) - baseline            # ST segment, ~J+40 to J+80 ms
    qrs = median_beat[at(-50):at(60)]
    features = {"ecg_hr": 60 / np.median(rr), "ecg_rr_std_ms": 1000 * np.std(rr)}
    for i, lead in enumerate(leads):
        features[f"ecg_st_{lead}"] = st[i]
        features[f"ecg_qrs_amp_{lead}"] = qrs[:, i].max() - qrs[:, i].min()
    return features

ecg = loader.store["ecg"]
attrs = dict(ecg.attrs)
records = pd.DataFrame({"row": np.arange(ecg["valid"].shape[0]), "subject_id": ecg["record_subject"][:],
                        "hadm_id": ecg["record_hadm"][:], "record_time": ecg["record_time"][:],
                        "offset_h": ecg["record_offset_h"][:], "valid": ecg["valid"][:]})
first_ecg = (records[records["valid"]].merge(df[["subject_id", "hadm_id"]], on=["subject_id", "hadm_id"])
             .sort_values("record_time").drop_duplicates("subject_id").sort_values("row"))  # row order: sequential reads
print(f"Computing features of {len(first_ecg)} ECGs...")
ecg_rows = [{"subject_id": r.subject_id, "ecg_offset_h": r.offset_h,
             **ecg_features(decode_records(ecg["data"][r.row], attrs), attrs["fs"], attrs["leads"])}
            for r in first_ecg.itertuples()]
df_ecg = pd.DataFrame(ecg_rows)
ecg_features_cols = [c for c in df_ecg.columns if c.startswith("ecg_") and c != "ecg_offset_h"]
df = df.merge(df_ecg, on="subject_id", how="left")

# --- Figure: one presenting ECG as stored in the cohort, standard 3x4 + rhythm strip layout ---
def plot_ecg(signal:np.ndarray, fs:int, leads:list[str], path:Path, title:str):
    '''12-lead ECG (in mV) on ECG paper: 25 mm/s, 10 mm/mV, 2.5 s per lead column, lead II as rhythm strip.'''
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    signal = np.nan_to_num(signal - np.nanmedian(signal, axis=0))
    b, a = butter(2, [0.5, 40], btype='band', fs=fs)               # same display band as ecg_features
    x = filtfilt(b, a, signal, axis=0)
    layout = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
    seg, row_gap = int(2.5 * fs), 3.0                               # samples per column, mV between rows
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    for r, row in enumerate(layout + [["II"]]):
        y0 = -r * row_gap
        if r == 3:                                                  # rhythm strip: lead II, 10 s
            ax.plot(np.arange(len(x)) / fs, x[:, leads.index("II")] + y0, color="#222222", lw=0.7)
        for c, lead in enumerate(row if r < 3 else []):
            ax.plot(c * 2.5 + np.arange(seg) / fs, x[c * seg:(c + 1) * seg, leads.index(lead)] + y0,
                    color="#222222", lw=0.7)
            if c:
                ax.plot([c * 2.5] * 2, [y0 - 0.3, y0 + 0.3], color="#222222", lw=0.8)  # column separator
        for c, lead in enumerate(row):
            ax.text(c * 2.5 + 0.12, y0 + 0.75, lead, fontsize=9, weight="bold", color="#222222")
    ax.set_xticks(np.arange(0, 10.001, 0.2)); ax.set_xticks(np.arange(0, 10.001, 0.04), minor=True)
    ax.set_yticks(np.arange(-10.5, 1.501, 0.5)); ax.set_yticks(np.arange(-10.5, 1.501, 0.1), minor=True)
    ax.grid(which="major", color="#f0a8a8", lw=0.5); ax.grid(which="minor", color="#fbe1e1", lw=0.25)
    ax.tick_params(which="both", length=0, labelbottom=False, labelleft=False)
    ax.set(xlim=(0, 10), ylim=(-10.5, 1.5), aspect=0.4)            # 1 mV = 10 mm, 1 s = 25 mm
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10, loc="left")
    ax.text(10, -10.45, "25 mm/s · 10 mm/mV · 0.5–40 Hz", fontsize=7, ha="right", va="bottom", color="#666666")
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path.with_suffix(".svg")); fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)

if args.plot_subject is not None:
    plot_row = first_ecg[first_ecg["subject_id"] == args.plot_subject].iloc[0]
else:   # first ECG without missing samples and with a regular rate (illustration only, no clinical selection)
    clean = df_ecg[df_ecg["ecg_hr"].between(55, 95) & (df_ecg["ecg_rr_std_ms"] < 50)]["subject_id"]
    plot_row = next(r for r in first_ecg[first_ecg["subject_id"].isin(clean)].itertuples()
                    if not np.isnan(decode_records(ecg["data"][r.row], attrs)).any())
plot_ecg(decode_records(ecg["data"][plot_row.row], attrs), attrs["fs"], attrs["leads"],
         Path("figures") / "ecg_example",
         f"Presenting 12-lead ECG, {plot_row.offset_h:+.1f} h from admission (MIMIC-IV-ECG)")
print(f"ECG figure: figures/ecg_example.svg (subject {plot_row.subject_id}, record row {plot_row.row})")

has_ecg = df["ecg_offset_h"].notna()
print(f"\nCohort: {len(df)} patients (1st MI admission), 28-day mortality {df['mortality_28d'].mean():.1%}.")
print(f"ECG within [-12 h, +24 h] of admission: {has_ecg.sum()} patients ({has_ecg.mean():.1%}), "
      f"median {df.loc[has_ecg, 'ecg_offset_h'].median():.1f} h from admission; "
      f"beat detection failed on {df.loc[has_ecg, 'ecg_hr'].isna().sum()} of them.")
print(f"28-day mortality: {df.loc[has_ecg, 'mortality_28d'].mean():.1%} with an ECG, "
      f"{df.loc[~has_ecg, 'mortality_28d'].mean():.1%} without.")

# --- Logistic regression, with and without ECG features, same 5 folds ---
# missing labs/ECG -> median imputation + missingness indicator, then standardization
def make_model(numeric_features):
    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median', add_indicator=True)),
            ('scale', StandardScaler()),
        ]), numeric_features),
    ])
    return Pipeline([('pre', preprocessor), ('model', LogisticRegression(max_iter=2000))])

categorical_features = ['gender', 'marital_status']
# ablation: having an ECG in the window is itself linked to mortality (see above). The "ECG available" model
# only knows whether an ECG exists, so (ECG features - ECG available) is the gain carried by the signal itself.
df["ecg_available"] = has_ecg.astype(int)
feature_sets = {
    "demographics + labs": ['anchor_age'] + lab_features,
    "demographics + labs + ECG available": ['anchor_age'] + lab_features + ['ecg_available'],
    "demographics + labs + ECG": ['anchor_age'] + lab_features + ecg_features_cols,
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y = df['mortality_28d']
scores = {}
print(f"\n5-fold cross-validated performance ({len(df)} patients, mean ± std over folds):")
for name, numeric in feature_sets.items():
    s = cross_validate(make_model(numeric), df[categorical_features + numeric], y, cv=cv,
                       scoring=['roc_auc', 'average_precision'])
    scores[name] = s
    print(f"  {name:<37} AUC = {s['test_roc_auc'].mean():.3f} ± {s['test_roc_auc'].std():.3f}   "
          f"AUPRC = {s['test_average_precision'].mean():.3f} ± {s['test_average_precision'].std():.3f}")
for label, a, b in [("ECG (total)", "demographics + labs + ECG", "demographics + labs"),
                    ("ECG availability only", "demographics + labs + ECG available", "demographics + labs"),
                    ("ECG signal (beyond availability)", "demographics + labs + ECG", "demographics + labs + ECG available")]:
    delta = scores[a]['test_roc_auc'] - scores[b]['test_roc_auc']
    print(f"  AUC gain, {label:<33} (paired over folds): {delta.mean():+.3f} ± {delta.std():.3f}, "
          f"positive in {(delta > 0).sum()}/5 folds")

# --- Most predictive ECG features (model fitted on the whole cohort, standardized coefficients) ---
full = make_model(feature_sets["demographics + labs + ECG"]).fit(df[categorical_features + feature_sets["demographics + labs + ECG"]], y)
coefs = pd.Series(full.named_steps['model'].coef_[0], index=full.named_steps['pre'].get_feature_names_out())
coefs = coefs[coefs.index.str.contains("ecg_")]
print("\nTop 10 ECG features (logistic regression coefficients, standardized):")
print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(10).round(3).to_string())
