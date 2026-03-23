# Zarrow

Tools to ingest and build a multimodal dataset from mimic-iv

Johnson, A.E.W., Bulgarelli, L., Shen, L. et al. MIMIC-IV, a freely accessible electronic health record dataset. Sci Data 10, 1 (2023). https://doi.org/10.1038/s41597-022-01899-x

# Build a zarr store

```

            download the csv from the clinical database
                                |
                                v
                        build a cohort
                                |
                                v
                        download the diverse modalities
                        from each cohort
                                |
                                v
                        store into zarr filesystem

```

# Demo

## Zarrow: a Modular Framework for Clinical Cohort Construction and Machine Learning Benchmarking on Multimodal medical datasets.
_Introduction_
AI model development in clinical research is hindered by a lack of reproducibility, heterogeneous data formats, and a lack of standardized cohort construction pipelines. Public datasets like MIMIC provide a valuable resource. Yet, the conversion of the raw relational tables into analysis ready datasets results in a case by case from scratch error-prone and time consuming pipeline redevelopment. The extension of the datasets to integrate new modalities like waveforms, ECG, X-ray, or text notes is also non-trivial.

_Methods_
We present a modular Python framework to ease clinical cohort construction and machine learning benchmarking. The framework allows for flexible cohort definition through composable filtering operation. It allows scalable processing and downloading through chunk data loading, and efficient storage through Zarr format for large scale datasets. The framework allows for multiple data modality including static tabular data and time-series. A demonstration usecase is implemented on MIMIC-IV to construct a cohort of patients with Acute Myocardial Infarction based on ICD codes and selected laboratory measurements and sociodemographic features.

_Results_
The framework enables end-to-end cohort building, from raw-data downloading to machine learning ready datasets in a reproducible manner. As a proof of concept, we derive a cohort of patients from MIMIC-IV and define a 28-day mortality endpoint for an incident myocardial infarction based on admission and death timestamp. The resulting dataset integrates heterogeneous clinical variables and can be readily used for downstream machine learning or deep learning tasks.

_Conclusion_
The work provides a modular extensible framework to standardize clinical data processing and machine learning workflow. By reducing the implementation overhead, the framework eases faster and more reliable implementation of machine learning models in healthcare. Future work aims at extending the framework to multimodal integration and representation learning approaches.

 