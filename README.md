# mimic-ingest

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
