# Implementation audit

- New branch and SHA-256 protocol lock were pushed before new target outcomes. Exact A1 150-checkpoint grid and R4 classifier replay were rechecked; B=0 ranking/decision identity was checked for every target cell.
- Fixed 50/50 candidate/query split and prior 64D UNCERTAINTY support trajectory were reused for all B8 readouts. This isolates readout estimation but does **not** test new acquisition policies.
- FIT-only standardized R4, FIT-only PCA, and FIT-patient-disjoint projection-optimizer train/validation. Stage-1 global lambda selection uses all FIT patients, including the later meta-validation patients, so that meta-validation is not independent of prior lambda tuning. Lambda/gamma/dimension selection never uses target query labels.
- All 20 repetitions and all variant query scores were frozen in a target cell before any query label or target oracle was read. Oracle-balanced B8 and FULL_POOL are nondeployable controls.
- Original current B8 and FULL_POOL AP were replayed against private prior per-repetition records to <1e-8.
- Patient-ID cluster bootstrap used 10,000 seed-42 draws; 20 repetitions were not treated as independent patients. Variant winner and threshold gates are retrospective on the same 65 target cells, with no family-wise multiplicity control; independent confirmation remains necessary. The 37 FIT selection/model files were hash-frozen before the first target evaluation.
- Legacy loader materializes all 80 labels. No outer predictions, metrics, or selection were computed, but strict no-outer-label-materialization is false.
