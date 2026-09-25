from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


NON_MODEL_COLUMNS={"patient_key","center","outcome_true","outer_fold","n_valid_seizures","raw_valid_seizure_count","feature_missing_fraction","raw_missing_fraction","p2_missing_fraction"}


def build_patient_table(feature:pd.DataFrame,raw:pd.DataFrame|None,p2:pd.DataFrame|None,outcomes:pd.DataFrame,folds:pd.DataFrame)->pd.DataFrame:
    table=outcomes[["patient_key","outcome_label"]].rename(columns={"outcome_label":"outcome_true"}).merge(folds[["patient_key","outer_fold"]],on="patient_key",validate="one_to_one").merge(feature,on="patient_key",how="left",validate="one_to_one",suffixes=("","_feature"))
    if "center_feature" in table:table["center"]=table.pop("center_feature")
    if raw is not None:
        value=raw.drop(columns=["center"],errors="ignore");table=table.merge(value,on="patient_key",how="left",validate="one_to_one")
    if p2 is not None:
        value=p2.drop(columns=["center","outer_fold"],errors="ignore");table=table.merge(value,on="patient_key",how="left",validate="one_to_one")
    groups={"feature":[c for c in table if c.startswith("feature__")],"raw":[c for c in table if c.startswith("raw__")],"p2":[c for c in table if c.startswith("p2__")]}
    for name,columns in groups.items():table[f"{name}_missing_fraction"]=table[columns].isna().mean(axis=1) if columns else 1.
    if table.patient_key.duplicated().any():raise ValueError("COP patient table contains duplicate patients")
    if table["center"].isna().any():raise ValueError("MISSING_REQUIRED_FIELD: COP patient center")
    return table.sort_values("patient_key").reset_index(drop=True)


def model_columns(table:pd.DataFrame,groups:Sequence[str])->list[str]:
    prefixes=tuple(f"{name}__" for name in groups);return [column for column in table if column.startswith(prefixes)]


__all__=["NON_MODEL_COLUMNS","build_patient_table","model_columns"]
