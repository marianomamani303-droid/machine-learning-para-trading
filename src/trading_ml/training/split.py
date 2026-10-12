from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_ml.config.settings import Training
from trading_ml.errors import DataError


@dataclass
class TemporalSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    purged_rows: int


def chronological_split(frame: pd.DataFrame, config: Training) -> TemporalSplit:
    """Purga etiquetas que alcanzan el siguiente período, incluso con h>1."""
    labelled = frame.dropna(subset=["target", "target_time"]).copy()
    n = len(labelled)
    train_end = int(n * config.train_fraction)
    validation_end = int(n * (config.train_fraction + config.validation_fraction))
    if not 0 < train_end < validation_end < n:
        raise DataError("Datos insuficientes para tres particiones")
    train = labelled.iloc[:train_end]
    validation = labelled.iloc[train_end:validation_end]
    test = labelled.iloc[validation_end:]
    train = train.loc[train.target_time < validation.reference_time.iloc[0]]
    validation = validation.loc[validation.target_time < test.reference_time.iloc[0]]
    if min(map(len, [train, validation, test])) < config.min_partition_rows:
        raise DataError("Partición demasiado pequeña después de la purga")
    if train.target.nunique() < 2:
        raise DataError("El entrenamiento debe contener las dos clases")
    return TemporalSplit(train, validation, test, n - len(train) - len(validation) - len(test))
