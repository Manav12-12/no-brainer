from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class DataSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame


def time_split(frame: pd.DataFrame, train_fraction: float = 0.6) -> DataSplit:
    if not frame["timestamp"].is_monotonic_increasing:
        raise ValueError("time split requires sorted timestamps")
    if not 0.0 < train_fraction < 0.8:
        raise ValueError("train_fraction must leave validation and test data")
    train_end = int(len(frame) * train_fraction)
    validation_end = int(len(frame) * 0.8)
    return DataSplit(
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def family_holdout_split(frame: pd.DataFrame, held_out: str) -> DataSplit:
    holdout = frame[frame["attack_family"] == held_out]
    remainder = frame[frame["attack_family"] != held_out]
    if holdout.empty:
        raise ValueError(f"held-out family is absent: {held_out}")
    cutoff = int(len(remainder) * 0.75)
    split = DataSplit(
        remainder.iloc[:cutoff].copy(),
        remainder.iloc[cutoff:].copy(),
        holdout.copy(),
    )
    assert_no_family_leakage(split, held_out)
    return split


def assert_no_family_leakage(split: DataSplit, held_out: str) -> None:
    if held_out in set(split.train["attack_family"]):
        raise AssertionError("held-out attack family leaked into training data")
    if set(split.test["attack_family"]) != {held_out}:
        raise AssertionError("test data must contain only the held-out family")
