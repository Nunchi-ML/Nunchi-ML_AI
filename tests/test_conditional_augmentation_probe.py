import importlib
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def module(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/"notebooks/experiments/sun_visible_probe"))
    return importlib.import_module("conditional_augmentation_probe")


def one(identity,group,digest):
    return pd.DataFrame([dict(id=identity,group=group,image_sha256=digest)])


def test_clean_auxiliary_pool_keeps_both_sources(module):
    result=module.training_pool(one("a","g1","h1"),one("b","g2","h2"),one("c","g3","h3"))
    assert list(result.id)==["a","b"]


@pytest.mark.parametrize("aux",[one("a","g2","h2"),one("b","g2","h1"),
                                one("c","g2","h2"),one("b","g3","h2"),one("b","g2","h3")])
def test_auxiliary_identity_hash_and_capture_leakage_are_rejected(module,aux):
    with pytest.raises(ValueError,match="leakage"):
        module.training_pool(one("a","g1","h1"),aux,one("c","g3","h3"))


def test_existing_results_cannot_be_overwritten(module,tmp_path):
    with pytest.raises(ValueError,match="new private output"):
        module.run(tmp_path,tmp_path)
