"""PolicyInduction's weight fit across scikit-learn versions (offline)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from think_reason_learn.core.llms import LLMChoice, OpenAIChoice
from think_reason_learn.policy_induction import PolicyInduction

pytestmark = pytest.mark.usefixtures("typesafe_key")

CHAT: list[LLMChoice] = [OpenAIChoice(model="gpt-4.1-nano")]


def _fitted(tmp_path: Path, penalty: str, **config: Any) -> PolicyInduction:
    """Fit weights on one informative policy and five noise policies."""
    rng = np.random.default_rng(0)
    n = 120
    y = np.where(np.arange(n) % 2 == 0, "YES", "NO")
    informative = np.where(rng.random(n) < 0.9, y == "YES", y != "YES") * 0.8 + 0.1
    columns: list[Any] = [informative] + [rng.random(n) for _ in range(5)]

    pi = PolicyInduction(
        gen_llmc=CHAT,
        config={"penalty": penalty, "Cs": (0.1,), "cv_folds": 3, **config},
        save_path=tmp_path,
        name="weights",
        confirm_requests=False,
    )
    pi._set_data(pd.DataFrame({"text": [f"sample {i}" for i in range(n)]}), list(y))
    pi._policies = [f"policy {j}" for j in range(len(columns))]
    pi._scores = np.column_stack(columns)
    pi._fit_weights()
    return pi


def test_penalty_kwargs_follow_the_sklearn_version() -> None:
    from think_reason_learn.policy_induction._policy_induction import (
        _penalty_kwargs,
    )

    assert _penalty_kwargs("l1", "1.7.2") == {"penalty": "l1"}
    assert _penalty_kwargs("l2", "1.7.2") == {"penalty": "l2"}
    assert _penalty_kwargs("l1", "1.8.0") == {"l1_ratio": 1.0}
    assert _penalty_kwargs("l2", "1.9.1") == {"l1_ratio": 0.0}
    assert _penalty_kwargs("l1", "1.10.0rc1") == {"l1_ratio": 1.0}


@pytest.mark.parametrize("version", ["1.7.2", "1.9.1"])
@pytest.mark.parametrize("penalty", ["L1", "elasticnet", None])
def test_an_unknown_penalty_is_refused(penalty: Any, version: str) -> None:
    from think_reason_learn.policy_induction._policy_induction import (
        _penalty_kwargs,
    )

    with pytest.raises(ValueError, match="penalty must be 'l1' or 'l2'"):
        _penalty_kwargs(penalty, version)


def test_l1_zeroes_noise_policies_and_l2_does_not(tmp_path: Path) -> None:
    l1 = _fitted(tmp_path, "l1").lr.coef_[0]
    l2 = _fitted(tmp_path, "l2").lr.coef_[0]

    assert l1[0] != 0
    assert (l1[1:] == 0).any()
    assert (l2 != 0).all()


def test_weight_fit_raises_no_sklearn_deprecation(tmp_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning, module="sklearn")
        warnings.filterwarnings("error", category=UserWarning, module="sklearn")
        _fitted(tmp_path, "l1")
        _fitted(tmp_path, "l2")


def test_one_se_rule_picks_a_stronger_c_than_the_best_scoring_one(
    tmp_path: Path,
) -> None:
    grid = {"Cs": (0.01, 0.03, 0.1, 0.3, 1.0, 10.0)}
    best = _fitted(tmp_path / "best", "l1", one_se_rule=False, **grid)
    one_se = _fitted(tmp_path / "se", "l1", one_se_rule=True, **grid)

    assert one_se.validation_result["best_C"] <= best.validation_result["best_C"]
    assert np.count_nonzero(one_se.lr.coef_[0]) <= np.count_nonzero(best.lr.coef_[0])


def test_a_c_that_zeroes_every_policy_is_never_chosen(tmp_path: Path) -> None:
    pi = _fitted(tmp_path, "l1", Cs=(1e-6, 0.3, 1.0))

    assert pi.validation_result["best_C"] > 1e-6
    assert np.count_nonzero(pi.lr.coef_[0]) > 0


def test_a_fit_where_every_policy_has_zero_weight_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="zero weight"):
        _fitted(tmp_path, "l1", Cs=(1e-6,))


def test_empty_cs_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one candidate"):
        _fitted(tmp_path, "l1", Cs=())


def test_a_class_with_one_row_cannot_be_cross_validated(tmp_path: Path) -> None:
    pi = _fitted(tmp_path / "ok", "l1")
    pi._y = np.array(["YES"] + ["NO"] * (len(pi._y) - 1))  # type: ignore[arg-type]

    from think_reason_learn.core.exceptions import DataError

    with pytest.raises(DataError, match="at least 2 rows"):
        pi._fit_weights()
