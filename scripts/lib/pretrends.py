"""
Pre-trends evaluation criteria.

Determines whether a specification's pre-event coefficients are consistent
with the parallel trends assumption.
"""

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy import stats as scipy_stats


@dataclass
class PreTrendsResult:
    """Results of pre-trends evaluation for one specification."""
    n_pre_significant_10: int
    n_pre_significant_05: int
    max_pre_abs_t: float
    avg_pre_abs_coef: float
    avg_post_abs_coef: float
    joint_f_stat: float
    joint_f_pval: float
    passes_strict: bool
    passes_moderate: bool
    pre_coefs: List[float] = field(default_factory=list)
    pre_ses: List[float] = field(default_factory=list)
    pre_ts: List[float] = field(default_factory=list)
    pre_ps: List[float] = field(default_factory=list)
    post_coefs: List[float] = field(default_factory=list)
    post_ts: List[float] = field(default_factory=list)


def evaluate_pretrends(
    pre_event_results: Dict[int, dict],
    post_event_results: Dict[int, dict],
) -> PreTrendsResult:
    """
    Evaluate whether pre-trends pass parallel trends criteria.

    Strict criteria (all must hold):
      1. At most 1 pre-event coef significant at p<0.10
      2. No pre-event |t| > 2.0
      3. Joint F-test p >= 0.10
      4. Average |pre-coef| < average |post-coef|

    Moderate criteria (all must hold):
      1. At most 2 pre-event coefs significant at p<0.10
      2. No pre-event |t| > 2.5
      3. Joint F-test p >= 0.05
    """
    if not pre_event_results:
        return PreTrendsResult(
            n_pre_significant_10=0, n_pre_significant_05=0,
            max_pre_abs_t=0, avg_pre_abs_coef=0, avg_post_abs_coef=0,
            joint_f_stat=0, joint_f_pval=1.0,
            passes_strict=True, passes_moderate=True,
        )

    pre_coefs = [v['coef'] for v in pre_event_results.values()]
    pre_ses = [v['se'] for v in pre_event_results.values()]
    pre_ts = [v['t'] for v in pre_event_results.values()]
    pre_ps = [v['p'] for v in pre_event_results.values()]
    post_coefs = [v['coef'] for v in post_event_results.values()]
    post_ts = [v['t'] for v in post_event_results.values()]

    n_sig_10 = sum(1 for p in pre_ps if p < 0.10)
    n_sig_05 = sum(1 for p in pre_ps if p < 0.05)
    max_abs_t = max(abs(t) for t in pre_ts) if pre_ts else 0
    avg_pre = np.mean([abs(c) for c in pre_coefs]) if pre_coefs else 0
    avg_post = np.mean([abs(c) for c in post_coefs]) if post_coefs else 0

    # Joint F-test: sum of squared t-stats / k ~ F(k, large)
    k = len(pre_coefs)
    if k > 0 and all(se > 0 for se in pre_ses):
        chi2 = sum(t ** 2 for t in pre_ts)
        f_stat = chi2 / k
        f_pval = 1 - scipy_stats.f.cdf(f_stat, k, 2000)
    else:
        f_stat, f_pval = 0.0, 1.0

    passes_strict = (
        n_sig_10 <= 1
        and max_abs_t <= 2.0
        and f_pval >= 0.10
        and (avg_pre < avg_post or avg_post == 0)
    )

    passes_moderate = (
        n_sig_10 <= 2
        and max_abs_t <= 2.5
        and f_pval >= 0.05
    )

    return PreTrendsResult(
        n_pre_significant_10=n_sig_10,
        n_pre_significant_05=n_sig_05,
        max_pre_abs_t=max_abs_t,
        avg_pre_abs_coef=avg_pre,
        avg_post_abs_coef=avg_post,
        joint_f_stat=f_stat,
        joint_f_pval=f_pval,
        passes_strict=passes_strict,
        passes_moderate=passes_moderate,
        pre_coefs=pre_coefs,
        pre_ses=pre_ses,
        pre_ts=pre_ts,
        pre_ps=pre_ps,
        post_coefs=post_coefs,
        post_ts=post_ts,
    )
