"""Self-calibrating completion estimates (scribe#8).

The estimator in eta.py is a per-stage linear model in things known at ack time. Its
constants were measured once, by hand, on one host, and were ~50% optimistic on real
jobs. Rather than refitting by hand, each stage carries a learned CORRECTION FACTOR:
an exponentially weighted mean of log(actual / predicted), updated when the stage
completes cleanly. The env constants stay as the priors a fresh spool starts from;
after a handful of jobs the data dominates.

Log space, because ratios are what drift (a slower host is 1.4x on everything) and
because it keeps a single pathological job from swinging the mean by minutes. The
spread is tracked too, and quotes use the upper side of it: an ack that lands early is
a pleasant surprise, one that lands late is the complaint this exists to fix.

Learned state lives in the `calibration` section of the runtime config file on the
PVC, next to the settings, written the same atomic way. Nothing here needs Loki or a
scheduled refit; the log lines are the audit trail for checking the learner behaves.
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

from scribe.config import Settings
from scribe.runtime_config import load_section, save_section

log = logging.getLogger("scribe.calibration")

SECTION = "calibration"
STAGES = ("ocr", "summarize_single", "summarize_map", "audio")

# Smoothing. The first few observations move the mean fast (1/n), then it settles to an
# EWMA with this alpha -- an effective memory of roughly 1/alpha jobs, so a rehomed
# ollama or a leaking Kokoro is tracked within a couple of dozen documents.
ALPHA = 0.1
# One observation can move a stage by at most e^2.5 ~ 12x; a factor can never leave
# [1/4, 8x] of the prior. Both guard the next ten estimates from one absurd job.
MAX_ABS_LOG_RATIO = 2.5
FACTOR_MIN, FACTOR_MAX = 0.25, 8.0
# Quote the mean plus this many spreads. 1.0 puts roughly 5 in 6 jobs at or before
# the quoted time, assuming a symmetric residual; the log lines tell whether that holds.
SPREAD_K = 1.0


@dataclass
class StageStats:
    log_factor: float = 0.0   # EWMA of log(actual/predicted)
    spread: float = 0.0       # EWMA of |residual| around log_factor
    n: int = 0
    updated: str | None = None

    @property
    def factor(self) -> float:
        return min(FACTOR_MAX, max(FACTOR_MIN, math.exp(self.log_factor)))

    @property
    def upper(self) -> float:
        """Factor for QUOTING: mean plus SPREAD_K spreads, still clamped."""
        return min(FACTOR_MAX, max(FACTOR_MIN, math.exp(self.log_factor + SPREAD_K * self.spread)))


class Calibration:
    def __init__(self, stats: dict[str, StageStats] | None = None):
        self.stats = {s: StageStats() for s in STAGES}
        if stats:
            self.stats.update({k: v for k, v in stats.items() if k in STAGES})

    @classmethod
    def load(cls, settings: Settings) -> Calibration:
        raw = load_section(settings, SECTION)
        stats: dict[str, StageStats] = {}
        for stage, v in raw.items():
            if stage in STAGES and isinstance(v, dict):
                try:
                    keys = ("log_factor", "spread", "n", "updated")
                    stats[stage] = StageStats(**{k: v[k] for k in keys if k in v})
                except (TypeError, ValueError):
                    continue
        return cls(stats)

    def save(self, settings: Settings) -> None:
        save_section(settings, SECTION, {s: asdict(v) for s, v in self.stats.items()})

    def quote(self, stage: str, predicted: float) -> float:
        """Predicted seconds scaled for quoting (mean + spread)."""
        return predicted * self.stats[stage].upper

    def expected(self, stage: str, predicted: float) -> float:
        """Predicted seconds scaled by the mean factor (for queue accounting)."""
        return predicted * self.stats[stage].factor

    def observe(self, settings: Settings, stage: str, predicted: float,
                actual: float) -> StageStats | None:
        """Fold one clean completion into `stage`. Returns the updated stats, or None if
        the observation was unusable (a zero prediction or duration teaches nothing)."""
        if stage not in STAGES or predicted <= 0 or actual <= 0:
            return None
        st = self.stats[stage]
        r = max(-MAX_ABS_LOG_RATIO, min(MAX_ABS_LOG_RATIO, math.log(actual / predicted)))
        alpha = max(ALPHA, 1.0 / (st.n + 1))
        before = st.factor
        resid = r - st.log_factor
        st.log_factor += alpha * resid
        # Spread is scatter AROUND the mean; a first observation defines the mean and
        # has no scatter, so it must not seed the spread with its whole residual (that
        # made one 1.5x job quote 2.25x).
        if st.n > 0:
            st.spread += alpha * (abs(resid) - st.spread)
        st.n += 1
        st.updated = datetime.now(UTC).isoformat(timespec="seconds")
        self.save(settings)
        log.info(
            "eta.learn stage=%s predicted=%.0f actual=%.0f ratio=%.2f factor_before=%.2f "
            "factor_after=%.2f spread=%.2f n=%d",
            stage, predicted, actual, actual / predicted, before, st.factor, st.spread, st.n,
        )
        return st

    def describe(self) -> str:
        parts = []
        for s in STAGES:
            st = self.stats[s]
            parts.append(f"{s} x{st.factor:.2f} (+{st.spread:.2f}, n={st.n})")
        return " · ".join(parts)
