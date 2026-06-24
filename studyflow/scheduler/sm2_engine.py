"""
SuperMemo-2 (SM-2) Spaced Repetition Engine for StudyFlow.

This module implements the SM-2 algorithm as originally published by
Piotr Woźniak (1990), enriched with Ebbinghaus forgetting-curve
retrievability estimation.  Every function is typed, documented with
LaTeX formulae, and unit-testable in isolation.

Mathematical Foundations
========================

Ebbinghaus Forgetting Curve
---------------------------

The probability that a memory is still retrievable after time *t* is
modelled by an exponential decay:

.. math::

    R(t) = e^{-\\frac{t}{S}}

where:

* :math:`R` — **retrievability** (probability of recall, :math:`R \\in [0, 1]`)
* :math:`t` — **time elapsed** since last review (days)
* :math:`S` — **stability** (the time constant; higher ⟹ slower forgetting)

Stability is proportional to the SM-2 interval:

.. math::

    S = I(n) \\cdot \\frac{-1}{\\ln(R_{\\text{target}})}

where :math:`R_{\\text{target}}` is the desired recall probability at the
end of each interval (typically 0.9).

SM-2 Algorithm
--------------

After each quiz attempt with quality score :math:`q \\in \\{0, 1, 2, 3, 4, 5\\}`:

1. **Easiness Factor update**:

   .. math::

       EF' = EF + \\bigl(0.1 - (5 - q) \\cdot (0.08 + (5 - q) \\cdot 0.02)\\bigr)

   clamped to :math:`EF' \\geq 1.3`.

2. **Repetition counter & interval**:

   If :math:`q \\geq 3` (correct recall):

   .. math::

       n' = n + 1, \\qquad
       I(n') = \\begin{cases}
           1       & \\text{if } n' = 1 \\\\
           6       & \\text{if } n' = 2 \\\\
           \\text{round}(I(n) \\cdot EF') & \\text{if } n' \\geq 3
       \\end{cases}

   If :math:`q < 3` (failed recall):

   .. math::

       n' = 0, \\qquad I(n') = 1

3. **Next review date**:

   .. math::

       d_{\\text{next}} = d_{\\text{today}} + I(n')
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# Data Transfer Object returned after each SM-2 update
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class SM2Result:
    """
    Immutable result of a single SM-2 update step.

    Attributes
    ----------
    easiness_factor : float
        Updated easiness factor (EF'), clamped ≥ 1.3.
    interval_days : int
        New inter-repetition interval *I(n')* in days.
    repetition_number : int
        Updated consecutive-correct-recall counter *n'*.
    next_review_date : date
        Calendar date for the next scheduled review.
    retrievability : float
        Current memory retrievability estimate *R(0)* = 1.0 right
        after a review.
    """

    easiness_factor: float
    interval_days: int
    repetition_number: int
    next_review_date: date
    retrievability: float = 1.0


# ---------------------------------------------------------------------------
# Core SM-2 Engine
# ---------------------------------------------------------------------------

class SM2Engine:
    """
    Stateless SM-2 calculator.

    All state lives in the database (:class:`~studyflow.models.TopicRecord`).
    This class is a pure-function wrapper: give it the current scheduling
    parameters and a quality score, and it returns the next parameters.

    Parameters
    ----------
    target_recall : float
        Desired recall probability at the end of each interval.
        Default is 0.9 (90 %).  Used only for the Ebbinghaus
        retrievability estimate—**not** for the core SM-2 logic.
    min_easiness_factor : float
        Floor for the easiness factor.  SM-2 specifies 1.3.

    Examples
    --------
    >>> engine = SM2Engine()
    >>> result = engine.update(
    ...     quality_score=4,
    ...     easiness_factor=2.5,
    ...     interval_days=0,
    ...     repetition_number=0,
    ...     review_date=date(2026, 6, 23),
    ... )
    >>> result.interval_days
    1
    >>> result.next_review_date
    datetime.date(2026, 6, 24)
    """

    # SM-2 magic constants (from the original 1990 paper)
    _EF_DELTA_BASE: float = 0.1
    _EF_DELTA_LINEAR: float = 0.08
    _EF_DELTA_QUADRATIC: float = 0.02
    _QUALITY_PASS_THRESHOLD: int = 3

    def __init__(
        self,
        target_recall: float = 0.9,
        min_easiness_factor: float = 1.3,
    ) -> None:
        if not (0.0 < target_recall < 1.0):
            raise ValueError("target_recall must be in (0, 1).")
        if min_easiness_factor < 1.0:
            raise ValueError("min_easiness_factor must be >= 1.0.")

        self._target_recall = target_recall
        self._min_ef = min_easiness_factor

    # ── Public API ────────────────────────────────────────────────────────

    def update(
        self,
        quality_score: int,
        easiness_factor: float,
        interval_days: int,
        repetition_number: int,
        review_date: Optional[date] = None,
    ) -> SM2Result:
        """
        Execute one SM-2 update step.

        Parameters
        ----------
        quality_score : int
            User performance on a 0–5 scale (SM-2 convention).
        easiness_factor : float
            Current EF before this review.
        interval_days : int
            Current inter-repetition interval *I(n)* in days.
        repetition_number : int
            Current consecutive-correct count *n*.
        review_date : date | None
            Date of the review.  Defaults to ``date.today()``.

        Returns
        -------
        SM2Result
            The full set of updated scheduling parameters.

        Raises
        ------
        ValueError
            If ``quality_score`` is outside [0, 5].

        Notes
        -----
        The easiness-factor update formula from the SM-2 paper:

        .. math::

            EF' = EF + \\bigl(0.1 - (5 - q)(0.08 + (5 - q) \\cdot 0.02)\\bigr)

        is applied **regardless** of whether the recall was correct.
        The repetition counter and interval are **reset** on failure
        (:math:`q < 3`).
        """
        if not (0 <= quality_score <= 5):
            raise ValueError(f"quality_score must be 0–5, got {quality_score}")

        if review_date is None:
            review_date = date.today()

        # Step 1: update easiness factor
        new_ef = self._update_easiness_factor(easiness_factor, quality_score)

        # Step 2: update repetition counter & interval
        if quality_score >= self._QUALITY_PASS_THRESHOLD:
            new_rep = repetition_number + 1
            new_interval = self._next_interval(new_rep, interval_days, new_ef)
        else:
            # Failed recall → restart
            new_rep = 0
            new_interval = 1

        # Step 3: compute next review date
        next_review = review_date + timedelta(days=new_interval)

        return SM2Result(
            easiness_factor=new_ef,
            interval_days=new_interval,
            repetition_number=new_rep,
            next_review_date=next_review,
            retrievability=1.0,  # Just reviewed
        )

    def retrievability(
        self,
        days_elapsed: float,
        stability: float,
    ) -> float:
        """
        Estimate memory retrievability via the Ebbinghaus forgetting curve.

        .. math::

            R(t) = e^{-\\frac{t}{S}}

        Parameters
        ----------
        days_elapsed : float
            Time *t* since last review, in days.
        stability : float
            Memory stability *S* (in days).  A higher value means
            slower forgetting.

        Returns
        -------
        float
            Retrievability in [0, 1].

        Raises
        ------
        ValueError
            If ``stability`` ≤ 0 or ``days_elapsed`` < 0.
        """
        if stability <= 0:
            raise ValueError(f"stability must be > 0, got {stability}")
        if days_elapsed < 0:
            raise ValueError(f"days_elapsed must be >= 0, got {days_elapsed}")

        return math.exp(-days_elapsed / stability)

    def stability_from_interval(self, interval_days: int) -> float:
        """
        Derive the stability constant *S* from a scheduled interval.

        We define *S* such that the retrievability at the end of the
        interval equals ``target_recall``:

        .. math::

            S = \\frac{-I(n)}{\\ln(R_{\\text{target}})}

        Parameters
        ----------
        interval_days : int
            The scheduled inter-repetition interval.

        Returns
        -------
        float
            Stability in days.
        """
        if interval_days <= 0:
            return 1.0  # Degenerate case: treat as 1 day
        # ln(target_recall) is negative, so negation makes S positive
        return -interval_days / math.log(self._target_recall)

    # ── Private helpers ───────────────────────────────────────────────────

    def _update_easiness_factor(self, ef: float, q: int) -> float:
        """
        Apply the SM-2 easiness-factor delta.

        .. math::

            EF' = \\max\\bigl(1.3,\\; EF + 0.1 - (5-q)(0.08 + (5-q) \\cdot 0.02)\\bigr)

        Parameters
        ----------
        ef : float
            Current easiness factor.
        q : int
            Quality score (0–5).

        Returns
        -------
        float
            Updated EF, clamped to ``min_easiness_factor``.
        """
        delta = (
            self._EF_DELTA_BASE
            - (5 - q) * (self._EF_DELTA_LINEAR + (5 - q) * self._EF_DELTA_QUADRATIC)
        )
        return max(self._min_ef, ef + delta)

    @staticmethod
    def _next_interval(
        repetition_number: int,
        prev_interval: int,
        easiness_factor: float,
    ) -> int:
        """
        Compute the next inter-repetition interval.

        .. math::

            I(n) = \\begin{cases}
                1                                & n = 1 \\\\
                6                                & n = 2 \\\\
                \\text{round}(I(n-1) \\cdot EF)  & n \\geq 3
            \\end{cases}

        Parameters
        ----------
        repetition_number : int
            The **new** repetition count *n'* (already incremented).
        prev_interval : int
            The previous interval *I(n-1)*.
        easiness_factor : float
            The (already updated) EF'.

        Returns
        -------
        int
            The new interval in days (minimum 1).
        """
        if repetition_number == 1:
            return 1
        elif repetition_number == 2:
            return 6
        else:
            return max(1, round(prev_interval * easiness_factor))


# ---------------------------------------------------------------------------
# Convenience: module-level singleton
# ---------------------------------------------------------------------------

_default_engine: SM2Engine = SM2Engine()


def sm2_update(
    quality_score: int,
    easiness_factor: float = 2.5,
    interval_days: int = 0,
    repetition_number: int = 0,
    review_date: Optional[date] = None,
) -> SM2Result:
    """
    Module-level shortcut that delegates to the default :class:`SM2Engine`.

    See :meth:`SM2Engine.update` for full documentation.
    """
    return _default_engine.update(
        quality_score=quality_score,
        easiness_factor=easiness_factor,
        interval_days=interval_days,
        repetition_number=repetition_number,
        review_date=review_date,
    )


def get_due_reviews(
    conn,
    target_date: Optional[date] = None,
):
    """
    High-level convenience: query the database for topics due for review.

    This is the function the Quiz Agent calls to surface what needs
    reviewing today.

    Parameters
    ----------
    conn : sqlite3.Connection
        An initialised StudyFlow database connection.
    target_date : date | None
        Defaults to ``date.today()``.

    Returns
    -------
    list[TopicRecord]
        Topics whose ``next_review_date ≤ target_date``.
    """
    from studyflow.db.crud import TopicRepository

    if target_date is None:
        target_date = date.today()

    repo = TopicRepository(conn)
    return repo.get_due_reviews(target_date)
