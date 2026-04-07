"""복합 스코어링 엔진"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from analysis.fundamental import FundamentalResult
from analysis.technical import TechnicalResult
from analysis.macro import MacroResult
from analysis.industry import IndustryResult
from analysis.qualitative import QualitativeResult
from analysis.risk import RiskResult

logger = logging.getLogger(__name__)

GRADE_STRONG_BUY = "STRONG BUY"
GRADE_BUY = "BUY"
GRADE_HOLD = "HOLD"
GRADE_SELL = "SELL"
GRADE_STRONG_SELL = "STRONG SELL"


@dataclass
class CompositeScore:
    ticker: str
    composite_score: float = 0.0
    grade: str = GRADE_HOLD
    # 모듈별 점수
    fundamental_score: float = 50.0
    technical_score: float = 50.0
    macro_score: float = 50.0
    industry_score: float = 50.0
    qualitative_score: float = 50.0
    risk_penalty: float = 0.0
    # 가중치
    weights: dict[str, float] = field(default_factory=dict)
    # 상세
    details: dict[str, Any] = field(default_factory=dict)


class ScoringEngine:
    """Composite Score = Σ(Module_Score_i × Weight_i) - Risk_Penalty"""

    def __init__(self, weights: dict[str, float] | None = None, mode: str = "default"):
        self.mode = mode
        self.weights = weights or self._default_weights(mode)

    def score(
        self,
        ticker: str,
        fundamental: FundamentalResult | None = None,
        technical: TechnicalResult | None = None,
        macro: MacroResult | None = None,
        industry: IndustryResult | None = None,
        qualitative: QualitativeResult | None = None,
        risk: RiskResult | None = None,
    ) -> CompositeScore:
        w = self.weights
        result = CompositeScore(ticker=ticker, weights=w)

        f_score = fundamental.total_score if fundamental else 50.0
        t_score = technical.total_score if technical else 50.0
        m_score = macro.score if macro else 50.0
        i_score = industry.score if industry else 50.0
        q_score = qualitative.total_score if qualitative else 50.0
        r_penalty = (risk.risk_score * w.get("risk_penalty", 0.05)) if risk else 0.0

        composite = (
            f_score * w.get("fundamental", 0.35)
            + t_score * w.get("technical", 0.20)
            + m_score * w.get("macro", 0.15)
            + i_score * w.get("industry", 0.15)
            + q_score * w.get("qualitative", 0.10)
            - r_penalty
        )
        composite = max(0.0, min(100.0, composite))

        result.composite_score = round(composite, 2)
        result.fundamental_score = round(f_score, 2)
        result.technical_score = round(t_score, 2)
        result.macro_score = round(m_score, 2)
        result.industry_score = round(i_score, 2)
        result.qualitative_score = round(q_score, 2)
        result.risk_penalty = round(r_penalty, 2)
        result.grade = self._grade(composite)

        result.details = {
            "mode": self.mode,
            "score_breakdown": {
                "fundamental": round(f_score * w.get("fundamental", 0.35), 2),
                "technical": round(t_score * w.get("technical", 0.20), 2),
                "macro": round(m_score * w.get("macro", 0.15), 2),
                "industry": round(i_score * w.get("industry", 0.15), 2),
                "qualitative": round(q_score * w.get("qualitative", 0.10), 2),
                "risk_penalty": round(r_penalty, 2),
            },
            "technical_signal": technical.short_term_signal if technical else "N/A",
            "technical_confidence": technical.short_confidence if technical else 0.5,
        }
        return result

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 90:
            return GRADE_STRONG_BUY
        elif score >= 75:
            return GRADE_BUY
        elif score >= 55:
            return GRADE_HOLD
        elif score >= 40:
            return GRADE_SELL
        else:
            return GRADE_STRONG_SELL

    @staticmethod
    def _default_weights(mode: str) -> dict[str, float]:
        presets = {
            "default":  {"fundamental": 0.35, "technical": 0.20, "macro": 0.15, "industry": 0.15, "qualitative": 0.10, "risk_penalty": 0.05},
            "growth":   {"fundamental": 0.40, "technical": 0.15, "macro": 0.15, "industry": 0.20, "qualitative": 0.10, "risk_penalty": 0.05},
            "value":    {"fundamental": 0.45, "technical": 0.10, "macro": 0.15, "industry": 0.20, "qualitative": 0.08, "risk_penalty": 0.02},
            "dividend": {"fundamental": 0.45, "technical": 0.10, "macro": 0.15, "industry": 0.20, "qualitative": 0.08, "risk_penalty": 0.02},
            "trading":  {"fundamental": 0.10, "technical": 0.45, "macro": 0.15, "industry": 0.10, "qualitative": 0.05, "risk_penalty": 0.15},
        }
        return presets.get(mode, presets["default"])
