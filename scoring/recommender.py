"""투자 추천 생성"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .engine import CompositeScore

logger = logging.getLogger(__name__)


@dataclass
class Recommendation:
    ticker: str
    grade: str
    composite_score: float
    # 목표주가 범위
    target_price: float | None = None
    target_low: float | None = None
    target_high: float | None = None
    # 진입 전략
    entry_price: float | None = None
    stop_loss: float | None = None
    suggested_weight: float | None = None  # 포트폴리오 비중 %
    # 핵심 투자 포인트
    key_points: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


class Recommender:
    """복합 점수 기반 투자 추천 생성"""

    def recommend(
        self,
        composite: CompositeScore,
        current_price: float | None = None,
        metrics: dict[str, Any] | None = None,
        risk_details: dict[str, Any] | None = None,
    ) -> Recommendation:
        m = metrics or {}
        rec = Recommendation(
            ticker=composite.ticker,
            grade=composite.grade,
            composite_score=composite.composite_score,
        )

        # 목표주가 산출
        if current_price:
            analyst_target = m.get("analyst_target")
            dcf = m.get("dcf_value")
            scenarios = (risk_details or {}).get("scenarios", {})

            targets = []
            if analyst_target:
                targets.append(float(analyst_target))
            if dcf:
                targets.append(float(dcf))
            if scenarios.get("base"):
                targets.append(float(scenarios["base"]))

            if targets:
                rec.target_price = round(sum(targets) / len(targets), 0)
                rec.target_low = round(rec.target_price * 0.85, 0)
                rec.target_high = round(rec.target_price * 1.15, 0)

            # 진입가 / 손절가
            rec.entry_price = round(current_price * 0.98, 0)
            atr = m.get("atr")
            if atr:
                rec.stop_loss = round(current_price - atr * 2, 0)
            else:
                rec.stop_loss = round(current_price * 0.92, 0)

        # 포트폴리오 비중 제안
        rec.suggested_weight = self._suggest_weight(composite.grade)

        # 핵심 투자 포인트
        rec.key_points = self._key_points(composite, m)
        rec.risks = (risk_details or {}).get("top_risks", [])[:5]

        return rec

    @staticmethod
    def _suggest_weight(grade: str) -> float:
        mapping = {
            "STRONG BUY": 8.0,
            "BUY": 5.0,
            "HOLD": 3.0,
            "SELL": 1.0,
            "STRONG SELL": 0.0,
        }
        return mapping.get(grade, 3.0)

    @staticmethod
    def _key_points(composite: CompositeScore, m: dict[str, Any]) -> list[str]:
        points = []
        # 기본적 분석
        if composite.fundamental_score >= 70:
            points.append(f"견고한 펀더멘털 (기본적 점수 {composite.fundamental_score:.0f}/100)")
        roe = m.get("roe")
        if roe and roe > 0.15:
            points.append(f"높은 ROE {roe:.1%} — 효율적 자본 운용")
        rev_growth = m.get("revenue_growth") or m.get("revenue_yoy")
        if rev_growth and rev_growth > 0.1:
            points.append(f"강한 매출 성장 ({rev_growth:.1%} YoY)")
        # 기술적 분석
        if composite.technical_score >= 65:
            points.append(f"기술적 추세 우호 (기술적 점수 {composite.technical_score:.0f}/100)")
        # 밸류에이션
        pe = m.get("pe_ratio")
        if pe and 0 < pe < 15:
            points.append(f"저평가 가능성 (P/E {pe:.1f}x)")
        fcf = m.get("free_cashflow")
        if fcf and fcf > 0:
            points.append("양의 잉여현금흐름(FCF) — 재무적 유연성 보유")
        return points[:5]
