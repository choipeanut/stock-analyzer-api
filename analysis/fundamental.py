"""기본적 분석 모듈 (FundamentalAnalyzer)"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from data.processors.data_processor import DataProcessor

logger = logging.getLogger(__name__)


@dataclass
class FundamentalResult:
    ticker: str
    # 세부 점수 (0~100)
    profitability_score: float = 50.0
    growth_score: float = 50.0
    stability_score: float = 50.0
    cashflow_score: float = 50.0
    valuation_score: float = 50.0
    # 가중합
    total_score: float = 50.0
    # 주요 지표 원본값
    metrics: dict[str, Any] = field(default_factory=dict)
    # Altman Z-Score, Piotroski F-Score
    altman_z: float | None = None
    piotroski_f: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


class FundamentalAnalyzer:
    """기본적 분석 — 수익성·성장성·안정성·현금흐름·밸류에이션"""

    SUB_WEIGHTS = {
        "profitability": 0.25,
        "growth": 0.25,
        "stability": 0.20,
        "cashflow": 0.15,
        "valuation": 0.15,
    }

    def __init__(self, processor: DataProcessor | None = None):
        self.processor = processor or DataProcessor()

    def analyze(
        self,
        ticker: str,
        metrics: dict[str, Any],
        price_df: pd.DataFrame | None = None,
        sector_peers: list[dict[str, Any]] | None = None,
    ) -> FundamentalResult:
        result = FundamentalResult(ticker=ticker, metrics=metrics)

        result.profitability_score = self._score_profitability(metrics)
        result.growth_score = self._score_growth(metrics)
        result.stability_score = self._score_stability(metrics)
        result.cashflow_score = self._score_cashflow(metrics)
        result.valuation_score = self._score_valuation(metrics, sector_peers)

        result.total_score = (
            result.profitability_score * self.SUB_WEIGHTS["profitability"]
            + result.growth_score * self.SUB_WEIGHTS["growth"]
            + result.stability_score * self.SUB_WEIGHTS["stability"]
            + result.cashflow_score * self.SUB_WEIGHTS["cashflow"]
            + result.valuation_score * self.SUB_WEIGHTS["valuation"]
        )

        result.altman_z = self._altman_z_score(metrics)
        result.piotroski_f = self._piotroski_f_score(metrics)

        result.details = {
            "sub_scores": {
                "profitability": result.profitability_score,
                "growth": result.growth_score,
                "stability": result.stability_score,
                "cashflow": result.cashflow_score,
                "valuation": result.valuation_score,
            },
            "altman_z": result.altman_z,
            "piotroski_f": result.piotroski_f,
            "dcf_value": self._dcf_valuation(metrics),
        }
        return result

    # ------------------------------------------------------------------ #
    # 수익성 점수
    # ------------------------------------------------------------------ #

    def _score_profitability(self, m: dict[str, Any]) -> float:
        scores = []
        # ROE (높을수록 좋음, 기준: 15% = 70점)
        roe = m.get("roe")
        if roe is not None:
            scores.append(self.processor.normalize_score(roe, -0.2, 0.5))
        # ROA (기준: 5% = 70점)
        roa = m.get("roa")
        if roa is not None:
            scores.append(self.processor.normalize_score(roa, -0.1, 0.25))
        # 영업이익률
        op_margin = m.get("operating_margin")
        if op_margin is not None:
            scores.append(self.processor.normalize_score(op_margin, -0.1, 0.4))
        # 순이익률
        net_margin = m.get("profit_margin")
        if net_margin is not None:
            scores.append(self.processor.normalize_score(net_margin, -0.1, 0.3))

        return float(np.mean(scores)) if scores else 50.0

    # ------------------------------------------------------------------ #
    # 성장성 점수
    # ------------------------------------------------------------------ #

    def _score_growth(self, m: dict[str, Any]) -> float:
        scores = []
        rev_growth = m.get("revenue_growth") or m.get("revenue_yoy")
        if rev_growth is not None:
            scores.append(self.processor.normalize_score(rev_growth, -0.2, 0.5))
        eps_growth = m.get("earnings_growth")
        if eps_growth is not None:
            scores.append(self.processor.normalize_score(eps_growth, -0.3, 0.6))
        # Forward EPS vs Trailing EPS
        fwd_eps = m.get("forward_eps")
        trailing_eps = m.get("eps")
        if fwd_eps and trailing_eps and trailing_eps != 0:
            fwd_growth = (fwd_eps - trailing_eps) / abs(trailing_eps)
            scores.append(self.processor.normalize_score(fwd_growth, -0.3, 0.5))
        return float(np.mean(scores)) if scores else 50.0

    # ------------------------------------------------------------------ #
    # 안정성 점수
    # ------------------------------------------------------------------ #

    def _score_stability(self, m: dict[str, Any]) -> float:
        scores = []
        # 부채비율 (낮을수록 좋음)
        de = m.get("debt_to_equity")
        if de is not None:
            scores.append(self.processor.normalize_score(de, 0, 200, invert=True))
        # 유동비율 (높을수록 좋음)
        cr = m.get("current_ratio")
        if cr is not None:
            scores.append(self.processor.normalize_score(cr, 0.5, 3.0))
        # 이자보상배율 (EBIT / 이자비용) — info 없으면 생략
        # Altman Z-Score 보조
        altman = self._altman_z_score(m)
        if altman is not None:
            # Z > 2.99: safe, 1.81~2.99: grey, < 1.81: distress
            z_score = self.processor.normalize_score(altman, 0, 5.0)
            scores.append(z_score)
        return float(np.mean(scores)) if scores else 50.0

    # ------------------------------------------------------------------ #
    # 현금흐름 점수
    # ------------------------------------------------------------------ #

    def _score_cashflow(self, m: dict[str, Any]) -> float:
        scores = []
        fcf = m.get("free_cashflow")
        ocf = m.get("operating_cashflow")
        market_cap = m.get("market_cap")
        if fcf is not None and market_cap and market_cap > 0:
            fcf_yield = fcf / market_cap
            scores.append(self.processor.normalize_score(fcf_yield, -0.05, 0.15))
        if ocf is not None and ocf > 0:
            scores.append(70.0)  # OCF 양수: 기본 70점
        elif ocf is not None:
            scores.append(30.0)
        return float(np.mean(scores)) if scores else 50.0

    # ------------------------------------------------------------------ #
    # 밸류에이션 점수
    # ------------------------------------------------------------------ #

    def _score_valuation(self, m: dict[str, Any], peers: list[dict] | None = None) -> float:
        scores = []
        # P/E (낮을수록 좋음, 음수 제외)
        pe = m.get("pe_ratio")
        if pe is not None and pe > 0:
            scores.append(self.processor.normalize_score(pe, 5, 50, invert=True))
        # Forward P/E
        fwd_pe = m.get("forward_pe")
        if fwd_pe is not None and fwd_pe > 0:
            scores.append(self.processor.normalize_score(fwd_pe, 5, 40, invert=True))
        # P/B
        pb = m.get("pb_ratio")
        if pb is not None and pb > 0:
            scores.append(self.processor.normalize_score(pb, 0.5, 5.0, invert=True))
        # EV/EBITDA
        ev_ebitda = m.get("ev_ebitda")
        if ev_ebitda is not None and ev_ebitda > 0:
            scores.append(self.processor.normalize_score(ev_ebitda, 3, 25, invert=True))
        # PEG
        peg = m.get("peg_ratio")
        if peg is not None and peg > 0:
            scores.append(self.processor.normalize_score(peg, 0.5, 3.0, invert=True))
        # 동종업계 상대 비교
        if peers:
            peer_pes = [p.get("pe_ratio") for p in peers if p.get("pe_ratio") and p["pe_ratio"] > 0]
            if peer_pes and pe and pe > 0:
                peer_series = pd.Series(peer_pes)
                pct = self.processor.percentile_score(peer_series, pe, invert=True)
                scores.append(pct)
        return float(np.mean(scores)) if scores else 50.0

    # ------------------------------------------------------------------ #
    # Altman Z-Score
    # ------------------------------------------------------------------ #

    def _altman_z_score(self, m: dict[str, Any]) -> float | None:
        """수정 Altman Z-Score (비제조업 모델)"""
        try:
            total_assets = m.get("total_assets")
            total_equity = m.get("total_equity")
            total_liabilities = m.get("total_liabilities")
            revenue = m.get("total_revenue")
            ebit = m.get("ebit")
            market_cap = m.get("market_cap")

            if not all([total_assets, total_equity, total_liabilities, revenue, ebit, market_cap]):
                return None
            if total_assets == 0:
                return None

            # 비제조업 Z' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4
            # X1 = Working Capital / Total Assets (근사: current_ratio 기반)
            cr = m.get("current_ratio") or 1.0
            wc = total_assets * (cr - 1) / cr  # 근사
            x1 = wc / total_assets
            x2 = (total_equity or 0) / total_assets
            x3 = ebit / total_assets
            x4 = market_cap / max(abs(total_liabilities), 1)

            z = 6.56 * x1 + 3.26 * x2 + 6.72 * x3 + 1.05 * x4
            return round(z, 3)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Piotroski F-Score
    # ------------------------------------------------------------------ #

    def _piotroski_f_score(self, m: dict[str, Any]) -> int | None:
        score = 0
        signals = 0

        roa = m.get("roa")
        if roa is not None:
            signals += 1
            if roa > 0:
                score += 1

        ocf = m.get("operating_cashflow")
        if ocf is not None:
            signals += 1
            if ocf > 0:
                score += 1

        rev_growth = m.get("earnings_growth")
        if rev_growth is not None:
            signals += 1
            if rev_growth > 0:
                score += 1

        de = m.get("debt_to_equity")
        if de is not None:
            signals += 1
            if de < 100:
                score += 1

        cr = m.get("current_ratio")
        if cr is not None:
            signals += 1
            if cr > 1:
                score += 1

        op_margin = m.get("operating_margin")
        if op_margin is not None:
            signals += 1
            if op_margin > 0:
                score += 1

        return score if signals >= 4 else None

    # ------------------------------------------------------------------ #
    # DCF 내재가치
    # ------------------------------------------------------------------ #

    def _dcf_valuation(
        self,
        m: dict[str, Any],
        wacc: float = 0.10,
        terminal_growth: float = 0.025,
        years: int = 5,
    ) -> float | None:
        """단순 DCF — FCF 기반"""
        try:
            fcf = m.get("free_cashflow")
            shares = m.get("shares_outstanding")
            rev_growth = m.get("revenue_growth") or m.get("earnings_growth") or 0.05

            if not fcf or not shares or shares == 0:
                return None
            if fcf <= 0:
                return None

            growth = min(max(float(rev_growth), -0.1), 0.3)
            pv = 0.0
            cf = fcf
            for i in range(1, years + 1):
                cf *= (1 + growth)
                pv += cf / (1 + wacc) ** i
            # 터미널 가치
            terminal_cf = cf * (1 + terminal_growth)
            terminal_value = terminal_cf / (wacc - terminal_growth)
            pv += terminal_value / (1 + wacc) ** years

            intrinsic_per_share = pv / shares
            return round(intrinsic_per_share, 2)
        except Exception:
            return None
