"""리스크 분석 모듈 (RiskAnalyzer)"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RiskResult:
    ticker: str = ""
    # 점수 (높을수록 위험 → 패널티로 사용)
    risk_score: float = 50.0
    beta: float | None = None
    mdd: float | None = None
    var_95: float | None = None
    var_99: float | None = None
    hv: float | None = None
    altman_z: float | None = None
    top_risks: list[str] = field(default_factory=list)
    scenarios: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class RiskAnalyzer:
    """리스크 분석 — 시장·신용·유동성 리스크"""

    def analyze(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        metrics: dict[str, Any],
    ) -> RiskResult:
        result = RiskResult(ticker=ticker)

        # Beta
        result.beta = metrics.get("beta")

        # 가격 기반 리스크 지표
        if not price_df.empty and "close" in price_df.columns:
            rets = price_df["close"].pct_change().dropna()
            result.mdd = self._mdd(price_df["close"])
            result.var_95 = self._var(rets, 0.95)
            result.var_99 = self._var(rets, 0.99)
            result.hv = float(rets.std() * np.sqrt(252)) if len(rets) > 1 else None

        # 종합 리스크 점수 산출 (0~100, 높을수록 위험)
        result.risk_score = self._score_risk(result, metrics)
        result.top_risks = self._identify_risks(result, metrics)
        result.scenarios = self._scenario_analysis(metrics)

        result.details = {
            "beta": result.beta,
            "mdd": result.mdd,
            "var_95": result.var_95,
            "var_99": result.var_99,
            "hv": result.hv,
            "top_risks": result.top_risks,
            "scenarios": result.scenarios,
        }
        return result

    # ------------------------------------------------------------------ #

    def _mdd(self, prices: pd.Series) -> float:
        """최대 낙폭 (Maximum Drawdown)"""
        rolling_max = prices.cummax()
        drawdown = (prices - rolling_max) / rolling_max.replace(0, np.nan)
        return float(drawdown.min())

    def _var(self, returns: pd.Series, confidence: float) -> float:
        """Historical VaR"""
        if len(returns) < 10:
            return 0.0
        return float(np.percentile(returns, (1 - confidence) * 100))

    def _score_risk(self, r: RiskResult, m: dict[str, Any]) -> float:
        """리스크 점수 (0=무위험, 100=극도 위험)"""
        penalties = []

        # 베타 리스크 (베타 > 1.5 = 고위험)
        if r.beta is not None:
            beta_risk = min(100, max(0, (abs(r.beta) - 0.5) * 40))
            penalties.append(beta_risk)

        # MDD 리스크 (MDD > 50% = 극고위험)
        if r.mdd is not None:
            mdd_risk = min(100, abs(r.mdd) * 150)
            penalties.append(mdd_risk)

        # VaR 리스크
        if r.var_95 is not None:
            var_risk = min(100, abs(r.var_95) * 500)
            penalties.append(var_risk)

        # 부채 리스크
        de = m.get("debt_to_equity")
        if de is not None:
            de_risk = min(100, de / 2)
            penalties.append(de_risk)

        # Altman Z-Score (낮을수록 위험)
        z = m.get("altman_z") or self._get_z(m)
        if z is not None:
            if z < 1.81:
                penalties.append(90.0)
            elif z < 2.99:
                penalties.append(50.0)
            else:
                penalties.append(10.0)

        return float(np.mean(penalties)) if penalties else 50.0

    def _get_z(self, m: dict[str, Any]) -> float | None:
        return m.get("altman_z")

    def _identify_risks(self, r: RiskResult, m: dict[str, Any]) -> list[str]:
        risks = []
        if r.beta and abs(r.beta) > 1.5:
            risks.append(f"고베타 리스크 (β={r.beta:.2f}) — 시장 변동 증폭")
        if r.mdd and abs(r.mdd) > 0.3:
            risks.append(f"큰 최대낙폭 (MDD={r.mdd:.1%}) — 과거 대형 손실 발생")
        if r.var_95 and abs(r.var_95) > 0.03:
            risks.append(f"높은 일일 VaR-95% ({r.var_95:.1%}) — 단기 손실 가능성")
        de = m.get("debt_to_equity")
        if de and de > 150:
            risks.append(f"높은 부채비율 ({de:.0f}%) — 신용 리스크")
        cr = m.get("current_ratio")
        if cr and cr < 1.0:
            risks.append(f"유동비율 위험 ({cr:.2f}) — 단기 유동성 부족")
        if r.hv and r.hv > 0.5:
            risks.append(f"높은 역사적 변동성 (HV={r.hv:.1%})")
        return risks[:5]

    def _scenario_analysis(self, m: dict[str, Any]) -> dict[str, float]:
        """베이스/불/베어 시나리오 내재가치"""
        dcf_base = m.get("dcf_value")
        if not dcf_base:
            return {}
        return {
            "bull": round(dcf_base * 1.3, 2),
            "base": round(dcf_base, 2),
            "bear": round(dcf_base * 0.7, 2),
        }
