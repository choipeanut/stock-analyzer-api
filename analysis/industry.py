"""산업·경쟁 분석 모듈 — 섹터 ETF 상대 성과 + 재무 경쟁력 (pre-fetched info)"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# 섹터명 → SPDR 섹터 ETF
_SECTOR_ETF: dict[str, str] = {
    "Technology":             "XLK",
    "Financial Services":     "XLF",
    "Healthcare":             "XLV",
    "Health Care":            "XLV",
    "Consumer Cyclical":      "XLY",
    "Consumer Discretionary": "XLY",
    "Consumer Defensive":     "XLP",
    "Consumer Staples":       "XLP",
    "Energy":                 "XLE",
    "Industrials":            "XLI",
    "Basic Materials":        "XLB",
    "Materials":              "XLB",
    "Real Estate":            "XLRE",
    "Utilities":              "XLU",
    "Communication Services": "XLC",
}


@dataclass
class IndustryResult:
    score: float = 50.0
    details: dict[str, Any] = field(default_factory=dict)


def _ret(ticker: str, period: str = "3mo") -> float | None:
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty or len(df) < 5:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        return float((close.iloc[-1] - close.iloc[0]) / close.iloc[0])
    except Exception as e:
        logger.debug(f"IndustryAnalyzer ret 실패 ({ticker}): {e}")
        return None


def _price_ret_from_df(price_df: pd.DataFrame, period_days: int = 63) -> float | None:
    """이미 가져온 가격 데이터에서 수익률 계산 (API 재호출 없이)"""
    try:
        if price_df is None or price_df.empty:
            return None
        col = "close" if "close" in price_df.columns else "Close"
        if col not in price_df.columns:
            return None
        close = price_df[col].dropna()
        if len(close) < 2:
            return None
        start_idx = max(0, len(close) - period_days)
        return float((close.iloc[-1] - close.iloc[start_idx]) / close.iloc[start_idx])
    except Exception:
        return None


class IndustryAnalyzer:
    """산업·경쟁 분석 — 섹터 ETF 상대 성과 + 재무 경쟁력"""

    def analyze(
        self,
        ticker: str,
        sector: str | None = None,
        info: dict[str, Any] | None = None,
        price_df: pd.DataFrame | None = None,
    ) -> IndustryResult:
        details: dict[str, Any] = {}
        scores: list[float] = []

        # info가 없으면 yfinance로 시도
        if not info:
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                info = {}

        etf = _SECTOR_ETF.get(sector or "") if sector else ""

        # 종목 3개월 수익률 (이미 가져온 price_df 우선, 없으면 API)
        stock_3m = _price_ret_from_df(price_df, 63) if price_df is not None else _ret(ticker)
        if stock_3m is not None:
            details["stock_3m_pct"] = round(stock_3m * 100, 1)

        sp500_3m = _ret("^GSPC", "3mo")

        # ── 섹터 ETF 상대 성과 ───────────────────────────────────────────────
        if etf:
            details["sector_etf"] = etf
            etf_3m = _ret(etf, "3mo")

            if etf_3m is not None:
                details["sector_3m_pct"] = round(etf_3m * 100, 1)

                # 섹터 vs 시장
                if sp500_3m is not None:
                    rel = etf_3m - sp500_3m
                    details["sector_vs_market_pct"] = round(rel * 100, 1)
                    scores.append(max(0, min(100, 50 + rel * 300)))

                # 종목 vs 섹터
                if stock_3m is not None:
                    rel2 = stock_3m - etf_3m
                    details["stock_vs_sector_pct"] = round(rel2 * 100, 1)
                    scores.append(max(0, min(100, 50 + rel2 * 300)))
        else:
            if stock_3m is not None and sp500_3m is not None:
                rel = stock_3m - sp500_3m
                details["stock_vs_market_pct"] = round(rel * 100, 1)
                scores.append(max(0, min(100, 50 + rel * 300)))

        # ── 재무 경쟁력 (pre-fetched info) ──────────────────────────────────
        # 영업이익률
        om = info.get("operatingMargins")
        if om is not None:
            details["op_margin_pct"] = round(float(om) * 100, 1)
            scores.append(85 if om > 0.25 else 70 if om > 0.15 else 55 if om > 0.05 else 40 if om > 0 else 20)

        # ROE
        roe = info.get("returnOnEquity")
        if roe is not None:
            scores.append(80 if roe > 0.20 else 65 if roe > 0.10 else 50 if roe > 0 else 25)

        # 매출 성장 vs 업종 기대치
        rg = info.get("revenueGrowth")
        if rg is not None:
            details["rev_growth_pct"] = round(float(rg) * 100, 1)
            scores.append(85 if rg > 0.15 else 65 if rg > 0.05 else 50 if rg > 0 else 30)

        # 시가총액 규모 (대형주 프리미엄 — 경쟁력 대리변수)
        cap = info.get("marketCap") or 0
        if cap > 0:
            if   cap > 10e12: scores.append(90)
            elif cap > 1e12:  scores.append(78)
            elif cap > 1e11:  scores.append(65)
            elif cap > 1e10:  scores.append(57)
            else:             scores.append(50)

        final = max(0.0, min(100.0, sum(scores) / len(scores) if scores else 50.0))

        if   final >= 65: note = f"{'섹터' if etf else '시장'} 대비 경쟁 우위"
        elif final >= 45: note = f"{'섹터' if etf else '시장'} 대비 중립적 경쟁력"
        else:             note = f"{'섹터' if etf else '시장'} 대비 상대적 열위"
        details["summary"] = note

        return IndustryResult(score=round(final, 1), details=details)
