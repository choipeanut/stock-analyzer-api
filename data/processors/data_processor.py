"""데이터 정제·정규화 모듈"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataProcessor:
    """원시 데이터를 분석 가능한 형태로 정제"""

    # ------------------------------------------------------------------ #
    # 주가 데이터 정제
    # ------------------------------------------------------------------ #

    def clean_price_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """주가 DataFrame 정제"""
        if df.empty:
            return df

        df = df.copy()
        # 컬럼명 소문자 통일
        df.columns = [c.lower() for c in df.columns]

        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                logger.warning(f"컬럼 누락: {col}")

        # NaN 보간 (전후 선형)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit=5)
        df = df.dropna(subset=["close"])

        # 인덱스 정렬
        df = df.sort_index()
        return df

    # ------------------------------------------------------------------ #
    # 재무 데이터 정제
    # ------------------------------------------------------------------ #

    def extract_financial_metrics(
        self,
        info: dict[str, Any],
        income_stmt: pd.DataFrame,
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
    ) -> dict[str, float | None]:
        """yfinance info + 재무제표에서 핵심 지표 추출"""
        m: dict[str, float | None] = {}

        # --- info 직접 추출 ---
        fields = {
            "market_cap": "marketCap",
            "enterprise_value": "enterpriseValue",
            "pe_ratio": "trailingPE",
            "forward_pe": "forwardPE",
            "pb_ratio": "priceToBook",
            "ps_ratio": "priceToSalesTrailing12Months",
            "ev_ebitda": "enterpriseToEbitda",
            "peg_ratio": "pegRatio",
            "dividend_yield": "dividendYield",
            "payout_ratio": "payoutRatio",
            "beta": "beta",
            "eps": "trailingEps",
            "forward_eps": "forwardEps",
            "book_value": "bookValue",
            "revenue_growth": "revenueGrowth",
            "earnings_growth": "earningsGrowth",
            "profit_margin": "profitMargins",
            "operating_margin": "operatingMargins",
            "roe": "returnOnEquity",
            "roa": "returnOnAssets",
            "current_ratio": "currentRatio",
            "debt_to_equity": "debtToEquity",
            "quick_ratio": "quickRatio",
            "total_revenue": "totalRevenue",
            "gross_profit": "grossProfits",
            "ebitda": "ebitda",
            "free_cashflow": "freeCashflow",
            "operating_cashflow": "operatingCashflow",
            "total_cash": "totalCash",
            "total_debt": "totalDebt",
            "shares_outstanding": "sharesOutstanding",
            "float_shares": "floatShares",
            "52w_high": "fiftyTwoWeekHigh",
            "52w_low": "fiftyTwoWeekLow",
            "analyst_target": "targetMeanPrice",
            "sector": "sector",
            "industry": "industry",
            "currency": "currency",
            "exchange": "exchange",
            "short_name": "shortName",
            "long_name": "longName",
        }
        for key, info_key in fields.items():
            val = info.get(info_key)
            m[key] = float(val) if isinstance(val, (int, float)) else val

        # --- 재무제표에서 추가 계산 ---
        m.update(self._calc_from_statements(income_stmt, balance_sheet, cash_flow))

        # --- PER / PBR fallback (한국 종목 등 yfinance가 제공 안 할 때) ---
        market_cap = m.get("market_cap")
        if market_cap and market_cap > 0:
            # PER fallback: 시가총액 / 순이익(TTM)
            if not m.get("pe_ratio") or m["pe_ratio"] <= 0:
                ni = m.get("net_income")
                if ni and ni > 0:
                    m["pe_ratio"] = round(market_cap / ni, 2)

            # PBR fallback: 시가총액 / 자기자본(장부가)
            if not m.get("pb_ratio") or m["pb_ratio"] <= 0:
                eq = m.get("total_equity")
                if eq and eq > 0:
                    m["pb_ratio"] = round(market_cap / eq, 2)

        # EPS fallback: 순이익 / 발행주식수
        if not m.get("eps") or m["eps"] == 0:
            ni = m.get("net_income")
            shares = m.get("shares_outstanding")
            if ni and shares and shares > 0:
                m["eps"] = round(ni / shares, 4)

        return m

    def _calc_from_statements(
        self,
        income: pd.DataFrame,
        balance: pd.DataFrame,
        cashflow: pd.DataFrame,
    ) -> dict[str, float | None]:
        result: dict[str, float | None] = {}
        try:
            if not income.empty:
                cols = income.columns.tolist()
                if len(cols) >= 2:
                    rev_0 = self._get_val(income, ["Total Revenue", "Revenue", "TotalRevenue"], cols[0])
                    rev_1 = self._get_val(income, ["Total Revenue", "Revenue", "TotalRevenue"], cols[1])
                    if rev_0 and rev_1 and rev_1 != 0:
                        result["revenue_yoy"] = (rev_0 - rev_1) / abs(rev_1)
                ebit = self._get_val(income, ["EBIT", "Operating Income", "OperatingIncome"], cols[0])
                result["ebit"] = ebit
                # 순이익 (PER 계산용) — 다양한 yfinance 필드명 시도
                ni = self._get_val(income, [
                    "Net Income",
                    "Net Income Common Stockholders",
                    "Net Income From Continuing Operation Net Minority Interest",
                    "Net Income Including Noncontrolling Interests",
                    "NetIncome",
                ], cols[0])
                result["net_income"] = ni

            if not balance.empty:
                cols = balance.columns.tolist()
                total_assets = self._get_val(balance, ["Total Assets", "TotalAssets"], cols[0])
                total_equity = self._get_val(balance, [
                    "Total Stockholder Equity",
                    "Stockholders Equity",
                    "Common Stock Equity",
                    "TotalEquityGrossMinorityInterest",
                    "Stockholders' Equity",
                ], cols[0])
                total_liabilities = self._get_val(
                    balance, ["Total Liab", "Total Liabilities Net Minority Interest"], cols[0]
                )
                result["total_assets"] = total_assets
                result["total_equity"] = total_equity
                result["total_liabilities"] = total_liabilities

            if not cashflow.empty:
                cols = cashflow.columns.tolist()
                capex = self._get_val(cashflow, ["Capital Expenditures", "CapitalExpenditures"], cols[0])
                result["capex"] = capex

        except Exception as e:
            logger.debug(f"재무 지표 계산 중 오류: {e}")

        return result

    @staticmethod
    def _get_val(df: pd.DataFrame, keys: list[str], col: Any) -> float | None:
        for k in keys:
            if k in df.index:
                val = df.loc[k, col]
                if pd.notna(val):
                    return float(val)
        return None

    # ------------------------------------------------------------------ #
    # 정규화
    # ------------------------------------------------------------------ #

    @staticmethod
    def normalize_score(value: float, low: float, high: float, invert: bool = False) -> float:
        """값을 0~100 점수로 정규화"""
        if high == low:
            return 50.0
        score = max(0.0, min(100.0, (value - low) / (high - low) * 100))
        return 100.0 - score if invert else score

    @staticmethod
    def percentile_score(series: pd.Series, value: float, invert: bool = False) -> float:
        """시리즈 내 백분위 점수 반환 (0~100)"""
        if series.empty or pd.isna(value):
            return 50.0
        clean = series.dropna()
        pct = float((clean <= value).mean() * 100)
        return 100.0 - pct if invert else pct
