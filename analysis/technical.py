"""기술적 분석 모듈 (TechnicalAnalyzer)"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 신호 방향
SIGNAL_STRONG_BUY = "STRONG BUY"
SIGNAL_BUY = "BUY"
SIGNAL_NEUTRAL = "NEUTRAL"
SIGNAL_SELL = "SELL"
SIGNAL_STRONG_SELL = "STRONG SELL"


@dataclass
class TechnicalResult:
    ticker: str
    # 신호
    short_term_signal: str = SIGNAL_NEUTRAL   # 1~4주
    mid_term_signal: str = SIGNAL_NEUTRAL     # 1~3개월
    short_confidence: float = 0.5
    mid_confidence: float = 0.5
    # 세부 점수 (0~100)
    trend_score: float = 50.0
    momentum_score: float = 50.0
    volume_score: float = 50.0
    volatility_score: float = 50.0
    # 종합 점수
    total_score: float = 50.0
    # 지지·저항
    support_levels: list[float] = field(default_factory=list)
    resistance_levels: list[float] = field(default_factory=list)
    # 피보나치
    fib_levels: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class TechnicalAnalyzer:
    """기술적 분석 — 추세·모멘텀·거래량·변동성·패턴"""

    SUB_WEIGHTS = {
        "trend": 0.30,
        "momentum": 0.25,
        "volume": 0.20,
        "volatility": 0.15,
        "pattern": 0.10,
    }

    def analyze(self, ticker: str, df: pd.DataFrame) -> TechnicalResult:
        result = TechnicalResult(ticker=ticker)

        if df.empty or len(df) < 30:
            logger.warning(f"{ticker}: 기술적 분석을 위한 데이터 부족 ({len(df)}행)")
            return result

        result.trend_score = self._score_trend(df)
        result.momentum_score = self._score_momentum(df)
        result.volume_score = self._score_volume(df)
        result.volatility_score = self._score_volatility(df)
        pattern_score = self._score_pattern(df)

        result.total_score = (
            result.trend_score * self.SUB_WEIGHTS["trend"]
            + result.momentum_score * self.SUB_WEIGHTS["momentum"]
            + result.volume_score * self.SUB_WEIGHTS["volume"]
            + result.volatility_score * self.SUB_WEIGHTS["volatility"]
            + pattern_score * self.SUB_WEIGHTS["pattern"]
        )

        result.short_term_signal, result.short_confidence = self._signal(result.total_score, df, "short")
        result.mid_term_signal, result.mid_confidence = self._signal(result.total_score, df, "mid")

        result.support_levels, result.resistance_levels = self._support_resistance(df)
        result.fib_levels = self._fibonacci(df)

        result.details = {
            "sub_scores": {
                "trend": result.trend_score,
                "momentum": result.momentum_score,
                "volume": result.volume_score,
                "volatility": result.volatility_score,
                "pattern": pattern_score,
            },
            "current_price": float(df["close"].iloc[-1]),
            "ma20": self._last(df, "ma20"),
            "ma60": self._last(df, "ma60"),
            "ma120": self._last(df, "ma120"),
            "ma200": self._last(df, "ma200"),
            "rsi": self._last(df, "rsi"),
            "macd": self._last(df, "macd"),
            "macd_signal": self._last(df, "macd_signal"),
            "bb_upper": self._last(df, "bb_upper"),
            "bb_lower": self._last(df, "bb_lower"),
            "stoch_k": self._last(df, "stoch_k"),
            "stoch_d": self._last(df, "stoch_d"),
            "atr": self._last(df, "atr"),
            "obv": self._last(df, "obv"),
            "vol_ratio": self._last(df, "vol_ratio"),
            "hv20": self._last(df, "hv20"),
        }
        return result

    # ------------------------------------------------------------------ #
    # 추세 점수
    # ------------------------------------------------------------------ #

    def _score_trend(self, df: pd.DataFrame) -> float:
        scores = []
        price = df["close"].iloc[-1]

        for col, weight in [("ma20", 1.5), ("ma60", 1.0), ("ma120", 0.8), ("ma200", 0.7)]:
            ma = self._last(df, col)
            if ma:
                scores.append(80.0 * weight if price > ma else 30.0 * weight)
                scores.append(weight)  # 분모

        # MACD 신호
        macd = self._last(df, "macd")
        macd_sig = self._last(df, "macd_signal")
        if macd is not None and macd_sig is not None:
            scores.append(75.0 if macd > macd_sig else 35.0)

        # 골든/데드 크로스
        ma20 = self._last(df, "ma20")
        ma60 = self._last(df, "ma60")
        if ma20 and ma60:
            if ma20 > ma60:
                scores.append(70.0)
            else:
                scores.append(35.0)

        return self._weighted_mean(scores)

    # ------------------------------------------------------------------ #
    # 모멘텀 점수
    # ------------------------------------------------------------------ #

    def _score_momentum(self, df: pd.DataFrame) -> float:
        scores = []
        # RSI
        rsi = self._last(df, "rsi")
        if rsi is not None:
            if rsi > 70:
                scores.append(80.0)  # 과매수 — 강세
            elif rsi < 30:
                scores.append(25.0)  # 과매도
            else:
                scores.append(40.0 + (rsi - 30) * (40.0 / 40))  # 30~70 → 40~80

        # 스토캐스틱
        stoch_k = self._last(df, "stoch_k")
        stoch_d = self._last(df, "stoch_d")
        if stoch_k is not None and stoch_d is not None:
            if stoch_k > stoch_d:
                scores.append(65.0)
            else:
                scores.append(40.0)

        # 수익률 모멘텀
        ret_20 = self._last(df, "ret_20d")
        if ret_20 is not None:
            scores.append(min(100, max(0, 50 + ret_20 * 200)))

        ret_60 = self._last(df, "ret_60d")
        if ret_60 is not None:
            scores.append(min(100, max(0, 50 + ret_60 * 100)))

        return float(np.mean(scores)) if scores else 50.0

    # ------------------------------------------------------------------ #
    # 거래량 점수
    # ------------------------------------------------------------------ #

    def _score_volume(self, df: pd.DataFrame) -> float:
        scores = []
        vol_ratio = self._last(df, "vol_ratio")
        if vol_ratio is not None:
            # 거래량 폭발 (2배 이상) — 방향과 함께 판단
            ret = self._last(df, "ret_1d") or 0
            if vol_ratio > 2 and ret > 0:
                scores.append(85.0)
            elif vol_ratio > 2 and ret < 0:
                scores.append(20.0)
            elif vol_ratio > 1.2:
                scores.append(65.0)
            else:
                scores.append(50.0)

        # OBV 추세
        if "obv" in df.columns and len(df) >= 20:
            obv_recent = df["obv"].iloc[-10:].mean()
            obv_prev = df["obv"].iloc[-20:-10].mean()
            if obv_recent > obv_prev:
                scores.append(70.0)
            else:
                scores.append(40.0)

        return float(np.mean(scores)) if scores else 50.0

    # ------------------------------------------------------------------ #
    # 변동성 점수 (낮은 변동성 = 안정 = 높은 점수)
    # ------------------------------------------------------------------ #

    def _score_volatility(self, df: pd.DataFrame) -> float:
        scores = []
        hv = self._last(df, "hv20")
        if hv is not None:
            # HV 20% 미만: 높은 점수
            scores.append(max(0, min(100, 100 - hv * 300)))

        # 볼린저 밴드 폭 (좁을수록 안정)
        bb_width = self._last(df, "bb_width")
        if bb_width is not None:
            scores.append(max(0, min(100, 100 - bb_width * 200)))

        return float(np.mean(scores)) if scores else 50.0

    # ------------------------------------------------------------------ #
    # 패턴 점수 (단순 규칙 기반)
    # ------------------------------------------------------------------ #

    def _score_pattern(self, df: pd.DataFrame) -> float:
        if len(df) < 5:
            return 50.0
        scores = []

        # 도지 패턴
        last = df.iloc[-1]
        if all(c in df.columns for c in ["open", "high", "low", "close"]):
            body = abs(last["close"] - last["open"])
            shadow = last["high"] - last["low"]
            if shadow > 0 and body / shadow < 0.1:
                scores.append(50.0)  # 도지: 중립

            # 해머 (하단 꼬리 길고 작은 몸통)
            lower_shadow = min(last["open"], last["close"]) - last["low"]
            if shadow > 0 and lower_shadow / shadow > 0.6 and body / shadow < 0.3:
                scores.append(70.0)

        # 가격이 BB 하단 근처 (반등 가능)
        bb_pct = self._last(df, "bb_pct")
        if bb_pct is not None:
            if bb_pct < 0.2:
                scores.append(70.0)
            elif bb_pct > 0.8:
                scores.append(55.0)
            else:
                scores.append(50.0)

        return float(np.mean(scores)) if scores else 50.0

    # ------------------------------------------------------------------ #
    # 신호 변환
    # ------------------------------------------------------------------ #

    def _signal(self, score: float, df: pd.DataFrame, horizon: str) -> tuple[str, float]:
        if score >= 80:
            sig, conf = SIGNAL_STRONG_BUY, 0.9
        elif score >= 65:
            sig, conf = SIGNAL_BUY, 0.7
        elif score >= 45:
            sig, conf = SIGNAL_NEUTRAL, 0.5
        elif score >= 30:
            sig, conf = SIGNAL_SELL, 0.7
        else:
            sig, conf = SIGNAL_STRONG_SELL, 0.9

        # 중기는 MA60/120 기반 조정
        if horizon == "mid" and len(df) >= 60:
            price = df["close"].iloc[-1]
            ma60 = self._last(df, "ma60")
            ma120 = self._last(df, "ma120")
            if ma60 and ma120:
                if price > ma60 and price > ma120 and score < 65:
                    sig = SIGNAL_BUY
                    conf = 0.6
        return sig, conf

    # ------------------------------------------------------------------ #
    # 지지·저항
    # ------------------------------------------------------------------ #

    def _support_resistance(self, df: pd.DataFrame) -> tuple[list[float], list[float]]:
        if len(df) < 30:
            return [], []
        price = df["close"].iloc[-1]
        highs = df["close"].rolling(10).max().dropna()
        lows = df["close"].rolling(10).min().dropna()

        resistances = sorted(
            [h for h in highs.unique() if h > price * 1.01],
            key=lambda x: abs(x - price),
        )[:3]
        supports = sorted(
            [lo for lo in lows.unique() if lo < price * 0.99],
            key=lambda x: abs(x - price),
        )[:3]
        return [round(s, 2) for s in supports], [round(r, 2) for r in resistances]

    # ------------------------------------------------------------------ #
    # 피보나치
    # ------------------------------------------------------------------ #

    def _fibonacci(self, df: pd.DataFrame, lookback: int = 120) -> dict[str, float]:
        sub = df.tail(lookback)
        if sub.empty:
            return {}
        high = float(sub["close"].max())
        low = float(sub["close"].min())
        diff = high - low
        return {
            "high": round(high, 2),
            "low": round(low, 2),
            "fib_23.6": round(high - diff * 0.236, 2),
            "fib_38.2": round(high - diff * 0.382, 2),
            "fib_50.0": round(high - diff * 0.500, 2),
            "fib_61.8": round(high - diff * 0.618, 2),
        }

    # ------------------------------------------------------------------ #
    # 유틸
    # ------------------------------------------------------------------ #

    @staticmethod
    def _last(df: pd.DataFrame, col: str) -> float | None:
        if col not in df.columns:
            return None
        val = df[col].dropna()
        return float(val.iloc[-1]) if not val.empty else None

    @staticmethod
    def _weighted_mean(values: list[float]) -> float:
        if not values:
            return 50.0
        return float(np.mean(values))
