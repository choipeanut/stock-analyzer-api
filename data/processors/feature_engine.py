"""기술적 지표 피처 엔지니어링"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FeatureEngine:
    """주가 DataFrame에 기술적 지표 컬럼 추가"""

    def add_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """모든 기술적 지표 추가"""
        if df.empty or "close" not in df.columns:
            return df
        df = df.copy()
        df = self.add_moving_averages(df)
        df = self.add_macd(df)
        df = self.add_rsi(df)
        df = self.add_bollinger_bands(df)
        df = self.add_stochastic(df)
        df = self.add_atr(df)
        df = self.add_obv(df)
        df = self.add_volume_features(df)
        df = self.add_returns(df)
        return df

    # ------------------------------------------------------------------ #
    # 이동평균
    # ------------------------------------------------------------------ #

    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        for n in [20, 60, 120, 200]:
            df[f"ma{n}"] = df["close"].rolling(n).mean()
        df["ema12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema26"] = df["close"].ewm(span=26, adjust=False).mean()
        return df

    # ------------------------------------------------------------------ #
    # MACD
    # ------------------------------------------------------------------ #

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema12 = df["close"].ewm(span=12, adjust=False).mean()
        ema26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema12 - ema26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]
        return df

    # ------------------------------------------------------------------ #
    # RSI
    # ------------------------------------------------------------------ #

    def add_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))
        return df

    # ------------------------------------------------------------------ #
    # 볼린저 밴드
    # ------------------------------------------------------------------ #

    def add_bollinger_bands(self, df: pd.DataFrame, period: int = 20, std: float = 2.0) -> pd.DataFrame:
        mid = df["close"].rolling(period).mean()
        sigma = df["close"].rolling(period).std()
        df["bb_upper"] = mid + std * sigma
        df["bb_mid"] = mid
        df["bb_lower"] = mid - std * sigma
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"].replace(0, np.nan)
        df["bb_pct"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)
        return df

    # ------------------------------------------------------------------ #
    # 스토캐스틱
    # ------------------------------------------------------------------ #

    def add_stochastic(self, df: pd.DataFrame, k: int = 14, d: int = 3) -> pd.DataFrame:
        if "high" not in df.columns or "low" not in df.columns:
            return df
        low_min = df["low"].rolling(k).min()
        high_max = df["high"].rolling(k).max()
        denom = (high_max - low_min).replace(0, np.nan)
        df["stoch_k"] = (df["close"] - low_min) / denom * 100
        df["stoch_d"] = df["stoch_k"].rolling(d).mean()
        return df

    # ------------------------------------------------------------------ #
    # ATR (Average True Range)
    # ------------------------------------------------------------------ #

    def add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        if "high" not in df.columns or "low" not in df.columns:
            return df
        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        df["atr"] = tr.rolling(period).mean()
        return df

    # ------------------------------------------------------------------ #
    # OBV (On Balance Volume)
    # ------------------------------------------------------------------ #

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in df.columns:
            return df
        direction = np.sign(df["close"].diff()).fillna(0)
        df["obv"] = (direction * df["volume"]).cumsum()
        return df

    # ------------------------------------------------------------------ #
    # 거래량 피처
    # ------------------------------------------------------------------ #

    def add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if "volume" not in df.columns:
            return df
        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["vol_ratio"] = df["volume"] / df["vol_ma20"].replace(0, np.nan)
        vol_mean = df["volume"].rolling(20).mean()
        vol_std = df["volume"].rolling(20).std()
        df["vol_zscore"] = (df["volume"] - vol_mean) / vol_std.replace(0, np.nan)
        # VWAP (당일 기준 rolling)
        if "high" in df.columns and "low" in df.columns:
            typical = (df["high"] + df["low"] + df["close"]) / 3
            df["vwap"] = (typical * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
        return df

    # ------------------------------------------------------------------ #
    # 수익률
    # ------------------------------------------------------------------ #

    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["ret_1d"] = df["close"].pct_change(1)
        df["ret_5d"] = df["close"].pct_change(5)
        df["ret_20d"] = df["close"].pct_change(20)
        df["ret_60d"] = df["close"].pct_change(60)
        df["ret_120d"] = df["close"].pct_change(120)
        df["hv20"] = df["ret_1d"].rolling(20).std() * np.sqrt(252)
        return df
