"""Yahoo Finance 데이터 수집 클라이언트"""
from __future__ import annotations

import logging
import time
from typing import Any

import pandas as pd
import yfinance as yf

from ..cache_manager import CacheManager

logger = logging.getLogger(__name__)


def _get_session():
    """curl_cffi 세션 반환 (브라우저 TLS 지문 모방 → rate limit 우회)
    curl_cffi 미설치 시 None 반환 (yfinance 기본 동작).
    """
    try:
        from curl_cffi import requests as cfr
        return cfr.Session(impersonate="chrome110")
    except Exception:
        return None


def _close_scalar(df: pd.DataFrame) -> float | None:
    """yfinance DataFrame에서 Close 마지막 값을 안전하게 float로 추출"""
    try:
        close = df["Close"]
        if isinstance(close, pd.DataFrame):   # MultiIndex 잔재
            close = close.iloc[:, 0]
        val = close.dropna().iloc[-1]
        return float(val)
    except Exception:
        return None


def _krx_to_yf(ticker: str, market: str) -> str:
    """KRX 종목코드를 yfinance 형식으로 변환"""
    market = market.upper()
    if market in ("KRX", "KOSPI", "KOSDAQ"):
        # 6자리 숫자면 한국 종목
        if ticker.isdigit() and len(ticker) == 6:
            return f"{ticker}.KS" if market in ("KRX", "KOSPI") else f"{ticker}.KQ"
    return ticker


class YFinanceClient:
    """Yahoo Finance API 클라이언트 (캐시 지원)"""

    def __init__(self, cache: CacheManager | None = None, retry: int = 3, delay: float = 5.0):
        self.cache = cache or CacheManager()
        self.retry = retry
        self.delay = delay
        self.session = _get_session()   # curl_cffi 세션 (rate limit 우회)

    # ------------------------------------------------------------------ #
    # Public Methods
    # ------------------------------------------------------------------ #

    def get_ticker_info(self, ticker: str, market: str = "KRX") -> dict[str, Any]:
        """종목 기본 정보 반환 (yfinance 0.2.x 호환 + fast_info 보완)"""
        yf_ticker = _krx_to_yf(ticker, market)
        cache_key = f"info:{yf_ticker}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        t = yf.Ticker(yf_ticker, session=self.session)

        # ① .info 시도 (느리지만 가장 많은 정보)
        data = self._fetch_with_retry(lambda: t.info) or {}

        # ② fast_info로 빠진 핵심 필드 보완 (yfinance 0.2.x에서 .info보다 안정적)
        try:
            fi = t.fast_info
            def _fi(attr):
                v = getattr(fi, attr, None)
                return float(v) if v is not None else None

            if not data.get("currentPrice") and not data.get("regularMarketPrice"):
                p = _fi("last_price") or _fi("regular_market_price")
                if p:
                    data["currentPrice"] = p
            if not data.get("marketCap"):
                mc = _fi("market_cap")
                if mc:
                    data["marketCap"] = mc
            if not data.get("fiftyTwoWeekHigh"):
                h = _fi("year_high")
                if h:
                    data["fiftyTwoWeekHigh"] = h
            if not data.get("fiftyTwoWeekLow"):
                lo = _fi("year_low")
                if lo:
                    data["fiftyTwoWeekLow"] = lo
            if not data.get("fiftyDayAverage"):
                ma50 = _fi("fifty_day_average")
                if ma50:
                    data["fiftyDayAverage"] = ma50
            if not data.get("twoHundredDayAverage"):
                ma200 = _fi("two_hundred_day_average")
                if ma200:
                    data["twoHundredDayAverage"] = ma200
        except Exception:
            pass

        # ③ 가격이 여전히 없으면 최근 1일 다운로드
        if not data.get("currentPrice") and not data.get("regularMarketPrice"):
            try:
                df1 = yf.download(yf_ticker, period="5d", interval="1d",
                                  progress=False, auto_adjust=True)
                if not df1.empty:
                    v = _close_scalar(df1)
                    if v:
                        data["currentPrice"] = v
            except Exception:
                pass

        self.cache.set(cache_key, data, ttl=3600)   # 1시간 캐시 (너무 오래되면 가격 stale)
        return data

    def get_price_history(
        self,
        ticker: str,
        market: str = "KRX",
        period: str = "3y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """주가 히스토리 반환"""
        yf_ticker = _krx_to_yf(ticker, market)
        cache_key = f"price:{yf_ticker}:{period}:{interval}"
        cached = self.cache.get_df(cache_key)
        if cached is not None:
            return cached

        df = self._fetch_with_retry(
            lambda: yf.download(yf_ticker, period=period, interval=interval,
                                progress=False, auto_adjust=True, session=self.session)
        )
        if df is None or df.empty:
            logger.warning(f"주가 데이터 없음: {yf_ticker}")
            return pd.DataFrame()

        # MultiIndex 컬럼 처리
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        self.cache.set_df(cache_key, df, ttl=300)
        return df

    def get_financials(self, ticker: str, market: str = "KRX") -> dict[str, pd.DataFrame]:
        """재무제표 데이터 반환 (손익계산서, 대차대조표, 현금흐름표)"""
        yf_ticker = _krx_to_yf(ticker, market)
        cache_key = f"financials:{yf_ticker}"
        cached = self.cache.get(cache_key)
        if cached:
            return {k: pd.DataFrame(v) for k, v in cached.items()}

        t = yf.Ticker(yf_ticker, session=self.session)
        result: dict[str, pd.DataFrame] = {}
        for name, attr in [
            ("income_stmt", "financials"),
            ("balance_sheet", "balance_sheet"),
            ("cash_flow", "cashflow"),
        ]:
            try:
                df = getattr(t, attr)
                result[name] = df if df is not None else pd.DataFrame()
            except Exception as e:
                logger.warning(f"{name} 수집 실패 ({yf_ticker}): {e}")
                result[name] = pd.DataFrame()

        serializable = {k: v.to_dict() for k, v in result.items()}
        self.cache.set(cache_key, serializable, ttl=86400)
        return result

    def get_dividends(self, ticker: str, market: str = "KRX") -> pd.Series:
        """배당 데이터"""
        yf_ticker = _krx_to_yf(ticker, market)
        t = yf.Ticker(yf_ticker)
        try:
            return t.dividends
        except Exception:
            return pd.Series(dtype=float)

    def get_universe(self, market: str, limit: int = 200) -> list[str]:
        """시장별 종목 유니버스 반환 (약식)"""
        if market.upper() in ("KOSPI", "KRX"):
            return self._get_kospi_tickers(limit)
        elif market.upper() == "KOSDAQ":
            return self._get_kosdaq_tickers(limit)
        elif market.upper() == "SP500":
            return self._get_sp500_tickers(limit)
        elif market.upper() == "NASDAQ":
            return self._get_nasdaq_tickers(limit)
        else:
            return []

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _fetch_with_retry(self, fn: Any, default: Any = None) -> Any:
        for attempt in range(self.retry):
            try:
                return fn()
            except Exception as e:
                logger.warning(f"API 호출 실패 ({attempt + 1}/{self.retry}): {e}")
                if attempt < self.retry - 1:
                    time.sleep(self.delay)
        logger.error("최대 재시도 초과 — 캐시 Fallback 또는 None 반환")
        return default

    def _get_kospi_tickers(self, limit: int) -> list[str]:
        """KOSPI 주요 종목 (KS 접미사)"""
        majors = [
            # 반도체·전자
            "005930", "000660", "009150", "006400", "003670",
            # IT·플랫폼
            "035420", "035720", "259960", "047050", "251270",
            # 자동차
            "005380", "000270", "012330", "064350", "241560",
            # 화학·소재
            "051910", "096770", "010950", "011170", "005490",
            # 금융
            "105560", "055550", "086790", "032830", "139480",
            # 바이오·의약
            "207940", "068270", "323410", "145020", "128940",
            # 통신·미디어
            "017670", "030200", "018260", "053210", "036570",
            # 건설·중공업
            "009540", "042660", "028260", "034730", "000720",
            # 유통·소비
            "004170", "069960", "010130", "001800", "008770",
            # 에너지
            "010060", "015760", "267250", "316140", "003490",
        ]
        return [f"{t}.KS" for t in majors[:limit]]

    def _get_kosdaq_tickers(self, limit: int) -> list[str]:
        majors = [
            # 바이오
            "247540", "091990", "196170", "086900", "214450",
            # 반도체·장비
            "112040", "357780", "166090", "039030", "131970",
            # IT·소프트웨어
            "263750", "293490", "095340", "215600", "041510",
            # 엔터·게임
            "122870", "058470", "035900", "036830", "095660",
            # 2차전지
            "066970", "357780", "011040", "048410", "006110",
        ]
        return [f"{t}.KQ" for t in majors[:limit]]

    def _get_sp500_tickers(self, limit: int) -> list[str]:
        sp500 = [
            # 빅테크
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
            # 금융
            "BRK-B", "JPM", "V", "MA", "BAC", "GS", "MS",
            # 헬스케어
            "UNH", "JNJ", "LLY", "ABBV", "MRK", "PFE", "TMO",
            # 소비재
            "PG", "KO", "PEP", "MCD", "COST", "WMT", "HD",
            # 에너지
            "XOM", "CVX", "COP", "SLB", "EOG",
            # 산업재
            "CAT", "HON", "RTX", "UPS", "BA",
            # 반도체·하드웨어
            "AVGO", "AMD", "QCOM", "INTC", "AMAT", "MU", "LRCX",
            # 소프트웨어·클라우드
            "ADBE", "CRM", "NOW", "SNOW", "PANW",
            # 통신
            "T", "VZ", "NFLX",
        ]
        return sp500[:limit]

    def _get_nasdaq_tickers(self, limit: int) -> list[str]:
        nasdaq = [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
            "AVGO", "ASML", "AMD", "QCOM", "INTC", "ADBE", "NFLX",
            "TXN", "AMAT", "MU", "PANW", "LRCX", "SNPS",
            "MRVL", "KLAC", "CDNS", "CRWD", "FTNT",
            "WDAY", "TEAM", "ZS", "DDOG", "MDB",
        ]
        return nasdaq[:limit]
