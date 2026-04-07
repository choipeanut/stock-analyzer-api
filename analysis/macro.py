"""거시경제 분석 모듈 — 실시간 시장 데이터 기반 (지정학 리스크 포함)

지정학 리스크 스코어링 원칙
─────────────────────────────
전쟁, 선거, 정책 충격 등 지정학 이벤트는 반드시 시장에 반영됩니다.
뉴스 자체를 인식하는 대신, 이벤트의 결과(금·유가·VIX·환율 변동)를
측정해 역산합니다.

  전쟁 장기화  → 금 급등 + 유가 급등 + VIX 상승
  휴전·협상    → 금 안정 + VIX 하락
  트럼프 재선  → USD 강세 + 관세 우려 → 신흥국 환율 약세
  금리 인상    → 장기금리 상승 + 성장주 조정
  경기 침체    → 금리 역전 + VIX 폭등 + 증시 급락

따라서 VIX·금·유가·DXY·KRW·KOSPI 지표 조합이 곧 지정학 리스크 온도계입니다.
뉴스 키워드는 시장이 미처 반영하지 못한 초기 이벤트를 잡는 보너스 신호입니다.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def _yf_session():
    try:
        from curl_cffi import requests as cfr
        return cfr.Session(impersonate="chrome110")
    except Exception:
        return None


_SESSION = _yf_session()

CYCLE_RECOVERY  = "Recovery"
CYCLE_EXPANSION = "Expansion"
CYCLE_SLOWDOWN  = "Slowdown"
CYCLE_RECESSION = "Recession"

# 섹터별 금리 민감도 (금리 상승 시 영향: 음수=부정적)
_RATE_SENSITIVITY: dict[str, float] = {
    "Technology":             -1.5,
    "Communication Services": -1.2,
    "Consumer Discretionary": -1.0,
    "Real Estate":            -2.0,
    "Utilities":              -1.5,
    "Financial Services":     +1.2,
    "Energy":                 +0.3,
    "Materials":              -0.5,
    "Industrials":            -0.5,
    "Consumer Staples":       -0.3,
    "Health Care":            -0.2,
}

# 섹터별 유가 민감도 (유가 상승 시 영향)
_OIL_SENSITIVITY: dict[str, float] = {
    "Energy":                 +2.0,
    "Materials":              +0.5,
    "Industrials":            -0.5,
    "Consumer Discretionary": -0.8,
    "Consumer Staples":       -0.6,
    "Technology":             -0.3,
    "Airlines":               -1.5,
}

# 지정학 고위험 키워드
_GEO_HIGH_RISK = [
    "war", "warfare", "invasion", "missile", "airstrike", "nuclear",
    "troops", "military offensive", "combat", "coup", "assassination",
    "전쟁", "침공", "미사일", "핵", "공습", "쿠데타", "암살",
    "sanctions", "embargo", "trade war", "blockade",
    "제재", "봉쇄", "수출 금지",
    "default", "banking crisis", "collapse",
    "채무 불이행", "금융 위기",
]
_GEO_MED_RISK = [
    "tension", "escalation", "unrest", "crisis", "threat", "warning",
    "tariff", "ban", "conflict", "terrorism", "attack",
    "긴장", "확전", "불안", "위기", "위협", "관세", "테러", "분쟁",
]


@dataclass
class MacroResult:
    sector_affinity: dict[str, float] = field(default_factory=dict)
    cycle: str = CYCLE_EXPANSION
    score: float = 50.0
    details: dict[str, Any] = field(default_factory=dict)


# ── 시장 데이터 헬퍼 ─────────────────────────────────────────────────────────

def _safe_last(ticker: str, period: str = "5d") -> float | None:
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True, session=_SESSION)
        if df.empty:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return float(close.dropna().iloc[-1])
    except Exception as e:
        logger.debug(f"macro safe_last 실패 ({ticker}): {e}")
        return None


def _pct_change(ticker: str, period: str = "1mo") -> float | None:
    try:
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True, session=_SESSION)
        if df.empty or len(df) < 2:
            return None
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        return float((close.iloc[-1] - close.iloc[0]) / close.iloc[0])
    except Exception as e:
        logger.debug(f"macro pct_change 실패 ({ticker}): {e}")
        return None


# ── 메인 분석기 ──────────────────────────────────────────────────────────────

class MacroAnalyzer:
    """실시간 거시경제 + 지정학 리스크 분석"""

    def analyze(self, sector: str | None = None) -> MacroResult:
        details: dict[str, Any] = {}
        scores:  list[float]    = []

        # ── 1. 미국 금리 (10Y) ──────────────────────────────────────────────
        us_10y = _safe_last("^TNX")
        us_3m  = _safe_last("^IRX")
        if us_10y is not None:
            details["us_10y_yield"] = round(us_10y, 2)
            # 2%=100점 / 4%=60점 / 6%=20점
            rate_score = max(0, min(100, 100 - (us_10y - 2.0) * 20))
            scores.append(rate_score)
        else:
            scores.append(50.0)

        yield_spread = None
        if us_10y is not None and us_3m is not None:
            yield_spread = us_10y - us_3m
            details["yield_spread"] = round(yield_spread, 2)

        # ── 2. VIX (공포지수) ───────────────────────────────────────────────
        vix = _safe_last("^VIX")
        if vix is not None:
            details["vix"] = round(vix, 1)
            # <15 낮음 / 15-20 보통 / 20-30 높음 / 30+ 공황
            vix_score = max(0, min(100, 100 - (vix - 10) * 2.5))
            scores.append(vix_score)
        else:
            scores.append(50.0)

        # ── 3. S&P500 1개월 추세 ────────────────────────────────────────────
        sp500_1m = _pct_change("^GSPC", "1mo")
        sp500_3m = _pct_change("^GSPC", "3mo")
        if sp500_1m is not None:
            details["sp500_1m"] = round(sp500_1m * 100, 1)
            scores.append(max(0, min(100, 50 + sp500_1m * 300)))
        else:
            scores.append(50.0)
        if sp500_3m is not None:
            details["sp500_3m"] = round(sp500_3m * 100, 1)

        # ── 4. KOSPI 1개월 추세 ─────────────────────────────────────────────
        kospi_1m = _pct_change("^KS11", "1mo")
        if kospi_1m is not None:
            details["kospi_1m"] = round(kospi_1m * 100, 1)
            scores.append(max(0, min(100, 50 + kospi_1m * 300)))
        else:
            scores.append(50.0)

        # ── 5. 원/달러 환율 ─────────────────────────────────────────────────
        krw     = _safe_last("KRW=X")
        krw_1m  = _pct_change("KRW=X", "1mo")
        krw_3m  = _pct_change("KRW=X", "3mo")
        if krw is not None:
            details["usd_krw"] = round(krw, 1)
        if krw_1m is not None:
            details["krw_1m_chg"] = round(krw_1m * 100, 1)
            scores.append(max(0, min(100, 50 - krw_1m * 200)))
        else:
            scores.append(50.0)
        if krw_3m is not None:
            details["krw_3m_chg"] = round(krw_3m * 100, 1)

        # ── 6. WTI 유가 ─────────────────────────────────────────────────────
        oil     = _safe_last("CL=F")
        oil_1m  = _pct_change("CL=F", "1mo")
        oil_3m  = _pct_change("CL=F", "3mo")
        if oil is not None:
            details["oil_wti"] = round(oil, 1)
        if oil_1m is not None:
            details["oil_1m_chg"] = round(oil_1m * 100, 1)
        if oil_3m is not None:
            details["oil_3m_chg"] = round(oil_3m * 100, 1)

        # ── 7. 달러 인덱스 (DXY) ────────────────────────────────────────────
        dxy_1m = _pct_change("DX-Y.NYB", "1mo")
        dxy_3m = _pct_change("DX-Y.NYB", "3mo")
        if dxy_1m is not None:
            details["dxy_1m_chg"] = round(dxy_1m * 100, 1)
            scores.append(max(0, min(100, 50 - dxy_1m * 250)))
        else:
            scores.append(50.0)
        if dxy_3m is not None:
            details["dxy_3m_chg"] = round(dxy_3m * 100, 1)

        # ── 8. 금 (안전자산 수요) ────────────────────────────────────────────
        gold_1m = _pct_change("GC=F", "1mo")
        gold_3m = _pct_change("GC=F", "3mo")
        if gold_1m is not None:
            details["gold_1m_chg"] = round(gold_1m * 100, 1)
            scores.append(max(0, min(100, 50 - gold_1m * 150)))
        else:
            scores.append(50.0)
        if gold_3m is not None:
            details["gold_3m_chg"] = round(gold_3m * 100, 1)

        # ── 9. 지정학 리스크 (시장 지표 역산 + 뉴스 보너스) ────────────────
        geo_score, geo_signals = _market_implied_geo_risk(
            vix=vix,
            gold_1m=gold_1m,
            gold_3m=gold_3m,
            oil_1m=oil_1m,
            oil_3m=oil_3m,
            krw_3m=krw_3m,
            kospi_1m=kospi_1m,
            sp500_1m=sp500_1m,
        )
        details["geopolitical_score"] = geo_score
        details["geopolitical_signals"] = geo_signals
        scores.append(geo_score)

        # ── 경기 사이클 ──────────────────────────────────────────────────────
        cycle = _detect_cycle(yield_spread, vix, sp500_1m)

        # ── 종합 점수 ────────────────────────────────────────────────────────
        base_score = sum(scores) / len(scores) if scores else 50.0
        sector_adj = _sector_adjustment(sector, us_10y, oil, oil_1m)
        final_score = max(0, min(100, base_score + sector_adj))

        details["cycle"] = cycle
        details["sector_adjustment"] = round(sector_adj, 1)
        details["indicators_fetched"] = sum(1 for s in scores if s != 50.0)

        return MacroResult(
            cycle=cycle,
            score=round(final_score, 1),
            details=details,
        )


# ── 지정학 리스크 (시장 역산) ────────────────────────────────────────────────

def _market_implied_geo_risk(
    vix:       float | None,
    gold_1m:   float | None,
    gold_3m:   float | None,
    oil_1m:    float | None,
    oil_3m:    float | None,
    krw_3m:    float | None,
    kospi_1m:  float | None,
    sp500_1m:  float | None,
) -> tuple[float, list[str]]:
    """시장 지표에서 역산한 지정학 리스크 점수 (0=위험, 100=안전)

    원리: 전쟁·제재·선거 충격 등 지정학 이벤트는 반드시 시장에 반영됩니다.
    금 급등 → 전쟁/위기, VIX 폭등 → 패닉, 유가 급등 → 중동·공급 위기,
    KRW 약세 3개월 → 신흥국 위험 프리미엄 상승, KOSPI vs S&P500 괴리 → 한국 지정학.
    """
    risk_pts = 0.0   # 누적 리스크 포인트 (높을수록 위험)
    signals: list[str] = []

    # ① 금 급등 → 안전자산 수요 = 전쟁·경제 위기
    if gold_3m is not None:
        if   gold_3m > 0.15:  risk_pts += 25; signals.append(f"금 3M +{gold_3m*100:.0f}% (위기 피난처 급매수)")
        elif gold_3m > 0.08:  risk_pts += 15; signals.append(f"금 3M +{gold_3m*100:.0f}% (안전자산 수요↑)")
        elif gold_3m > 0.04:  risk_pts +=  7; signals.append(f"금 3M +{gold_3m*100:.0f}% (완만한 불안)")
        elif gold_3m < -0.05: risk_pts -=  5; signals.append(f"금 3M {gold_3m*100:.0f}% (지정학 완화)")
    if gold_1m is not None:
        if   gold_1m > 0.08:  risk_pts += 15; signals.append(f"금 1M +{gold_1m*100:.0f}% (급격한 위기 반응)")
        elif gold_1m > 0.04:  risk_pts +=  8

    # ② VIX 수준 → 시장 공포 (전쟁·정치 충격 포함)
    if vix is not None:
        if   vix > 40:  risk_pts += 30; signals.append(f"VIX {vix:.0f} (공황 수준)")
        elif vix > 30:  risk_pts += 20; signals.append(f"VIX {vix:.0f} (극도 공포)")
        elif vix > 25:  risk_pts += 12; signals.append(f"VIX {vix:.0f} (높은 불안)")
        elif vix > 20:  risk_pts +=  6; signals.append(f"VIX {vix:.0f} (불안 구간)")
        elif vix < 13:  risk_pts -=  8; signals.append(f"VIX {vix:.0f} (평온)")

    # ③ 유가 급등 → 중동 분쟁·공급 충격 (or 수요 급증)
    if oil_3m is not None:
        if   oil_3m > 0.25:  risk_pts += 20; signals.append(f"WTI 3M +{oil_3m*100:.0f}% (공급 충격·분쟁)")
        elif oil_3m > 0.15:  risk_pts += 12; signals.append(f"WTI 3M +{oil_3m*100:.0f}% (공급 불안)")
        elif oil_3m > 0.08:  risk_pts +=  5
        elif oil_3m < -0.15: risk_pts -=  5; signals.append(f"WTI 3M {oil_3m*100:.0f}% (수요 우려)")

    # ④ KRW 3개월 약세 → 신흥국 위험 프리미엄 (무역·제재·불안)
    if krw_3m is not None:
        if   krw_3m > 0.08:  risk_pts += 18; signals.append(f"KRW 3M +{krw_3m*100:.0f}% (한국 위험↑·달러 강세)")
        elif krw_3m > 0.04:  risk_pts += 10; signals.append(f"KRW 3M +{krw_3m*100:.0f}% (원화 약세)")
        elif krw_3m > 0.02:  risk_pts +=  5
        elif krw_3m < -0.03: risk_pts -=  6; signals.append(f"KRW 3M {krw_3m*100:.0f}% (원화 강세·안정)")

    # ⑤ KOSPI vs S&P500 괴리 → 한국 특정 지정학 리스크 (북한·무역)
    if kospi_1m is not None and sp500_1m is not None:
        gap = kospi_1m - sp500_1m
        if   gap < -0.08:  risk_pts += 15; signals.append(f"KOSPI-S&P500 괴리 {gap*100:.0f}%p (한국 리스크)")
        elif gap < -0.04:  risk_pts +=  8; signals.append(f"KOSPI 상대 부진 {gap*100:.0f}%p")
        elif gap >  0.04:  risk_pts -=  5; signals.append(f"KOSPI 상대 강세 +{gap*100:.0f}%p")

    # ⑥ 뉴스 보너스 (yfinance 뉴스 — 실패 시 무시)
    news_adj = _news_risk_bonus()
    risk_pts += news_adj
    if news_adj > 5:
        signals.append(f"뉴스 위험 신호 +{news_adj:.0f}")
    elif news_adj < -3:
        signals.append(f"뉴스 안정 신호 {news_adj:.0f}")

    # 점수 변환: risk_pts 0=50점 / +30=20점 / -15=65점
    # risk_pts 범위: 약 -25 ~ +100
    score = max(5.0, min(95.0, 70.0 - risk_pts * 0.8))
    return round(score, 1), signals[:6]


def _news_risk_bonus() -> float:
    """뉴스 키워드 스캔 → 리스크 조정치 반환 (실패 시 0.0)"""
    tickers = ["^GSPC", "GC=F", "CL=F"]
    all_headlines: list[str] = []
    for sym in tickers:
        try:
            for item in (yf.Ticker(sym).news or [])[:10]:
                title = (item.get("title") or "").lower()
                if title:
                    all_headlines.append(title)
        except Exception:
            pass
        time.sleep(0.2)

    if not all_headlines:
        return 0.0

    high = sum(1 for h in all_headlines for kw in _GEO_HIGH_RISK if kw in h)
    med  = sum(1 for h in all_headlines for kw in _GEO_MED_RISK  if kw in h)
    return min(20.0, high * 3.0 + med * 1.0)


# ── 경기 사이클 ──────────────────────────────────────────────────────────────

def _detect_cycle(
    yield_spread: float | None,
    vix: float | None,
    sp500_trend: float | None,
) -> str:
    rec = 0
    exp = 0

    if yield_spread is not None:
        if   yield_spread < -0.5: rec += 2
        elif yield_spread < 0:    rec += 1
        else:                     exp += 1

    if vix is not None:
        if   vix > 30: rec += 2
        elif vix > 20: rec += 1
        elif vix < 15: exp += 2
        else:          exp += 1

    if sp500_trend is not None:
        if   sp500_trend < -0.05: rec += 1
        elif sp500_trend > 0.03:  exp += 1

    if   rec >= 3: return CYCLE_RECESSION
    elif rec >= 2: return CYCLE_SLOWDOWN
    elif exp >= 3: return CYCLE_EXPANSION
    else:          return CYCLE_RECOVERY


# ── 섹터 보정 ────────────────────────────────────────────────────────────────

def _sector_adjustment(
    sector: str | None,
    us_10y: float | None,
    oil: float | None,
    oil_1m: float | None,
) -> float:
    if not sector:
        return 0.0
    adj = 0.0
    if us_10y is not None:
        s = _RATE_SENSITIVITY.get(sector, 0.0)
        adj += s * (us_10y - 3.0) * (-3)
    if oil_1m is not None:
        s = _OIL_SENSITIVITY.get(sector, 0.0)
        adj += s * oil_1m * 30
    return max(-15, min(15, adj))
