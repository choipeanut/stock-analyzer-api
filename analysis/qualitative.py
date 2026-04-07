"""정성적 분석 모듈 — 재무 품질(pre-fetched info) + 뉴스 감성 보너스"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import yfinance as yf

logger = logging.getLogger(__name__)

# 긍정 키워드
_POSITIVE = [
    "beat", "surpassed", "record", "growth", "profit", "strong", "upgrade",
    "raised guidance", "expansion", "breakthrough", "partnership", "contract",
    "dividend increase", "buyback", "innovation", "rally", "surge", "award",
    "실적 호조", "매출 성장", "흑자", "호실적", "수주", "계약", "배당", "자사주",
]

# 부정 키워드
_NEGATIVE = [
    "miss", "missed", "loss", "decline", "fell", "cut", "downgrade", "warning",
    "investigation", "lawsuit", "fraud", "bankruptcy", "layoff", "recall",
    "guidance cut", "shortfall", "scandal", "slump", "plunge",
    "실적 부진", "적자", "손실", "소송", "리콜", "구조조정", "하향", "불법",
]


@dataclass
class QualitativeResult:
    management_score: float = 5.0
    business_model_score: float = 5.0
    esg_score: float = 5.0
    news_sentiment: float = 0.0
    total_score: float = 50.0
    summary: str = ""
    details: dict[str, Any] = field(default_factory=dict)


class QualitativeAnalyzer:
    """정성적 분석 — 재무 품질 70% + 뉴스 감성 30%

    run_analysis()에서 이미 가져온 info를 재사용해
    추가 API 호출 없이 채점하고, 뉴스는 보너스 신호로만 활용합니다.
    """

    def analyze(
        self,
        ticker: str,
        info: dict[str, Any] | None = None,
        news: list[str] | None = None,
    ) -> QualitativeResult:
        details: dict[str, Any] = {}

        # 재무 품질 — info가 없으면 yfinance로 직접 시도
        if not info:
            try:
                info = yf.Ticker(ticker).info or {}
            except Exception:
                info = {}

        fin_score  = self._financial_quality(info, details)
        news_score = self._news_sentiment(ticker, details)

        # 재무 품질 70%, 뉴스 30%
        total = fin_score * 0.7 + news_score * 0.3
        total = max(0.0, min(100.0, total))

        if total >= 70:
            summary = "재무 품질 및 시장 평판이 전반적으로 우수합니다."
        elif total >= 55:
            summary = "전반적으로 양호한 정성적 환경입니다."
        elif total >= 40:
            summary = "중립적인 정성적 환경입니다."
        else:
            summary = "재무 품질 저하 또는 부정적 뉴스가 감지됩니다."

        details["financial_quality_score"] = round(fin_score, 1)
        details["news_score"] = round(news_score, 1)

        return QualitativeResult(
            management_score=round(fin_score / 10, 1),
            business_model_score=round(fin_score / 10, 1),
            news_sentiment=round((news_score - 50) / 50, 2),
            total_score=round(total, 1),
            summary=summary,
            details=details,
        )

    # ── 내부 메서드 ──────────────────────────────────────────────────────────

    def _financial_quality(self, info: dict, details: dict) -> float:
        """ROE·마진·성장·부채·FCF 기반 재무 품질 (pre-fetched info 활용)"""
        scores: list[float] = []

        # ROE
        roe = info.get("returnOnEquity")
        if roe is not None:
            details["roe_pct"] = round(float(roe) * 100, 1)
            scores.append(85 if roe > 0.20 else 70 if roe > 0.10 else 50 if roe > 0 else 20)

        # 영업이익률
        om = info.get("operatingMargins")
        if om is not None:
            details["op_margin_pct"] = round(float(om) * 100, 1)
            scores.append(85 if om > 0.20 else 70 if om > 0.10 else 50 if om > 0 else 20)

        # 순이익률
        pm = info.get("profitMargins")
        if pm is not None:
            details["profit_margin_pct"] = round(float(pm) * 100, 1)
            scores.append(80 if pm > 0.15 else 65 if pm > 0.05 else 50 if pm > 0 else 25)

        # 매출 성장률
        rg = info.get("revenueGrowth")
        if rg is not None:
            details["rev_growth_pct"] = round(float(rg) * 100, 1)
            scores.append(85 if rg > 0.20 else 70 if rg > 0.10 else 55 if rg > 0 else 30)

        # 부채비율 (낮을수록 좋음)
        de = info.get("debtToEquity")
        if de is not None:
            details["debt_to_equity"] = round(float(de), 1)
            scores.append(85 if de < 30 else 70 if de < 100 else 45 if de < 200 else 25)

        # FCF 양수 여부
        fcf = info.get("freeCashflow")
        if fcf is not None:
            details["fcf_positive"] = fcf > 0
            scores.append(72 if fcf > 0 else 28)

        # 현금 비율 (유동성)
        cr = info.get("currentRatio")
        if cr is not None:
            details["current_ratio"] = round(float(cr), 2)
            scores.append(80 if cr > 2.0 else 65 if cr > 1.5 else 55 if cr > 1.0 else 30)

        if not scores:
            details["note"] = "재무 데이터 없음 (yfinance 제공 안 함)"
            return 50.0

        return sum(scores) / len(scores)

    def _news_sentiment(self, ticker: str, details: dict) -> float:
        """yfinance 뉴스 헤드라인 감성 (실패해도 중립 50 반환)"""
        try:
            items = yf.Ticker(ticker).news or []
            headlines = [
                (it.get("title") or "").lower()
                for it in items[:20]
                if it.get("title")
            ]
        except Exception:
            details["news_count"] = 0
            return 50.0

        if not headlines:
            details["news_count"] = 0
            return 50.0

        pos = sum(1 for h in headlines for kw in _POSITIVE if kw in h)
        neg = sum(1 for h in headlines for kw in _NEGATIVE if kw in h)

        details["news_count"] = len(headlines)
        details["news_positive"] = pos
        details["news_negative"] = neg

        score = 50.0 + pos * 5.0 - neg * 5.0
        return max(15.0, min(90.0, score))
