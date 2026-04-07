"""시나리오 기반 미래 예측 엔진

현재 거시경제·지정학 이슈를 시장 데이터로 탐지하고,
각 시나리오(장기화 / 현상 유지 / 해소)별로 주가에 미치는
정량적 영향을 추정합니다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── 섹터별 시나리오 충격 계수 (%) ────────────────────────────────────────────
# yfinance sector 값 기준 (Technology, Financial Services, Consumer Cyclical 등)

_DEFAULT_IMPACT = 0.0   # 섹터 매핑 없을 때 기본값

_SECTOR_IMPACT: dict[str, dict[str, float]] = {

    # 지정학 악화 (전쟁 장기화·확전·제재 강화)
    "geo_escalate": {
        "Energy":                 +15.0,
        "Basic Materials":         +7.0,
        "Utilities":               +4.0,
        "Healthcare":              +2.0,
        "Consumer Defensive":      -3.0,
        "Consumer Cyclical":      -11.0,
        "Technology":              -9.0,
        "Financial Services":      -6.0,
        "Communication Services":  -5.0,
        "Industrials":             -7.0,
        "Real Estate":             -4.0,
    },

    # 지정학 완화 (휴전·협상 타결)
    "geo_ease": {
        "Energy":                  -9.0,
        "Basic Materials":         -5.0,
        "Utilities":               -2.0,
        "Healthcare":               0.0,
        "Consumer Defensive":      +3.0,
        "Consumer Cyclical":      +11.0,
        "Technology":              +9.0,
        "Financial Services":      +7.0,
        "Communication Services":  +7.0,
        "Industrials":             +9.0,
        "Real Estate":             +6.0,
    },

    # 금리 고착화 (Higher for Longer)
    "rate_hike": {
        "Financial Services":      +6.0,
        "Energy":                  +1.0,
        "Technology":             -13.0,
        "Consumer Cyclical":      -11.0,
        "Real Estate":            -16.0,
        "Utilities":              -13.0,
        "Healthcare":              -4.0,
        "Consumer Defensive":      -5.0,
        "Industrials":             -7.0,
        "Communication Services":  -9.0,
        "Basic Materials":         -5.0,
    },

    # 금리 인하·완화
    "rate_cut": {
        "Financial Services":      -4.0,
        "Energy":                  +2.0,
        "Technology":             +15.0,
        "Consumer Cyclical":      +13.0,
        "Real Estate":            +18.0,
        "Utilities":              +11.0,
        "Healthcare":              +5.0,
        "Consumer Defensive":      +4.0,
        "Industrials":            +10.0,
        "Communication Services": +12.0,
        "Basic Materials":         +7.0,
    },

    # 경기 침체 현실화
    "recession": {
        "Consumer Defensive":      +5.0,
        "Healthcare":              +6.0,
        "Utilities":               +3.0,
        "Energy":                  -8.0,
        "Technology":             -17.0,
        "Consumer Cyclical":      -23.0,
        "Financial Services":     -14.0,
        "Industrials":            -16.0,
        "Real Estate":            -12.0,
        "Communication Services": -10.0,
        "Basic Materials":        -11.0,
    },

    # 경기 반등·회복
    "recovery": {
        "Consumer Cyclical":      +17.0,
        "Financial Services":     +14.0,
        "Industrials":            +14.0,
        "Technology":             +12.0,
        "Energy":                 +10.0,
        "Real Estate":            +12.0,
        "Consumer Defensive":      +3.0,
        "Healthcare":              +5.0,
        "Utilities":               +2.0,
        "Communication Services": +10.0,
        "Basic Materials":        +11.0,
    },

    # 무역전쟁·관세 강화
    "trade_war": {
        "Technology":             -15.0,
        "Industrials":            -12.0,
        "Consumer Cyclical":      -10.0,
        "Energy":                  +3.0,
        "Consumer Defensive":      -6.0,
        "Financial Services":      -8.0,
        "Healthcare":              -2.0,
        "Utilities":               +2.0,
        "Real Estate":             -3.0,
        "Communication Services":  -7.0,
        "Basic Materials":         -9.0,
    },

    # 무역협상 타결
    "trade_ease": {
        "Technology":             +12.0,
        "Industrials":            +12.0,
        "Consumer Cyclical":      +10.0,
        "Financial Services":      +8.0,
        "Consumer Defensive":      +5.0,
        "Healthcare":              +3.0,
        "Utilities":               -1.0,
        "Real Estate":             +5.0,
        "Communication Services":  +8.0,
        "Basic Materials":         +9.0,
        "Energy":                  +2.0,
    },

    # 유가 급등 (공급 충격·중동 분쟁)
    "oil_spike": {
        "Energy":                 +18.0,
        "Basic Materials":         +4.0,
        "Industrials":             -8.0,
        "Consumer Cyclical":      -10.0,
        "Consumer Defensive":      -7.0,
        "Technology":              -5.0,
        "Financial Services":      -6.0,
        "Utilities":               -7.0,
        "Real Estate":             -4.0,
        "Communication Services":  -5.0,
        "Healthcare":              -3.0,
    },

    # 유가 급락 (수요 감소·공급 확대)
    "oil_drop": {
        "Energy":                 -14.0,
        "Basic Materials":         -3.0,
        "Industrials":             +6.0,
        "Consumer Cyclical":       +9.0,
        "Consumer Defensive":      +6.0,
        "Technology":              +4.0,
        "Financial Services":      +4.0,
        "Utilities":               +6.0,
        "Real Estate":             +5.0,
        "Communication Services":  +4.0,
        "Healthcare":              +3.0,
    },
}


@dataclass
class Scenario:
    name: str
    probability: float          # 0.0 ~ 1.0
    time_horizon: str           # "3개월 이내" / "3~6개월" / "6개월~1년"
    price_impact_pct: float     # 현재가 대비 예상 등락(%)
    projected_price: float      # 예상 주가
    key_drivers: list[str]
    description: str
    sentiment: str              # "bearish" / "neutral" / "bullish"


@dataclass
class Issue:
    name: str
    emoji: str
    category: str               # "geopolitical" / "monetary" / "market" / "trade"
    severity: str               # "HIGH" / "MEDIUM"
    signal: str                 # 탐지 근거 요약
    scenarios: list[Scenario]


class ScenarioEngine:
    """현재 거시 이슈 탐지 + 주식별 시나리오 주가 영향 계산"""

    # ── 공개 API ──────────────────────────────────────────────────────────────

    def analyze(
        self,
        current_price: float,
        sector: str,
        beta: float,
        macro_details: dict[str, Any],
    ) -> list[Issue]:
        """탐지된 이슈 목록 반환. 각 이슈는 2~3개 시나리오를 포함."""
        issues: list[Issue] = []
        b = max(beta or 1.0, 0.3)   # 베타 최솟값 보정

        d = macro_details  # 편의 alias

        vix        = d.get("vix", 20.0) or 20.0
        us_10y     = d.get("us_10y_yield", 4.0) or 4.0
        sp500_1m   = d.get("sp500_1m", 0.0) or 0.0
        sp500_3m   = d.get("sp500_3m", 0.0) or 0.0
        gold_1m    = d.get("gold_1m_chg", 0.0) or 0.0
        gold_3m    = d.get("gold_3m_chg", 0.0) or 0.0
        oil_1m     = d.get("oil_1m_chg", 0.0) or 0.0
        oil_3m     = d.get("oil_3m_chg", 0.0) or 0.0
        krw_1m     = d.get("krw_1m_chg", 0.0) or 0.0
        krw_3m     = d.get("krw_3m_chg", 0.0) or 0.0
        dxy_1m     = d.get("dxy_1m_chg", 0.0) or 0.0
        dxy_3m     = d.get("dxy_3m_chg", 0.0) or 0.0
        cycle      = d.get("cycle", "Expansion")
        geo_score  = d.get("geopolitical_score", 50.0) or 50.0

        # ── 이슈 탐지 ────────────────────────────────────────────────────────

        # 1. 지정학 리스크
        issues += self._geo_issue(current_price, sector, b, vix, gold_1m, gold_3m,
                                  oil_1m, oil_3m, krw_3m, sp500_1m, geo_score)

        # 2. 금리 리스크
        issues += self._rate_issue(current_price, sector, b, us_10y)

        # 3. 무역전쟁·관세 리스크
        issues += self._trade_issue(current_price, sector, b, dxy_1m, dxy_3m, sp500_1m)

        # 4. 경기 사이클 리스크
        issues += self._cycle_issue(current_price, sector, b, cycle, sp500_1m, sp500_3m, vix)

        # 5. 유가 충격 (에너지/소비재 직접 영향)
        issues += self._oil_issue(current_price, sector, b, oil_1m, oil_3m)

        return issues

    # ── 이슈 탐지 메서드 ──────────────────────────────────────────────────────

    def _geo_issue(self, cp, sec, b, vix, gold_1m, gold_3m, oil_1m, oil_3m,
                   krw_3m, sp500_1m, geo_score) -> list[Issue]:
        # 지정학 점수 역산: 낮을수록 위험 (0=위험, 100=안전)
        risk_lvl = 100 - geo_score   # 높을수록 위험

        # 탐지 기준: 지정학 위험 30 이상 or 금 1M +4% 이상
        if risk_lvl < 30 and gold_1m < 4:
            return []

        severity = "HIGH" if risk_lvl > 50 else "MEDIUM"
        signals = []
        if gold_3m > 5:  signals.append(f"금 3M +{gold_3m:.1f}%")
        if gold_1m > 3:  signals.append(f"금 1M +{gold_1m:.1f}%")
        if vix > 22:     signals.append(f"VIX {vix:.0f}")
        if oil_3m > 8:   signals.append(f"유가 3M +{oil_3m:.1f}%")
        if krw_3m > 3:   signals.append(f"원화약세 3M +{krw_3m:.1f}%")
        signal = " | ".join(signals) if signals else f"지정학 리스크 지수 {risk_lvl:.0f}"

        esc  = self._impact("geo_escalate", sec, b)
        ease = self._impact("geo_ease",     sec, b)

        return [Issue(
            name="지정학적 리스크 고조",
            emoji="⚔️",
            category="geopolitical",
            severity=severity,
            signal=signal,
            scenarios=[
                Scenario(
                    name="분쟁 확전·장기화",
                    probability=0.25,
                    time_horizon="6개월~1년",
                    price_impact_pct=esc * 1.5,
                    projected_price=cp * (1 + esc * 1.5 / 100),
                    key_drivers=["유가·금 추가 급등", "안전자산 선호 심화", "글로벌 공급망 교란", "인플레 재점화"],
                    description="분쟁이 확대·장기화되며 에너지·식량 공급 차질, 인플레 재상승과 글로벌 성장 둔화 우려",
                    sentiment="bearish" if esc < 0 else "bullish",
                ),
                Scenario(
                    name="교착 상태 지속",
                    probability=0.50,
                    time_horizon="3~6개월",
                    price_impact_pct=esc * 0.4,
                    projected_price=cp * (1 + esc * 0.4 / 100),
                    key_drivers=["지정학 프리미엄 현수준 유지", "시장 내성 점차 형성", "간헐적 긴장 고조"],
                    description="분쟁이 이어지지만 시장은 점차 적응하며 지정학 프리미엄은 현 수준에서 횡보",
                    sentiment="neutral",
                ),
                Scenario(
                    name="협상·휴전 타결",
                    probability=0.25,
                    time_horizon="3개월 이내",
                    price_impact_pct=ease,
                    projected_price=cp * (1 + ease / 100),
                    key_drivers=["리스크 프리미엄 급감", "안전자산 매도·위험자산 매수", "글로벌 교역 회복 기대"],
                    description="협상 타결 시 금·유가 급락, 위험자산 선호 급반전하며 주식 시장 전반 상승",
                    sentiment="bullish" if ease > 0 else "bearish",
                ),
            ]
        )]

    def _rate_issue(self, cp, sec, b, us_10y) -> list[Issue]:
        if us_10y < 4.2:
            return []

        severity = "HIGH" if us_10y > 4.8 else "MEDIUM"
        hike = self._impact("rate_hike", sec, b)
        cut  = self._impact("rate_cut",  sec, b)

        return [Issue(
            name="고금리 지속 리스크",
            emoji="📈",
            category="monetary",
            severity=severity,
            signal=f"미국 10Y 국채 {us_10y:.2f}% (중립 수준 대비 고점)",
            scenarios=[
                Scenario(
                    name="금리 고착화 (Higher for Longer)",
                    probability=0.35,
                    time_horizon="6개월~1년",
                    price_impact_pct=hike,
                    projected_price=cp * (1 + hike / 100),
                    key_drivers=["인플레 재점화", "Fed 긴축 장기화", "할인율 상승으로 성장주 밸류에이션 압박"],
                    description="인플레가 끈질기게 이어져 Fed가 금리를 오래 유지 — PER 높은 성장주와 부채 많은 기업 타격",
                    sentiment="bearish" if hike < 0 else "bullish",
                ),
                Scenario(
                    name="금리 동결·관망",
                    probability=0.40,
                    time_horizon="3~6개월",
                    price_impact_pct=hike * 0.3,
                    projected_price=cp * (1 + hike * 0.3 / 100),
                    key_drivers=["인플레 둔화 확인 대기", "고용 지표 주시", "불확실성 지속"],
                    description="Fed가 인플레와 성장 사이에서 관망 — 추가 충격은 제한되지만 불확실성으로 횡보",
                    sentiment="neutral",
                ),
                Scenario(
                    name="예상보다 이른 금리 인하",
                    probability=0.25,
                    time_horizon="3개월 이내",
                    price_impact_pct=cut,
                    projected_price=cp * (1 + cut / 100),
                    key_drivers=["인플레 빠른 둔화", "성장 우려 부각", "금리 민감 섹터 급등"],
                    description="인플레가 빠르게 잡히며 조기 인하 단행 — 성장주·부동산·유틸리티 수혜",
                    sentiment="bullish" if cut > 0 else "bearish",
                ),
            ]
        )]

    def _trade_issue(self, cp, sec, b, dxy_1m, dxy_3m, sp500_1m) -> list[Issue]:
        # 달러 강세(관세 우려) + 주가 하락이 겹칠 때
        if dxy_3m < 3 and dxy_1m < 2:
            return []

        war  = self._impact("trade_war",  sec, b)
        ease = self._impact("trade_ease", sec, b)

        return [Issue(
            name="무역분쟁·관세 리스크",
            emoji="🛃",
            category="trade",
            severity="HIGH" if dxy_3m > 6 else "MEDIUM",
            signal=f"DXY 3M +{dxy_3m:.1f}% | DXY 1M +{dxy_1m:.1f}%",
            scenarios=[
                Scenario(
                    name="관세 전면 확대",
                    probability=0.30,
                    time_horizon="6개월",
                    price_impact_pct=war,
                    projected_price=cp * (1 + war / 100),
                    key_drivers=["추가 관세 부과", "보복 관세 연쇄", "글로벌 공급망 재편 비용 급증"],
                    description="관세 확대로 글로벌 교역 위축 — 수출 의존 기업·반도체·자동차 등 타격",
                    sentiment="bearish" if war < 0 else "bullish",
                ),
                Scenario(
                    name="협상 진행·부분 완화",
                    probability=0.45,
                    time_horizon="3~6개월",
                    price_impact_pct=ease * 0.4,
                    projected_price=cp * (1 + ease * 0.4 / 100),
                    key_drivers=["협상 진전", "일부 품목 관세 면제", "불확실성 잔존"],
                    description="협상이 진행되며 부분 완화 — 최악은 회피하나 완전 해소까지 불확실성 지속",
                    sentiment="neutral",
                ),
                Scenario(
                    name="포괄 합의·관세 철폐",
                    probability=0.25,
                    time_horizon="6개월 이내",
                    price_impact_pct=ease,
                    projected_price=cp * (1 + ease / 100),
                    key_drivers=["포괄적 무역 합의", "공급망 불확실성 해소", "글로벌 교역 정상화"],
                    description="포괄 합의로 관세 철폐 — 글로벌 교역 정상화, 제조·기술·소비재 급반등",
                    sentiment="bullish" if ease > 0 else "bearish",
                ),
            ]
        )]

    def _cycle_issue(self, cp, sec, b, cycle, sp500_1m, sp500_3m, vix) -> list[Issue]:
        if cycle not in ("Slowdown", "Recession") and sp500_3m > -5:
            return []

        severity = "HIGH" if cycle == "Recession" else "MEDIUM"
        rec  = self._impact("recession", sec, b)
        recov= self._impact("recovery",  sec, b)

        return [Issue(
            name="경기 침체 리스크",
            emoji="📉",
            category="market",
            severity=severity,
            signal=f"경기 사이클 {cycle} | S&P500 3M {sp500_3m:.1f}% | VIX {vix:.0f}",
            scenarios=[
                Scenario(
                    name="경기 침체 현실화",
                    probability=0.30,
                    time_horizon="6개월~1년",
                    price_impact_pct=rec,
                    projected_price=cp * (1 + rec / 100),
                    key_drivers=["소비·투자 급감", "기업 이익 하락", "실업률 상승", "신용 리스크 부각"],
                    description="본격 침체로 기업 매출·이익 감소 — 경기 민감 섹터 타격, 방어주 상대적 강세",
                    sentiment="bearish" if rec < 0 else "bullish",
                ),
                Scenario(
                    name="연착륙 (Soft Landing)",
                    probability=0.50,
                    time_horizon="3~6개월",
                    price_impact_pct=rec * 0.25,
                    projected_price=cp * (1 + rec * 0.25 / 100),
                    key_drivers=["성장 둔화 but 침체 회피", "기업 실적 소폭 감소", "정책 부양 기대"],
                    description="성장은 둔화되나 침체를 피하는 연착륙 — 기업 실적 소폭 감소, 시장 횡보",
                    sentiment="neutral",
                ),
                Scenario(
                    name="경기 반등·V자 회복",
                    probability=0.20,
                    time_horizon="3개월 이내",
                    price_impact_pct=recov,
                    projected_price=cp * (1 + recov / 100),
                    key_drivers=["예상 밖 정책 부양", "소비 서프라이즈", "기업 실적 상향"],
                    description="예상보다 강한 경기 회복 — 경기 민감 섹터 주도로 V자 반등",
                    sentiment="bullish" if recov > 0 else "bearish",
                ),
            ]
        )]

    def _oil_issue(self, cp, sec, b, oil_1m, oil_3m) -> list[Issue]:
        if abs(oil_3m) < 10 and abs(oil_1m) < 6:
            return []

        spike = self._impact("oil_spike", sec, b)
        drop  = self._impact("oil_drop",  sec, b)
        is_spike = oil_3m > 0

        return [Issue(
            name="유가 급등 리스크" if is_spike else "유가 급락 리스크",
            emoji="🛢️",
            category="geopolitical",
            severity="HIGH" if abs(oil_3m) > 20 else "MEDIUM",
            signal=f"WTI 3M {oil_3m:+.1f}% | 1M {oil_1m:+.1f}%",
            scenarios=[
                Scenario(
                    name="유가 추가 상승 (공급 충격 심화)" if is_spike else "유가 추가 하락 (수요 붕괴)",
                    probability=0.30,
                    time_horizon="3~6개월",
                    price_impact_pct=spike if is_spike else drop,
                    projected_price=cp * (1 + (spike if is_spike else drop) / 100),
                    key_drivers=["OPEC 추가 감산", "중동 공급 차질"] if is_spike else ["글로벌 경기 둔화", "미국 증산"],
                    description="공급 충격 심화로 유가 추가 상승 → 비용 인플레 재점화" if is_spike
                               else "수요 감소로 유가 추가 하락 → 에너지 섹터 이익 급감",
                    sentiment="bearish" if (is_spike and spike < 0) or (not is_spike and drop < 0) else "bullish",
                ),
                Scenario(
                    name="현 수준 안정화",
                    probability=0.45,
                    time_horizon="3~6개월",
                    price_impact_pct=(spike if is_spike else drop) * 0.3,
                    projected_price=cp * (1 + (spike if is_spike else drop) * 0.3 / 100),
                    key_drivers=["공급·수요 균형 회복", "전략비축유 방출", "OPEC 협조"],
                    description="유가가 현 수준에서 안정화 — 시장은 적응하며 추가 충격 제한",
                    sentiment="neutral",
                ),
                Scenario(
                    name="유가 급반전 (수요 회복)" if is_spike else "유가 반등 (공급 축소)",
                    probability=0.25,
                    time_horizon="3개월 이내",
                    price_impact_pct=drop if is_spike else spike,
                    projected_price=cp * (1 + (drop if is_spike else spike) / 100),
                    key_drivers=["수요 감소·미국 증산"] if is_spike else ["OPEC 감산", "지정학 긴장"],
                    description="유가 급락 → 소비 회복 수혜, 에너지 비용 절감" if is_spike
                               else "유가 급반등 → 에너지 섹터 회복, 비용 인플레 우려",
                    sentiment="bullish" if (is_spike and drop > 0) or (not is_spike and spike > 0) else "bearish",
                ),
            ]
        )]

    # ── 내부 헬퍼 ────────────────────────────────────────────────────────────

    def _impact(self, issue_type: str, sector: str, beta: float) -> float:
        """섹터 + 베타 보정 주가 영향(%) 계산"""
        m = _SECTOR_IMPACT.get(issue_type, {})
        base = m.get(sector, _DEFAULT_IMPACT)
        if base == 0.0:
            # 섹터 매핑 없으면 전체 평균의 절반으로 fallback
            vals = list(m.values())
            base = sum(vals) / len(vals) * 0.5 if vals else 0.0
        return round(base * beta, 2)
