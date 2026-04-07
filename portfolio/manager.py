"""모의 투자 포트폴리오 매니저

모든 금액은 내부적으로 KRW로 통일합니다.
외화(USD 등) 종목은 거래 시점 환율을 적용해 KRW로 변환하며,
원래 단가(price_native)와 통화(currency), 환율(fx_rate)은
거래 내역 조회용으로 함께 저장합니다.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

PORTFOLIO_FILE = Path.home() / ".stock_analyzer_portfolio.json"

BUY_FEE_RATE  = 0.00015   # 매수 수수료 0.015%
SELL_FEE_RATE = 0.00015   # 매도 수수료 0.015%
SELL_TAX_RATE = 0.0020    # 증권거래세 0.2%

USD_MARKETS = {"NASDAQ", "NYSE", "SP500", "AMEX"}


def is_usd_market(market: str) -> bool:
    return market.upper() in USD_MARKETS


class PortfolioManager:
    """세션별 모의 투자 포트폴리오 (다통화 지원, 평가는 KRW)

    data 파라미터를 넘기면 해당 dict를 직접 사용 (세션 격리 모드, 파일 저장 없음).
    data=None 이면 파일에서 로드 (하위 호환).
    """

    def __init__(self, data: dict[str, Any] | None = None):
        if data is not None:
            self.data = data          # 외부 dict 참조 (session_state)
            self._file_mode = False
        else:
            self.data = self._load()  # 파일 로드 (단독 실행 등)
            self._file_mode = True

    # ── 로드 / 저장 ──────────────────────────────────────────────────────────

    def _load(self) -> dict[str, Any]:
        try:
            if PORTFOLIO_FILE.exists():
                with open(PORTFOLIO_FILE, encoding="utf-8") as f:
                    return json.load(f)
        except Exception:
            pass
        return {
            "cash": 10_000_000,
            "initial_cash": 10_000_000,
            "transactions": [],
            "snapshots": [],
        }

    def _save(self) -> None:
        if not self._file_mode:
            return   # 세션 모드: 파일 저장 불필요, session_state에 이미 반영됨
        try:
            with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
                json.dump(self.data, f, ensure_ascii=False, indent=2, default=str)
        except Exception:
            pass

    # ── 매수 ─────────────────────────────────────────────────────────────────

    def buy(
        self,
        ticker: str,
        market: str,
        name: str,
        price_native: float,    # 원화: 원 단위 / 달러: 달러 단위
        quantity: int,
        currency: str = "KRW",  # "KRW" or "USD"
        fx_rate: float = 1.0,   # USD/KRW 환율 (KRW 종목이면 1.0)
        note: str = "",
    ) -> dict[str, Any]:
        price_krw = price_native * fx_rate          # 내부 KRW 단가
        fee       = round(price_krw * quantity * BUY_FEE_RATE)
        total_out = round(price_krw * quantity + fee)

        if total_out > self.data["cash"]:
            raise ValueError(
                f"잔액 부족 — 필요: {total_out:,.0f}원  보유: {self.data['cash']:,.0f}원"
            )

        tx: dict[str, Any] = {
            "id":           str(uuid.uuid4())[:8],
            "type":         "buy",
            "ticker":       ticker,
            "market":       market,
            "name":         name,
            "currency":     currency,
            "price_native": round(price_native, 4),   # 표시용 원래 통화 단가
            "price":        round(price_krw, 2),       # 내부 KRW 단가
            "fx_rate":      round(fx_rate, 2),
            "quantity":     quantity,
            "fee":          fee,
            "tax":          0,
            "total":        total_out,
            "date":         datetime.now().strftime("%Y-%m-%d %H:%M"),
            "note":         note,
        }
        self.data["transactions"].append(tx)
        self.data["cash"] = round(self.data["cash"] - total_out)
        self._save()
        return tx

    # ── 매도 ─────────────────────────────────────────────────────────────────

    def sell(
        self,
        ticker: str,
        market: str,
        name: str,
        price_native: float,
        quantity: int,
        currency: str = "KRW",
        fx_rate: float = 1.0,
        note: str = "",
    ) -> dict[str, Any]:
        holdings = self.get_holdings()
        h = holdings.get(ticker)
        if not h or h["quantity"] < quantity:
            avail = h["quantity"] if h else 0
            raise ValueError(f"보유 수량 부족 — 보유: {avail}주, 매도 요청: {quantity}주")

        price_krw = price_native * fx_rate
        fee       = round(price_krw * quantity * SELL_FEE_RATE)
        tax       = round(price_krw * quantity * SELL_TAX_RATE)
        net_in    = round(price_krw * quantity - fee - tax)
        realized  = round((price_krw - h["avg_price_krw"]) * quantity - fee - tax)

        tx: dict[str, Any] = {
            "id":            str(uuid.uuid4())[:8],
            "type":          "sell",
            "ticker":        ticker,
            "market":        market,
            "name":          name,
            "currency":      currency,
            "price_native":  round(price_native, 4),
            "price":         round(price_krw, 2),
            "fx_rate":       round(fx_rate, 2),
            "quantity":      quantity,
            "fee":           fee,
            "tax":           tax,
            "total":         net_in,
            "avg_price_krw": round(h["avg_price_krw"], 2),
            "realized_pl":   realized,
            "date":          datetime.now().strftime("%Y-%m-%d %H:%M"),
            "note":          note,
        }
        self.data["transactions"].append(tx)
        self.data["cash"] = round(self.data["cash"] + net_in)
        self._save()
        return tx

    # ── 보유 현황 ─────────────────────────────────────────────────────────────

    def get_holdings(self) -> dict[str, dict[str, Any]]:
        """보유 종목 집계 — 가중평균 매입단가 (KRW + 원화 통화 모두 추적)"""
        holdings: dict[str, dict[str, Any]] = {}

        for tx in self.data["transactions"]:
            ticker   = tx["ticker"]
            qty      = tx["quantity"]
            currency = tx.get("currency", "KRW")

            if tx["type"] == "buy":
                if ticker not in holdings:
                    holdings[ticker] = {
                        "ticker":           ticker,
                        "market":           tx["market"],
                        "name":             tx["name"],
                        "currency":         currency,
                        "quantity":         0,
                        "total_cost_krw":   0.0,
                        "total_native":     0.0,
                        "avg_price_krw":    0.0,
                        "avg_price_native": 0.0,
                        "avg_price":        0.0,   # alias for KRW (backward compat)
                        "total_cost":       0.0,
                    }
                h = holdings[ticker]
                cost_krw    = tx["price"] * qty + tx.get("fee", 0)
                cost_native = tx.get("price_native", tx["price"]) * qty
                h["total_cost_krw"]  += cost_krw
                h["total_native"]    += cost_native
                h["quantity"]        += qty
                h["avg_price_krw"]    = h["total_cost_krw"]  / h["quantity"]
                h["avg_price_native"] = h["total_native"]     / h["quantity"]
                h["avg_price"]        = h["avg_price_krw"]
                h["total_cost"]       = h["total_cost_krw"]

            elif tx["type"] == "sell" and ticker in holdings:
                h = holdings[ticker]
                reduce_krw    = h["avg_price_krw"]    * qty
                reduce_native = h["avg_price_native"]  * qty
                h["total_cost_krw"]  -= reduce_krw
                h["total_native"]    -= reduce_native
                h["quantity"]        -= qty
                if h["quantity"] <= 0:
                    del holdings[ticker]
                else:
                    h["avg_price_krw"]    = h["total_cost_krw"]  / h["quantity"]
                    h["avg_price_native"] = h["total_native"]     / h["quantity"]
                    h["avg_price"]        = h["avg_price_krw"]

        return {k: v for k, v in holdings.items() if v.get("quantity", 0) > 0}

    # ── 실현 손익 ─────────────────────────────────────────────────────────────

    def realized_pl(self) -> float:
        return sum(
            tx.get("realized_pl", 0)
            for tx in self.data["transactions"]
            if tx["type"] == "sell"
        )

    # ── 스냅샷 ───────────────────────────────────────────────────────────────

    def save_snapshot(self, total_value: float) -> None:
        today = datetime.now().strftime("%Y-%m-%d")
        snaps = self.data.setdefault("snapshots", [])
        if snaps and snaps[-1]["date"] == today:
            snaps[-1]["value"] = total_value
        else:
            snaps.append({"date": today, "value": total_value})
        self._save()

    # ── 관리 ─────────────────────────────────────────────────────────────────

    def reset(self, initial_cash: float = 10_000_000) -> None:
        self.data.clear()
        self.data.update({
            "cash": initial_cash,
            "initial_cash": initial_cash,
            "transactions": [],
            "snapshots": [],
        })
        self._save()

    def add_cash(self, amount: float) -> None:
        self.data["cash"]         = round(self.data["cash"] + amount)
        self.data["initial_cash"] = round(self.data.get("initial_cash", 0) + amount)
        self._save()

    # ── 속성 ─────────────────────────────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self.data["cash"]

    @property
    def initial_cash(self) -> float:
        return self.data.get("initial_cash", 10_000_000)

    @property
    def transactions(self) -> list[dict[str, Any]]:
        return self.data.get("transactions", [])

    @property
    def snapshots(self) -> list[dict[str, Any]]:
        return self.data.get("snapshots", [])

    def export_json(self) -> str:
        return json.dumps(self.data, ensure_ascii=False, indent=2, default=str)

    def import_json(self, raw: str) -> None:
        loaded = json.loads(raw)
        self.data.clear()
        self.data.update(loaded)
        self._save()
