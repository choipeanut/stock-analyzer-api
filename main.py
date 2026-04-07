"""FastAPI 백엔드 — 기존 stock_analyzer 패키지를 REST API로 노출

실행:
    uvicorn main:app --reload --port 8000

Flutter 앱이 이 서버에 HTTP 요청을 보냅니다.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import traceback

app = FastAPI(title="Stock Analyzer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 요청/응답 모델 ──────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker: str
    market: str = "KRX"
    mode: str = "default"
    period: str = "1y"

class ScreenRequest(BaseModel):
    market: str = "KOSPI"
    mode: str = "default"
    top_n: int = 20
    min_score: float = 40.0

class BuyRequest(BaseModel):
    session_id: str
    ticker: str
    market: str
    name: str
    price_native: float
    quantity: int
    currency: str = "KRW"
    fx_rate: float = 1.0
    note: str = ""

class SellRequest(BaseModel):
    session_id: str
    ticker: str
    market: str
    name: str
    price_native: float
    quantity: int
    currency: str = "KRW"
    fx_rate: float = 1.0
    note: str = ""

# ── 인메모리 포트폴리오 (세션별) ───────────────────────────────────────────────
_portfolios: dict = {}

def _get_portfolio(session_id: str):
    from portfolio.manager import PortfolioManager
    if session_id not in _portfolios:
        _portfolios[session_id] = {}
    pm = PortfolioManager(data=_portfolios[session_id])
    if not _portfolios[session_id]:
        _portfolios[session_id].update(pm.data)
    return pm

# ── 헬퍼 ───────────────────────────────────────────────────────────────────────

def _krx_to_yf(ticker: str, market: str) -> str:
    market = market.upper()
    if market in ("KRX", "KOSPI") and ticker.isdigit():
        return ticker + ".KS"
    if market == "KOSDAQ" and ticker.isdigit():
        return ticker + ".KQ"
    return ticker

# ── 엔드포인트 ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    try:
        from data.collectors.yfinance_client import YFinanceClient
        from data.processors.data_processor import DataProcessor
        from analysis.fundamental import FundamentalAnalyzer
        from analysis.technical import TechnicalAnalyzer
        from analysis.macro import MacroAnalyzer
        from analysis.industry import IndustryAnalyzer
        from analysis.qualitative import QualitativeAnalyzer
        from analysis.risk import RiskAnalyzer
        from analysis.scenario import ScenarioEngine
        from scoring.engine import ScoringEngine
        from scoring.recommender import Recommender
        import yaml, pathlib

        weights_path = pathlib.Path(__file__).parent / "../../stock/stock_analyzer/config/weights.yaml"
        with open(weights_path) as f:
            weights = yaml.safe_load(f)

        yf_ticker = _krx_to_yf(req.ticker, req.market)
        client = YFinanceClient()
        info = client.get_ticker_info(yf_ticker) or {}
        price_df = client.get_price_history(yf_ticker, period=req.period)

        processor = DataProcessor()
        price_df_feat = processor.add_features(price_df)
        metrics = processor.extract_financial_metrics(info, price_df)

        current_price = None
        try:
            close = price_df["Close"]
            if hasattr(close, "iloc"):
                v = close.dropna().iloc[-1]
                current_price = float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
        except Exception:
            current_price = info.get("regularMarketPrice") or info.get("currentPrice")

        sector = info.get("sector", "")
        f_res  = FundamentalAnalyzer().analyze(yf_ticker, info=info, metrics=metrics)
        t_res  = TechnicalAnalyzer().analyze(yf_ticker, price_df=price_df_feat)
        macro  = MacroAnalyzer().analyze(sector)
        ind    = IndustryAnalyzer().analyze(yf_ticker, sector=sector, info=info, price_df=price_df)
        qual   = QualitativeAnalyzer().analyze(yf_ticker, info=info)
        risk   = RiskAnalyzer().analyze(yf_ticker, price_df_feat, metrics)

        comp = ScoringEngine(weights=weights, mode=req.mode).score(
            yf_ticker, fundamental=f_res, technical=t_res,
            macro=macro, industry=ind, qualitative=qual, risk=risk,
        )
        rec = Recommender().recommend(comp, current_price=current_price,
                                      metrics=metrics, risk_details=risk.details)

        beta = info.get("beta") or 1.0
        issues = ScenarioEngine().analyze(
            current_price=current_price or 0.0,
            sector=sector, beta=beta,
            macro_details=macro.details,
        )

        def _safe(v):
            if v is None: return None
            try: return float(v)
            except: return str(v)

        return {
            "ticker": req.ticker,
            "yf_ticker": yf_ticker,
            "name": info.get("shortName") or info.get("longName") or req.ticker,
            "sector": sector,
            "industry": info.get("industry", ""),
            "current_price": _safe(current_price),
            "currency": "KRW" if req.market in ("KRX","KOSPI","KOSDAQ") else "USD",
            "grade": comp.grade,
            "composite_score": round(comp.composite_score, 1),
            "scores": {
                "fundamental": round(comp.fundamental_score, 1),
                "technical":   round(comp.technical_score, 1),
                "macro":       round(comp.macro_score, 1),
                "industry":    round(comp.industry_score, 1),
                "qualitative": round(comp.qualitative_score, 1),
                "risk":        round(risk.risk_score, 1),
            },
            "metrics": {
                "per":              _safe(metrics.get("pe_ratio")),
                "pbr":              _safe(metrics.get("pb_ratio")),
                "roe":              _safe(metrics.get("roe")),
                "roa":              _safe(metrics.get("roa")),
                "operating_margin": _safe(metrics.get("operating_margin")),
                "debt_to_equity":   _safe(metrics.get("debt_to_equity")),
                "current_ratio":    _safe(metrics.get("current_ratio")),
                "dividend_yield":   _safe(metrics.get("dividend_yield")),
                "market_cap":       _safe(metrics.get("market_cap")),
                "revenue_growth":   _safe(metrics.get("revenue_growth")),
                "beta":             _safe(info.get("beta")),
                "mdd":              _safe(risk.mdd),
                "var_95":           _safe(risk.var_95),
            },
            "target_price":    _safe(rec.target_price),
            "stop_loss":       _safe(rec.stop_loss),
            "suggested_weight": rec.suggested_weight,
            "key_points":      rec.key_points or [],
            "risks":           rec.risks or [],
            "macro": {
                "cycle": macro.cycle,
                "score": round(macro.score, 1),
                "vix":   macro.details.get("vix"),
                "us_10y_yield": macro.details.get("us_10y_yield"),
                "geopolitical_score": macro.details.get("geopolitical_score"),
                "geopolitical_signals": macro.details.get("geopolitical_signals", []),
            },
            "technical": {
                "short_signal": t_res.short_term_signal,
                "mid_signal":   t_res.mid_term_signal,
                "rsi":    t_res.details.get("rsi"),
                "ma20":   t_res.details.get("ma20"),
                "ma200":  t_res.details.get("ma200"),
                "macd":   t_res.details.get("macd"),
                "support_levels":    t_res.support_levels,
                "resistance_levels": t_res.resistance_levels,
            },
            "scenarios": [
                {
                    "name": iss.name,
                    "emoji": iss.emoji,
                    "severity": iss.severity,
                    "signal": iss.signal,
                    "scenarios": [
                        {
                            "name": sc.name,
                            "probability": sc.probability,
                            "time_horizon": sc.time_horizon,
                            "price_impact_pct": sc.price_impact_pct,
                            "projected_price": sc.projected_price,
                            "description": sc.description,
                            "sentiment": sc.sentiment,
                        }
                        for sc in iss.scenarios
                    ],
                }
                for iss in issues
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}\n{traceback.format_exc()}")


@app.get("/ticker-info")
def ticker_info(ticker: str, market: str = "KRX"):
    """종목명 + 현재가 조회 (매수 폼 자동완성용)"""
    try:
        from data.collectors.yfinance_client import YFinanceClient
        import requests as req_lib
        yf_t = _krx_to_yf(ticker, market)
        client = YFinanceClient()
        info = client.get_ticker_info(yf_t) or {}
        name = None
        is_kr = market.upper() in ("KRX","KOSPI","KOSDAQ")
        if is_kr:
            try:
                r = req_lib.get(f"https://m.stock.naver.com/api/stock/{ticker}/basic",
                                headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
                d = r.json()
                name = d.get("stockName") or d.get("itemName")
            except Exception:
                pass
        if not name:
            name = info.get("shortName") or info.get("longName") or ticker
        price = info.get("regularMarketPrice") or info.get("currentPrice")
        currency = "USD" if market.upper() in ("NASDAQ","NYSE","SP500","AMEX") else "KRW"
        return {"name": name, "price_native": price, "currency": currency}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/usd-krw")
def usd_krw():
    """USD/KRW 실시간 환율"""
    try:
        from data.collectors.yfinance_client import YFinanceClient, _close_scalar
        import yfinance as yf
        client = YFinanceClient()
        df = yf.download("KRW=X", period="2d", interval="1d",
                         progress=False, auto_adjust=True, session=client.session)
        if not df.empty:
            v = _close_scalar(df)
            if v: return {"rate": v}
    except Exception:
        pass
    return {"rate": 1350.0}


@app.get("/portfolio/{session_id}")
def get_portfolio(session_id: str):
    pm = _get_portfolio(session_id)
    return {
        "cash": pm.cash,
        "initial_cash": pm.initial_cash,
        "holdings": list(pm.get_holdings().values()),
        "transactions": pm.transactions[-50:],
        "realized_pl": pm.realized_pl(),
    }


@app.post("/portfolio/buy")
def portfolio_buy(req: BuyRequest):
    pm = _get_portfolio(req.session_id)
    try:
        tx = pm.buy(req.ticker, req.market, req.name,
                    req.price_native, req.quantity,
                    currency=req.currency, fx_rate=req.fx_rate, note=req.note)
        _portfolios[req.session_id] = pm.data
        return {"ok": True, "transaction": tx, "cash": pm.cash}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/portfolio/sell")
def portfolio_sell(req: SellRequest):
    pm = _get_portfolio(req.session_id)
    try:
        tx = pm.sell(req.ticker, req.market, req.name,
                     req.price_native, req.quantity,
                     currency=req.currency, fx_rate=req.fx_rate, note=req.note)
        _portfolios[req.session_id] = pm.data
        return {"ok": True, "transaction": tx, "cash": pm.cash}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/portfolio/{session_id}/reset")
def portfolio_reset(session_id: str, initial_cash: float = 10_000_000):
    pm = _get_portfolio(session_id)
    pm.reset(initial_cash)
    _portfolios[session_id] = pm.data
    return {"ok": True}
