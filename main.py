"""FastAPI 백엔드 — 기존 stock_analyzer 패키지를 REST API로 노출

실행:
    uvicorn main:app --reload --port 8000

Flutter 앱이 이 서버에 HTTP 요청을 보냅니다.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
from pathlib import Path
import traceback

# ── 디스크 기반 포트폴리오 영속성 (/tmp는 sleep/wake 사이에서 유지됨) ────────────
_PF_DIR = Path("/tmp/stock_pf")
_PF_DIR.mkdir(exist_ok=True)

def _pf_path(session_id: str) -> Path:
    # session_id는 UUID 형식이라 파일명으로 안전
    safe = "".join(c for c in session_id if c.isalnum() or c == "-")
    return _PF_DIR / f"{safe}.json"

def _save_to_disk(session_id: str, data: dict) -> None:
    try:
        with open(_pf_path(session_id), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, default=str)
    except Exception:
        pass

def _load_from_disk(session_id: str) -> dict | None:
    try:
        p = _pf_path(session_id)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None

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
        # 디스크에서 복원 시도 (서버 재기동 / sleep→wake 후)
        disk = _load_from_disk(session_id)
        _portfolios[session_id] = disk if disk is not None else {}
    pm = PortfolioManager(data=_portfolios[session_id])
    if "cash" not in _portfolios[session_id]:
        pm.reset(10_000_000)
        _save_to_disk(session_id, _portfolios[session_id])
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
        from data.processors.feature_engine import FeatureEngine
        from analysis.fundamental import FundamentalAnalyzer
        from analysis.technical import TechnicalAnalyzer
        from analysis.macro import MacroAnalyzer
        from analysis.industry import IndustryAnalyzer
        from analysis.qualitative import QualitativeAnalyzer
        from analysis.risk import RiskAnalyzer
        from analysis.scenario import ScenarioEngine
        from scoring.engine import ScoringEngine
        from scoring.recommender import Recommender
        import yaml, pathlib, pandas as pd

        # ── weights 로드 ──────────────────────────────────────────────────
        weights_path = pathlib.Path(__file__).parent / "config/weights.yaml"
        with open(weights_path) as f:
            weights = yaml.safe_load(f)

        # ── 데이터 수집 ───────────────────────────────────────────────────
        yf_ticker = _krx_to_yf(req.ticker, req.market)
        client    = YFinanceClient()

        # get_ticker_info(ticker, market)
        info = client.get_ticker_info(yf_ticker, req.market) or {}

        # get_price_history(ticker, market, period, interval)
        price_df = client.get_price_history(yf_ticker, req.market, period=req.period)

        # get_financials(ticker, market) → dict{"income_stmt", "balance_sheet", "cash_flow", ...}
        fin = {}
        try:
            fin = client.get_financials(yf_ticker, req.market) or {}
        except Exception:
            pass

        income_stmt   = fin.get("income_stmt",   pd.DataFrame())
        balance_sheet = fin.get("balance_sheet", pd.DataFrame())
        cash_flow     = fin.get("cash_flow",     pd.DataFrame())

        # ── 전처리 ────────────────────────────────────────────────────────
        processor     = DataProcessor()
        price_df_clean = processor.clean_price_df(price_df) if price_df is not None and not price_df.empty else pd.DataFrame()
        price_df_feat  = FeatureEngine().add_all_features(price_df_clean) if not price_df_clean.empty else price_df_clean

        # extract_financial_metrics(info, income_stmt, balance_sheet, cash_flow)
        metrics = processor.extract_financial_metrics(info, income_stmt, balance_sheet, cash_flow)

        # ── 현재가 ────────────────────────────────────────────────────────
        current_price = None
        try:
            close = price_df_clean["Close"]
            v = close.dropna().iloc[-1]
            current_price = float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
        except Exception:
            current_price = info.get("regularMarketPrice") or info.get("currentPrice")

        # ── 분석 ──────────────────────────────────────────────────────────
        sector = info.get("sector", "")

        # DataFrame None 안전 처리
        _pdf_clean = price_df_clean if (price_df_clean is not None and not price_df_clean.empty) else None
        _pdf_feat  = price_df_feat  if (price_df_feat  is not None and not price_df_feat.empty)  else pd.DataFrame()

        # FundamentalAnalyzer.analyze(ticker, metrics, price_df, sector_peers)
        f_res = FundamentalAnalyzer().analyze(yf_ticker, metrics=metrics, price_df=_pdf_clean)

        # TechnicalAnalyzer.analyze(ticker, df)
        t_res = TechnicalAnalyzer().analyze(yf_ticker, df=_pdf_feat)

        # MacroAnalyzer.analyze(sector)
        macro = MacroAnalyzer().analyze(sector)

        # IndustryAnalyzer.analyze(ticker, sector, info, price_df)
        ind = IndustryAnalyzer().analyze(yf_ticker, sector=sector, info=info, price_df=_pdf_clean)

        # QualitativeAnalyzer.analyze(ticker, info, news)
        qual = QualitativeAnalyzer().analyze(yf_ticker, info=info)

        # RiskAnalyzer.analyze(ticker, price_df, metrics)
        risk = RiskAnalyzer().analyze(yf_ticker, price_df_feat, metrics)

        # ScoringEngine.score(ticker, fundamental, technical, macro, industry, qualitative, risk)
        comp = ScoringEngine(weights=weights, mode=req.mode).score(
            yf_ticker,
            fundamental=f_res, technical=t_res,
            macro=macro, industry=ind, qualitative=qual, risk=risk,
        )

        # Recommender.recommend(composite, current_price, metrics, risk_details)
        rec = Recommender().recommend(
            comp,
            current_price=current_price,
            metrics=metrics,
            risk_details=getattr(risk, "details", {}),
        )

        # ScenarioEngine.analyze(current_price, sector, beta, macro_details)
        beta   = info.get("beta") or 1.0
        issues = ScenarioEngine().analyze(
            current_price=float(current_price or 0.0),
            sector=sector,
            beta=float(beta),
            macro_details=macro.details,
        )

        # ── 직렬화 ────────────────────────────────────────────────────────
        def _safe(v):
            if v is None: return None
            try:    return float(v)
            except: return str(v)

        return {
            "ticker":    req.ticker,
            "yf_ticker": yf_ticker,
            "name":      info.get("shortName") or info.get("longName") or req.ticker,
            "sector":    sector,
            "industry":  info.get("industry", ""),
            "current_price": _safe(current_price),
            "market":   req.market,
            "currency": "KRW" if req.market.upper() in ("KRX","KOSPI","KOSDAQ") else "USD",
            "grade":           comp.grade,
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
                "mdd":              _safe(getattr(risk, "mdd", None)),
                "var_95":           _safe(getattr(risk, "var_95", None)),
            },
            "target_price":     _safe(rec.target_price),
            "stop_loss":        _safe(rec.stop_loss),
            "suggested_weight": rec.suggested_weight,
            "key_points":       rec.key_points or [],
            "risks":            rec.risks or [],
            "macro": {
                "cycle": macro.cycle,
                "score": round(macro.score, 1),
                "vix":               macro.details.get("vix"),
                "us_10y_yield":      macro.details.get("us_10y_yield"),
                "geopolitical_score": macro.details.get("geopolitical_score"),
                "geopolitical_signals": macro.details.get("geopolitical_signals", []),
            },
            "technical": {
                "short_signal":      t_res.short_term_signal,
                "mid_signal":        t_res.mid_term_signal,
                "rsi":               t_res.details.get("rsi"),
                "ma20":              t_res.details.get("ma20"),
                "ma200":             t_res.details.get("ma200"),
                "macd":              t_res.details.get("macd"),
                "support_levels":    t_res.support_levels,
                "resistance_levels": t_res.resistance_levels,
            },
            "scenarios": [
                {
                    "name":     iss.name,
                    "emoji":    iss.emoji,
                    "severity": iss.severity,
                    "signal":   iss.signal,
                    "scenarios": [
                        {
                            "name":             sc.name,
                            "probability":      sc.probability,
                            "time_horizon":     sc.time_horizon,
                            "price_impact_pct": sc.price_impact_pct,
                            "projected_price":  sc.projected_price,
                            "description":      sc.description,
                            "sentiment":        sc.sentiment,
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


@app.get("/price-history")
def price_history(ticker: str, market: str = "KRX", period: str = "6mo"):
    """주가 히스토리 (선 그래프용)"""
    try:
        from data.collectors.yfinance_client import YFinanceClient
        from data.processors.data_processor import DataProcessor
        import pandas as pd

        yf_t = _krx_to_yf(ticker, market)
        client = YFinanceClient()
        df = client.get_price_history(yf_t, market, period=period)
        if df is None or (hasattr(df, 'empty') and df.empty):
            return {"prices": []}

        # 멀티레벨 컬럼 평탄화
        if hasattr(df.columns, 'levels'):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        processor = DataProcessor()
        try:
            df = processor.clean_price_df(df)
        except Exception:
            pass

        if df is None or (hasattr(df, 'empty') and df.empty):
            return {"prices": []}

        # Close 컬럼 찾기
        close_col = None
        for c in ['Close', 'close', 'Adj Close']:
            if c in df.columns:
                close_col = c
                break
        if close_col is None:
            return {"prices": []}

        prices = []
        for i, (idx, row) in enumerate(df.iterrows()):
            try:
                v = row[close_col]
                if hasattr(v, 'iloc'):
                    v = float(v.iloc[0])
                else:
                    v = float(v)
                if v != v:  # NaN
                    continue
                prices.append({"t": i, "c": round(v, 2), "d": str(idx)[:10]})
            except Exception:
                continue

        return {"prices": prices}
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
    # Flutter 앱 모델 필드명에 맞게 변환 (avg_price_krw → avg_cost_krw)
    def _normalize_holding(h: dict) -> dict:
        avg_krw = h.get("avg_price_krw") or h.get("avg_cost_krw") or h.get("avg_price", 0)
        avg_native = h.get("avg_price_native") or avg_krw
        return {
            "ticker":          h.get("ticker", ""),
            "market":          h.get("market", ""),
            "name":            h.get("name", ""),
            "quantity":        h.get("quantity", 0),
            "avg_cost_krw":    avg_krw,
            "avg_cost_native": avg_native,
            "currency":        h.get("currency", "KRW"),
        }
    def _normalize_tx(tx: dict) -> dict:
        return {**tx, "total_krw": tx.get("total_krw") or tx.get("total", 0)}
    return {
        "cash":         pm.cash,
        "initial_cash": pm.initial_cash,
        "holdings":     [_normalize_holding(h) for h in pm.get_holdings().values()],
        "transactions": [_normalize_tx(t) for t in pm.transactions[-50:]],
        "realized_pl":  pm.realized_pl(),
    }


@app.post("/portfolio/buy")
def portfolio_buy(req: BuyRequest):
    pm = _get_portfolio(req.session_id)
    try:
        tx = pm.buy(req.ticker, req.market, req.name,
                    req.price_native, req.quantity,
                    currency=req.currency, fx_rate=req.fx_rate, note=req.note)
        _portfolios[req.session_id] = pm.data
        _save_to_disk(req.session_id, pm.data)
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
        _save_to_disk(req.session_id, pm.data)
        return {"ok": True, "transaction": tx, "cash": pm.cash}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/portfolio/{session_id}/reset")
def portfolio_reset(session_id: str, initial_cash: float = 10_000_000):
    pm = _get_portfolio(session_id)
    pm.reset(initial_cash)
    _portfolios[session_id] = pm.data
    _save_to_disk(session_id, pm.data)
    return {"ok": True}


@app.post("/portfolio/{session_id}/restore")
def portfolio_restore(session_id: str, body: dict = Body(...)):
    """앱 로컬 캐시에서 포트폴리오 복원 (서버가 재기동돼 데이터를 잃었을 때)"""
    from portfolio.manager import PortfolioManager
    # body는 GET /portfolio 응답 형식
    transactions = body.get("transactions", [])
    for tx in transactions:
        # total_krw → total 필드명 매핑
        if "total" not in tx and "total_krw" in tx:
            tx["total"] = tx["total_krw"]
    manager_data = {
        "cash":         body.get("cash", 10_000_000),
        "initial_cash": body.get("initial_cash", 10_000_000),
        "transactions": transactions,
        "snapshots":    body.get("snapshots", []),
    }
    _portfolios[session_id] = manager_data
    _save_to_disk(session_id, manager_data)
    return {"ok": True}
