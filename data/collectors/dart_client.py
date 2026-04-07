"""DART 전자공시 데이터 수집 클라이언트"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

from ..cache_manager import CacheManager

logger = logging.getLogger(__name__)

DART_API_BASE = "https://opendart.fss.or.kr/api"


class DARTClient:
    """DART 전자공시 OpenAPI 클라이언트"""

    def __init__(self, api_key: str, cache: CacheManager | None = None):
        self.api_key = api_key
        self.cache = cache or CacheManager()

    # ------------------------------------------------------------------ #
    # Public
    # ------------------------------------------------------------------ #

    def get_company_info(self, corp_code: str) -> dict[str, Any]:
        """기업 기본 정보"""
        cache_key = f"dart:company:{corp_code}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        params = {"crtfc_key": self.api_key, "corp_code": corp_code}
        data = self._get("/company.json", params)
        if data:
            self.cache.set(cache_key, data, ttl=86400 * 7)
        return data or {}

    def get_financial_statements(
        self, corp_code: str, bsns_year: str, reprt_code: str = "11011"
    ) -> pd.DataFrame:
        """재무제표 수집
        reprt_code: 11011=사업보고서, 11012=반기, 11013=1분기, 11014=3분기
        """
        cache_key = f"dart:fs:{corp_code}:{bsns_year}:{reprt_code}"
        cached = self.cache.get(cache_key)
        if cached:
            return pd.DataFrame(cached)

        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bsns_year": bsns_year,
            "reprt_code": reprt_code,
            "fs_div": "CFS",  # 연결재무제표
        }
        data = self._get("/fnlttSinglAcntAll.json", params)
        if not data or "list" not in data:
            return pd.DataFrame()

        df = pd.DataFrame(data["list"])
        self.cache.set(cache_key, df.to_dict(orient="records"), ttl=86400)
        return df

    def get_disclosures(self, corp_code: str, days: int = 30) -> pd.DataFrame:
        """최근 공시 목록"""
        import datetime
        end = datetime.date.today()
        start = end - datetime.timedelta(days=days)

        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bgn_de": start.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "sort": "date",
            "sort_mth": "desc",
            "page_count": 40,
        }
        data = self._get("/list.json", params)
        if not data or "list" not in data:
            return pd.DataFrame()
        return pd.DataFrame(data["list"])

    def search_corp_code(self, stock_code: str) -> str | None:
        """주식 코드로 DART corp_code 조회"""
        # DART 고유번호 검색 (전체 목록 XML 파싱 대신 간이 API 사용)
        params = {"crtfc_key": self.api_key, "stock_code": stock_code}
        data = self._get("/company.json", params)
        if data and data.get("status") == "000":
            return data.get("corp_code")
        return None

    # ------------------------------------------------------------------ #
    # Internal
    # ------------------------------------------------------------------ #

    def _get(self, endpoint: str, params: dict) -> dict | None:
        if not self.api_key:
            logger.warning("DART API 키가 설정되지 않았습니다.")
            return None
        try:
            resp = requests.get(DART_API_BASE + endpoint, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data.get("status") not in ("000", "013"):
                logger.warning(f"DART API 오류: {data.get('message')}")
                return None
            return data
        except Exception as e:
            logger.error(f"DART API 호출 실패: {e}")
            return None
