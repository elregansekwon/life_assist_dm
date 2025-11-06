import os
from pathlib import Path
from typing import Dict, Any

# YAML은 선택적 의존성: 없으면 파이썬 기본값 사용
try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

# 파이썬 기본값 (fallback)
from .intent_config import PERSONAL_INFO_KEYWORDS as DEFAULT_PERSONAL_INFO_KEYWORDS
from .intent_config import EXCEL_SHEETS as DEFAULT_EXCEL_SHEETS
from .intent_config import USER_INFO_COLUMNS as DEFAULT_USER_INFO_COLUMNS

CONFIG_FILE_NAME = "intent_config.yaml"


def _get_config_path() -> Path:
    current = Path(__file__).resolve()
    return current.parent / CONFIG_FILE_NAME


def load_yaml_config() -> Dict[str, Any]:
    """YAML 설정을 로드. 실패 시 빈 dict 반환."""
    if yaml is None:
        return {}
    cfg_path = _get_config_path()
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def get_personal_info_config() -> Dict[str, Any]:
    """YAML을 기본값과 병합하여 반환. YAML이 우선하되, 기본 키는 유지."""
    yaml_cfg = load_yaml_config().get("PERSONAL_INFO_KEYWORDS")
    if not isinstance(yaml_cfg, dict):
        return DEFAULT_PERSONAL_INFO_KEYWORDS
    merged: Dict[str, Any] = dict(DEFAULT_PERSONAL_INFO_KEYWORDS)
    # 얕은 병합: 상위 키 기준으로 YAML이 덮어씀
    for k, v in yaml_cfg.items():
        merged[k] = v
    return merged


def get_excel_sheets() -> Dict[str, str]:
    sheets = load_yaml_config().get("EXCEL_SHEETS")
    return sheets if isinstance(sheets, dict) else DEFAULT_EXCEL_SHEETS


def get_user_info_columns() -> Any:
    cols = load_yaml_config().get("USER_INFO_COLUMNS")
    return cols if isinstance(cols, list) else DEFAULT_USER_INFO_COLUMNS
