import os
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import logging
import threading
import time

logger = logging.getLogger(__name__)

import json
import re


def _normalize_schedule_title(title: str) -> str:
    """제목 비교용: strip 후 연속 공백을 하나로. '미용실  예약' == '미용실 예약' """
    if not title or not isinstance(title, str):
        return ""
    return re.sub(r"\s+", " ", title.strip())


def _normalize_date_for_compare(date_val: str) -> str:
    """날짜 비교용: '2026-02-26 00:00:00' 또는 '2026-02-26' -> '2026-02-26' """
    if not date_val:
        return ""
    s = str(date_val).strip()
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    return s


class ScheduleConflictException(Exception):
    """일정 충돌 예외: 날짜+시간이 같은데 제목이 다른 일정이 있을 때 발생"""
    def __init__(self, existing_schedule: Dict[str, Any], new_schedule: Dict[str, Any], user_name: str):
        self.existing_schedule = existing_schedule
        self.new_schedule = new_schedule
        self.user_name = user_name
        super().__init__(f"일정 충돌: {existing_schedule.get('제목', '')} vs {new_schedule.get('제목', '')}")

SHEET_SCHEMAS = {
    "물건위치": ["날짜", "물건이름", "장소", "세부위치", "출처", "엔티티타입"],
    "복약정보": ["날짜", "약이름", "용량", "단위", "시간", "복용방법", "복용기간", "엔티티타입"],
    "일정": ["날짜", "제목", "시간", "장소", "정보", "엔티티타입"],
    "가족관계": ["날짜", "관계", "이름", "정보", "엔티티타입"],
    "감정기록": ["날짜", "감정", "정보", "엔티티타입"],
    "음식기록": ["날짜", "끼니", "시간", "메뉴", "엔티티타입"],
    "사용자정보KV": ["날짜", "키", "값", "출처", "확신도", "엔티티타입"],
    "대화기록": ["날짜", "시간", "대화요약"],
}

def _get_package_dir() -> Path:
    """user_excel_manager.py와 같은 패키지 디렉터리 (user_information 상위)"""
    current_file = Path(__file__).resolve()
    return current_file.parent


def _get_user_information_dir() -> Path:
    """
    불러오기·저장 모두 소스 트리의 user_information 한 경로로 통일.
    - install 실행 시: .../src/life_assist_dm/life_assist_dm/user_information
    - 소스 실행 시: .../life_assist_dm/life_assist_dm/user_information
    """
    current_file = Path(__file__).resolve()
    parts = current_file.parts

    # install 공간에서 실행 중이면 workspace/src/.../user_information 로 맞춤
    if "install" in parts:
        idx = parts.index("install")
        workspace = Path(*parts[:idx])
        source_user_info = workspace / "src" / "life_assist_dm" / "life_assist_dm" / "user_information"
        return source_user_info.resolve()

    # 소스에서 실행 시: 이 파일 옆의 user_information
    return (_get_package_dir() / "user_information").resolve()


class UserExcelManager:
    """엑셀 로드/저장은 반드시 user_information 디렉터리 한 곳에서만 수행."""

    def __init__(self, base_dir: str = None):
        if base_dir is not None and str(base_dir).strip():
            self.base_dir = Path(os.path.expanduser(base_dir)).resolve()
        else:
            self.base_dir = _get_user_information_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"엑셀 경로 [user_information]: {self.base_dir}")

        self._buffered_changes = defaultdict(list)

        self._flush_lock = threading.Lock()
        self._pending_flush = {}
        self._flush_delay = 1.0
    
    def _get_sheet_name(self, entity_type: str) -> str:
        mapping = {
            "물건": "물건위치",
            "user.물건": "물건위치",
            "약": "복약정보",
            "user.약": "복약정보",
            "일정": "일정",
            "user.일정": "일정",
            "식사": "음식기록",
            "user.식사": "음식기록",
            "음식": "음식기록",
            "user.음식": "음식기록",
            "정서": "감정기록",
            "감정": "감정기록",
            "user.건강상태": "감정기록",
            "가족": "가족관계",
            "user.가족": "가족관계",
            "사용자": "사용자정보KV",
            "user.사용자": "사용자정보KV",
            "취향": "사용자정보KV",
            "선호": "사용자정보KV",
            "기념일": "사용자정보KV",
            "취미": "사용자정보KV",
        }

        sheet_name = mapping.get(entity_type, "사용자정보KV")
        if entity_type not in mapping:
            logger.info(f"[INFO] '{entity_type}' 엔티티 타입이 매핑되지 않아 사용자정보KV로 저장")
        return sheet_name
        
    def get_user_excel_path(self, user_name: str) -> Path:

        file_name = f"{user_name}.xlsx"
        return self.base_dir / file_name
    
    def load_user_excel(self, user_name: str) -> Optional[pd.ExcelFile]:
        excel_path = self.get_user_excel_path(user_name)
        if not excel_path.exists():
            return None
        try:
            return pd.ExcelFile(excel_path)
        except Exception as e:
            logger.error(f"엑셀 파일 로드 실패: {e}")
            return None
    
    def load_sheet_data(self, user_name: str, sheet_name: str) -> pd.DataFrame:
        excel_file = self.load_user_excel(user_name)
        if excel_file is None:
            return pd.DataFrame()
        try:
            if sheet_name in excel_file.sheet_names:
                return pd.read_excel(excel_file, sheet_name=sheet_name)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"시트 데이터 로드 실패 ({sheet_name}): {e}")
            return pd.DataFrame()
    
    def safe_load_sheet(self, user_name: str, sheet_name: str) -> pd.DataFrame:
        try:
            df = self.load_sheet_data(user_name, sheet_name)
            schema = SHEET_SCHEMAS.get(sheet_name, [])
            if df is None or df.empty:
                return pd.DataFrame(columns=schema)

            for col in schema:
                if col not in df.columns:
                    df[col] = ""

            return df[schema]
        except Exception as e:
            logger.error(f"[ERROR] safe_load_sheet 실패: {e}")
            schema = SHEET_SCHEMAS.get(sheet_name, [])
            return pd.DataFrame(columns=schema)
    
    def save_data_to_sheet(self, user_name: str, sheet_name: str, data: List[Dict[str, Any]], 
                           append: bool = True):
        excel_path = self.get_user_excel_path(user_name)
        
        def _cleanup_lockfile(path: Path):
            try:
                lock_path = Path(str(path) + ".lock")
                if lock_path.exists():
                    lock_path.unlink(missing_ok=True)
                    logger.debug(f"[LOCK CLEANUP] Lock 파일 제거됨: {lock_path}")
            except Exception as e:
                logger.warning(f"[LOCK CLEANUP 실패] {e}")
        
        existing_data = []
        if excel_path.exists() and append:
            try:
                df_existing = self.load_sheet_data(user_name, sheet_name)
                existing_data = df_existing.to_dict('records')
            except Exception as e:
                logger.warning(f"기존 데이터 로드 실패: {e}")
        
        if append:
            existing_data.extend(data)
        else:
            existing_data = data
        
        df = pd.DataFrame(existing_data)

        schema = SHEET_SCHEMAS.get(sheet_name, [])
        if schema:

            for col in schema:
                if col not in df.columns:
                    df[col] = ""

            df = df[schema]
        
        try:
            mode = 'a' if excel_path.exists() else 'w'
            with pd.ExcelWriter(
                excel_path, engine='openpyxl', mode=mode, if_sheet_exists='replace'
            ) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)

            _cleanup_lockfile(excel_path)
        except TypeError:

            if excel_path.exists():
                excel_file = self.load_user_excel(user_name)
                if excel_file is None:
                    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                        df.to_excel(writer, sheet_name=sheet_name, index=False)
                else:
                    excel_data = {}
                    for sheet in excel_file.sheet_names:
                        if sheet == sheet_name:
                            excel_data[sheet] = df
                        else:
                            excel_data[sheet] = pd.read_excel(excel_file, sheet_name=sheet)
                    if sheet_name not in excel_data:
                        excel_data[sheet_name] = df
                    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                        for sheet_name_key, df_data in excel_data.items():
                            df_data.to_excel(writer, sheet_name=sheet_name_key, index=False)
                    _cleanup_lockfile(excel_path)
            else:
                with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                _cleanup_lockfile(excel_path)
    
    def _convert_duration_to_date_range(self, duration_str: str) -> str:
        if not duration_str:
            return ""
        
        import re
        from datetime import datetime, timedelta
        
        if "~" in duration_str or "-" in duration_str:

            if re.match(r"\d{4}-\d{2}-\d{2}", duration_str.split("~")[0].strip()):
                return duration_str
        
        duration_match = re.search(r"(\d+)\s*(일|주|개월|년)", duration_str)
        if not duration_match:

            return duration_str
        
        days_to_add = 0
        number = int(duration_match.group(1))
        unit = duration_match.group(2)
        
        if unit == "일":
            days_to_add = number
        elif unit == "주":
            days_to_add = number * 7
        elif unit == "개월":
            days_to_add = number * 30
        elif unit == "년":
            days_to_add = number * 365
        
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_to_add - 1)
        
        start_str = start_date.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_date.strftime("%Y-%m-%d %H:%M:%S")
        return f"{start_str}~{end_str}"
    
    def _normalize_entity(self, entity_type: str, data: dict) -> dict:
        norm = {}
        try:
            if entity_type in ["물건", "user.물건"]:

                # 원본 이름
                raw_name = data.get("물건이름") or data.get("이름", "")
                if isinstance(raw_name, str):
                    name = raw_name.strip()
                    # LLM 포맷에서 온 "치약, location: null" 같은 노이즈 제거
                    # - ", location: null" 전체 제거
                    # - "location: ..." 패턴이 섞여 있으면 그 앞까지를 물건 이름으로 사용
                    lower = name.lower()
                    if "location:" in lower:
                        # 쉼표 기준 앞부분만 사용 (예: "치약, location: null" → "치약")
                        name = name.split("location:")[0].rstrip(", ").strip()
                    # 혹시 남아 있을 수 있는 "null" 토큰 제거
                    if name.lower().endswith("null"):
                        name = name[:-4].rstrip(", ").strip()
                    norm["물건이름"] = name
                else:
                    norm["물건이름"] = str(raw_name) if raw_name is not None else ""

                norm["장소"] = str(data.get("장소", "")).strip()
                norm["세부위치"] = str(data.get("세부위치", "")).strip()
                
                # 세부위치에서 조사 제거 (예: "위에" → "위", "앞에서" → "앞")
                if norm["세부위치"]:
                    import re
                    # 끝에 오는 조사 제거: 에, 에서, 로, 으로, 의, 와, 과 등
                    norm["세부위치"] = re.sub(r'(에|에서|로|으로|의|와|과|까지|부터|만|도|조차|마저|부터|까지)$', '', norm["세부위치"]).strip()

                if not norm["장소"] and not norm["세부위치"]:
                    location = str(data.get("위치", "")).strip()
                    if location:

                        import re

                        if "내 방" in location or "내방" in location:

                            if location.startswith("내 방") or location.startswith("내방"):

                                remaining = location.replace("내 방", "").replace("내방", "").strip()
                                norm["장소"] = "내 방" if "내 방" in location else "내방"
                                norm["세부위치"] = remaining
                            else:

                                norm["장소"] = "내 방" if "내 방" in location else "내방"
                                norm["세부위치"] = location.replace("내 방", "").replace("내방", "").strip()
                        else:

                            room_keywords = ["안방", "다용도실", "화장실", "주방", "거실", "침실", "현관", "베란다", "방"]
                            room_keywords_sorted = sorted(room_keywords, key=len, reverse=True)
                            for room in room_keywords_sorted:
                                if room in location:
                                    norm["장소"] = room
                                    norm["세부위치"] = location.replace(room, "").strip()
                                    break
                        if not norm["장소"]:

                            norm["세부위치"] = location
                        
                        # location 파싱 후에도 세부위치에서 조사 제거
                        if norm["세부위치"]:
                            import re
                            norm["세부위치"] = re.sub(r'(에|에서|로|으로|의|와|과|까지|부터|만|도|조차|마저|부터|까지)$', '', norm["세부위치"]).strip()

                norm["출처"] = data.get("출처") or data.get("추출방법", "사용자 발화")

            elif entity_type in ["약", "user.약"]:

                norm["약이름"] = data.get("약이름") or data.get("약명") or data.get("이름", "")

                dose = str(data.get("용량", "")).strip()
                unit = str(data.get("단위", "")).strip()
                norm["용량"] = dose if dose else ""
                norm["단위"] = unit if unit else ""

                norm["시간"] = data.get("시간대") or data.get("시간") or data.get("복용시간", "")

                norm["복용방법"] = data.get("복용방법") or data.get("메모") or ""

                복용기간_원본 = data.get("복용기간") or ""
                if 복용기간_원본:

                    norm["복용기간"] = self._convert_duration_to_date_range(복용기간_원본)
                else:
                    norm["복용기간"] = ""
            elif entity_type == "일정":
                norm["제목"] = data.get("제목", "")

                date_value = data.get("날짜", "")
                if date_value:
                    try:
                        from life_assist_dm.support_chains import _normalize_date_to_iso
                        date_str = str(date_value).strip()
                        if date_str and date_str.lower() not in ("nan", "none", ""):
                            # 일정 시트는 날짜 컬럼을 날짜(YYYY-MM-DD)만 유지
                            norm["날짜"] = _normalize_date_to_iso(date_str)
                        else:
                            norm["날짜"] = ""
                    except Exception as e:
                        logger.warning(f"일정 날짜 정규화 실패: {e}, 원본 값 사용: {date_value}")
                        norm["날짜"] = str(date_value) if date_value else ""
                else:
                    norm["날짜"] = ""
                norm["시간"] = data.get("시간", "")
                norm["장소"] = data.get("장소", "")
                norm["정보"] = data.get("정보", "")
            elif entity_type in ["식사", "음식"]:
                norm["끼니"] = data.get("끼니", "")
                norm["시간"] = data.get("시간", "") or data.get("시간대", "")
                if isinstance(data.get("메뉴"), list):
                    norm["메뉴"] = ", ".join(str(m) for m in data["메뉴"])
                else:
                    norm["메뉴"] = str(data.get("메뉴", "")).strip()

                date_value = data.get("날짜", "")
                if date_value:
                    try:
                        from life_assist_dm.support_chains import _normalize_date_to_iso
                        date_str = str(date_value).strip()
                        if date_str and date_str.lower() not in ("nan", "none", ""):
                            base = _normalize_date_to_iso(date_str)
                            norm["날짜"] = f"{base} 00:00:00"
                        else:

                            norm["날짜"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    except Exception as e:
                        logger.warning(f"날짜 정규화 실패: {e}, 원본 값 사용: {date_value}")
                        norm["날짜"] = str(date_value) if date_value else ""
                else:

                    norm["날짜"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elif entity_type == "정서" or entity_type == "감정":
                norm["감정"] = data.get("감정") or data.get("상태") or data.get("증상", "")
                norm["정보"] = data.get("정보", "") or data.get("원문", "")
            elif entity_type == "가족":
                norm["관계"] = data.get("관계", "")
                norm["이름"] = data.get("이름", "")
                norm["정보"] = data.get("정보", "")
            elif entity_type in ["취향", "선호", "기념일", "취미"]:

                norm["내용"] = data.get("내용") or json.dumps(data, ensure_ascii=False)
                norm["정보"] = data.get("정보") or entity_type
            else:

                norm["내용"] = json.dumps(data, ensure_ascii=False)
                norm["정보"] = ""
        except Exception as e:
            logger.warning(f"엔티티 정규화 중 오류: {e}")
            norm["내용"] = json.dumps(data, ensure_ascii=False)
            norm["정보"] = ""
        return norm
    
    def save_entity_data(self, user_name: str, entity_type: str, data: Dict[str, Any]):

        if not user_name or not str(user_name).strip() or user_name == "사용자":
            logger.warning(f"[WARN] 잘못된 사용자명으로 저장 시도: {user_name}")
            return
        if entity_type is None or (isinstance(entity_type, str) and entity_type.strip() in ("", "None")):
            logger.warning(f"[WARN] entity_type이 비어 있어 저장 생략: {entity_type!r}")
            return

        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sheet_name = self._get_sheet_name(entity_type)
            
            non_user_entity_types = ["물건", "약", "일정", "식사", "음식", "정서", "감정", "가족", "user.물건", "user.약", "user.일정", "user.식사", "user.음식", "user.건강상태", "user.가족"]
            if entity_type not in non_user_entity_types:
                try:
                    from .dialog_manager.config.config_loader import get_excel_sheets
                    sheets = get_excel_sheets()
                    kv_sheet = sheets.get("user_info_kv", "사용자정보KV")
                    
                    normalized_user = {}
                    import re
                    
                    if entity_type == "사용자":

                        if "나이" in data and data["나이"]:
                            m = re.search(r"(\d+)", str(data["나이"]))
                            if m:
                                normalized_user["나이"] = f"{m.group(1)}살"

                        if "학교" in data and data["학교"]:
                            raw_school = str(data["학교"]).strip()
                            raw_school = re.sub(r"^(?:나는|난|저는)\s*", "", raw_school)
                            raw_school = re.sub(r"\s*(?:에\s*다녀.*|다녀.*)$", "", raw_school)
                            m = re.search(r"([가-힣A-Za-z\s]+?(?:중학교|고등학교|대학교|초등학교|학교))", raw_school)
                            if m:
                                normalized_user["학교"] = m.group(1).strip()

                        for k in ["이름", "별칭", "직업", "취미", "회사", "인턴"]:
                            if k in data and data[k]:
                                normalized_user[k] = data[k]
                    elif entity_type in ["취향", "선호"]:

                        content = data.get("내용", "") or data.get("값", "") or json.dumps(data, ensure_ascii=False)
                        if content:
                            normalized_user["취향"] = content
                    elif entity_type == "기념일":

                        if "제목" in data and data["제목"]:
                            normalized_user["기념일"] = f"{data.get('제목', '')} ({data.get('날짜', '')})"
                        elif "날짜" in data and data["날짜"]:
                            normalized_user["기념일"] = data.get("날짜", "")
                    elif entity_type == "취미":

                        hobby = data.get("이름", "") or data.get("취미", "") or ""
                        if hobby:
                            normalized_user["취미"] = hobby
                    else:

                        import json
                        entity_json = json.dumps(data, ensure_ascii=False)
                        normalized_user[entity_type] = entity_json
                    
                    kv_rows = []
                    if entity_type == "사용자":
                        for k in ["이름", "별칭", "나이", "학교", "직업", "취미", "회사", "인턴"]:
                            if k in normalized_user and str(normalized_user[k]).strip() != "":
                                kv_rows.append({
                                    "날짜": now,
                                    "키": k,
                                    "값": normalized_user[k],
                                    "출처": "사용자 발화",
                                    "확신도": "",
                                    "엔티티타입": entity_type,
                                })
                    else:

                        for k, v in normalized_user.items():
                            if v and str(v).strip() != "":
                                kv_rows.append({
                                    "날짜": now,
                                    "키": k,
                                    "값": v,
                                    "출처": "사용자 발화",
                                    "확신도": "",
                                    "엔티티타입": entity_type,
                                })
                    if kv_rows:

                        self._buffered_changes[(user_name, kv_sheet)].extend(kv_rows)
                        logger.info(f"[BUFFER] {user_name}:{kv_sheet} 엔티티 버퍼링됨 ({entity_type})")
                except Exception as e:

                    logger.error(f"[ERROR] 사용자정보KV 저장 실패: {e}")
                    try:
                        kv_sheet = "사용자정보KV"
                        now_local = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        kv_rows = []
                        for k in ["이름", "나이", "학교", "직업", "취미", "회사", "인턴"]:
                            if k in data and str(data[k]).strip() != "":
                                kv_rows.append({
                                    "날짜": now_local,
                                    "키": k,
                                    "값": data[k],
                                    "출처": "사용자 발화",
                                    "확신도": "",
                                    "엔티티타입": entity_type,
                                })
                        if kv_rows:
                            self._buffered_changes[(user_name, kv_sheet)].extend(kv_rows)
                            logger.info(f"[BUFFER] {user_name}:{kv_sheet} 엔티티 버퍼링됨 ({entity_type})")
                    except Exception:
                        pass
                return
            
            normalized = self._normalize_entity(entity_type, data)

            date_value = normalized.get("날짜", "")
            if sheet_name == "일정":
                # 일정 시트: '발화 시각'이 아닌 '일정 날짜'만 저장
                # - 발화에서 날짜가 추출된 경우에만 YYYY-MM-DD로 저장
                # - 날짜가 전혀 없으면 빈 값으로 두어, 사용자가 나중에 보거나 수정할 때도
                #   "언제 하는 일정인지 모르는 일정"으로 명확히 보이게 함
                if date_value and str(date_value).strip().lower() not in ("nan", "none"):
                    dv = str(date_value).strip()
                    # 혹시 datetime 형태가 들어오면 앞의 날짜 부분만 사용
                    if len(dv) >= 10 and dv[4] == "-" and dv[7] == "-":
                        normalized["날짜"] = dv[:10]
                    else:
                        normalized["날짜"] = dv
                else:
                    normalized["날짜"] = ""
            else:
                if not date_value or str(date_value).strip() == "" or str(date_value).lower() in ("nan", "none"):
                    # 기본은 항상 YYYY-MM-DD HH:MM:SS 포맷
                    normalized["날짜"] = now
                else:
                    dv = str(date_value).strip()
                    # YYYY-MM-DD 형태만 들어온 경우에는 시간 00:00:00을 붙여서 통일
                    if len(dv) == 10 and dv[4] == "-" and dv[7] == "-":
                        normalized["날짜"] = f"{dv} 00:00:00"
                    else:
                        normalized["날짜"] = dv
            normalized["엔티티타입"] = entity_type
            
            schema = SHEET_SCHEMAS.get(sheet_name, SHEET_SCHEMAS["사용자정보KV"])
            for col in schema:
                if col not in normalized:
                    normalized[col] = ""
            
            if entity_type in ["물건", "user.물건"]:
                logger.debug(f"[SAVE DEBUG] 물건 저장 - normalized: {normalized}")
                logger.debug(f"[SAVE DEBUG] 물건 저장 - schema: {schema}")
            
            record = {k: str(normalized[k]) if normalized[k] is not None else "" for k in schema}
            
            if entity_type in ["물건", "user.물건"]:
                logger.debug(f"[SAVE DEBUG] 물건 저장 - record: {record}")
            
            buffer_key = (user_name, sheet_name)
            
            # ✅ 일정 중복 저장 방지: 날짜 + 제목 + 시간 + 장소가 모두 같으면 중복
            if entity_type in ["일정", "user.일정"]:
                # 버퍼에서 중복 확인
                existing_buffer = self._buffered_changes.get(buffer_key, [])
                duplicate_found = False
                
                record_date = record.get("날짜", "").strip()
                record_title = record.get("제목", "").strip()
                record_time = record.get("시간", "").strip()
                record_location = record.get("장소", "").strip()
                record_date_norm = _normalize_date_for_compare(record_date)
                record_title_norm = _normalize_schedule_title(record_title)
                
                for i, existing_record in enumerate(existing_buffer):
                    existing_date = str(existing_record.get("날짜", "")).strip()
                    existing_title = str(existing_record.get("제목", "")).strip()
                    existing_time = str(existing_record.get("시간", "")).strip()
                    existing_location = str(existing_record.get("장소", "")).strip()
                    existing_date_norm = _normalize_date_for_compare(existing_date)
                    existing_title_norm = _normalize_schedule_title(existing_title)
                    
                    # 날짜(일만), 제목(공백 정규화), 시간, 장소가 모두 같으면 중복
                    if (record_date_norm == existing_date_norm and 
                        record_title_norm == existing_title_norm and 
                        record_time == existing_time and 
                        record_location == existing_location):
                        # 기존 레코드를 새 것으로 교체
                        existing_buffer[i] = record
                        duplicate_found = True
                        logger.info(f"[DUPLICATE] 일정 중복 발견 - 기존 레코드 교체: {record_title} ({record_date} {record_time})")
                        break
                
                if duplicate_found:
                    # 버퍼 업데이트 (이미 교체됨)
                    self._buffered_changes[buffer_key] = existing_buffer
                    logger.info(f"[BUFFER] {user_name}:{sheet_name} 일정 중복 방지 - 기존 레코드 교체됨 ({entity_type})")
                else:
                    # ✅ 날짜+시간 충돌 확인 (제목이 다른 경우)
                    conflict_found = False
                    conflict_existing = None
                    
                    # 버퍼에서 날짜+시간 충돌 확인 (날짜는 일만, 제목은 공백 정규화 후 비교)
                    for existing_record in existing_buffer:
                        existing_date = str(existing_record.get("날짜", "")).strip()
                        existing_time = str(existing_record.get("시간", "")).strip()
                        existing_title = str(existing_record.get("제목", "")).strip()
                        existing_date_norm = _normalize_date_for_compare(existing_date)
                        existing_title_norm = _normalize_schedule_title(existing_title)
                        
                        # 날짜+시간이 같고, 정규화한 제목이 다를 때만 충돌
                        if (record_date_norm == existing_date_norm and 
                            record_time == existing_time and 
                            record_title_norm != existing_title_norm):
                            conflict_found = True
                            conflict_existing = existing_record
                            logger.info(f"[SCHEDULE CONFLICT] 버퍼에서 충돌 발견: {existing_title} vs {record_title} ({record_date} {record_time})")
                            break
                    
                    # 엑셀 파일에서도 날짜+시간 충돌 확인
                    if not conflict_found:
                        try:
                            df_existing = self.load_sheet_data(user_name, sheet_name)
                            if df_existing is not None and not df_existing.empty:
                                for _, row in df_existing.iterrows():
                                    existing_date = str(row.get("날짜", "")).strip()
                                    existing_title = str(row.get("제목", "")).strip()
                                    existing_time = str(row.get("시간", "")).strip()
                                    existing_location = str(row.get("장소", "")).strip()
                                    existing_date_norm = _normalize_date_for_compare(existing_date)
                                    existing_title_norm = _normalize_schedule_title(existing_title)
                                    
                                    # 날짜(일만)+제목(정규화)+시간+장소가 모두 같으면 완전 중복 (저장 안 함)
                                    if (record_date_norm == existing_date_norm and 
                                        record_title_norm == existing_title_norm and 
                                        record_time == existing_time and 
                                        record_location == existing_location):
                                        logger.info(f"[DUPLICATE] 일정 중복 발견 (엑셀) - 저장 건너뜀: {record_title} ({record_date} {record_time})")
                                        return  # 저장하지 않고 종료
                                    
                                    # 날짜+시간이 같고 정규화한 제목이 다를 때만 충돌
                                    if (record_date_norm == existing_date_norm and 
                                        record_time == existing_time and 
                                        record_title_norm != existing_title_norm):
                                        conflict_found = True
                                        conflict_existing = {
                                            "날짜": existing_date,
                                            "제목": existing_title,
                                            "시간": existing_time,
                                            "장소": existing_location,
                                            "정보": str(row.get("정보", "")).strip()
                                        }
                                        logger.info(f"[SCHEDULE CONFLICT] 엑셀에서 충돌 발견: {existing_title} vs {record_title} ({record_date} {record_time})")
                                        break
                        except Exception as e:
                            logger.debug(f"[DUPLICATE CHECK] 엑셀 중복 확인 실패 (무시): {e}")
                    
                    # ✅ 충돌이 있으면 예외 발생 (호출하는 쪽에서 처리)
                    if conflict_found:
                        from life_assist_dm.user_excel_manager import ScheduleConflictException
                        raise ScheduleConflictException(
                            existing_schedule=conflict_existing,
                            new_schedule=record,
                            user_name=user_name
                        )
                    
                    # 중복이 없으면 버퍼에 추가
                    self._buffered_changes[buffer_key].append(record)
                    logger.info(f"[BUFFER] {user_name}:{sheet_name} 엔티티 버퍼링됨 ({entity_type})")
            else:
                # 일정이 아닌 경우 기존 로직대로 추가
                self._buffered_changes[buffer_key].append(record)
                logger.info(f"[BUFFER] {user_name}:{sheet_name} 엔티티 버퍼링됨 ({entity_type})")
            
            logger.debug(f"[BUFFER DEBUG] 버퍼링 직후 - 버퍼 키: {buffer_key}, 레코드 수: {len(self._buffered_changes[buffer_key])}")
            
        except ScheduleConflictException:
            # 일정 충돌은 호출한 쪽에서 처리해야 하므로 그대로 전달
            raise
        except Exception as e:
            logger.error(f"[ERROR] save_entity_data 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def save_conversation_summary(self, user_name: str, summary: str, 
                                 timestamp: Optional[str] = None):
        if timestamp is None:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        record = {
            "날짜": timestamp,
            "시간": timestamp.split()[1] if len(timestamp.split()) > 1 else "",
            "대화요약": summary,
            "엔티티타입": "대화기록"
        }
        
        key = (user_name, "대화기록")
        self._buffered_changes[key].append(record)
        logger.info(f"[BUFFER] 대화 요약 버퍼링됨: {user_name}")

        try:
            if len(self._buffered_changes.get(key, [])) >= 3:
                self.request_flush(user_name)
                logger.info(f"[FLUSH] 대화요약 누적 3회 → Excel 동기화 예약 ({user_name})")
            else:
                logger.debug(f"[BUFFER] 대화요약 누적 {len(self._buffered_changes.get(key, []))}회 (미flush)")
        except Exception:
            pass
    
    def initialize_user_excel(self, user_name: str):
        excel_path = self.get_user_excel_path(user_name)
        
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
            for sheet_name, columns in SHEET_SCHEMAS.items():
                df = pd.DataFrame(columns=columns)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"사용자 엑셀 파일 초기화 완료: {user_name}")
    
    def user_exists(self, user_name: str) -> bool:
        return self.get_user_excel_path(user_name).exists()

    def cleanup_all_locks(self):
        try:
            for lockfile in self.base_dir.glob("*.xlsx.lock"):
                try:
                    lockfile.unlink(missing_ok=True)
                    logger.debug(f"[LOCK CLEANUP] 세션 종료 전 제거됨: {lockfile}")
                except Exception as e:
                    logger.warning(f"[LOCK CLEANUP 실패] {e}")
        except Exception as e:
            logger.warning(f"[LOCK CLEANUP 스캔 실패] {e}")
    
    def request_flush(self, user_name: str, delay: float = None):
        """
        flush_to_excel()을 바로 실행하지 않고, 약간 지연시켜
        동시에 여러 요청이 들어올 때 한 번만 실행되게 병합한다.
        
        Args:
            user_name: 사용자 이름
            delay: 지연 시간 (초), None이면 기본값(self._flush_delay) 사용
        """
        if delay is None:
            delay = self._flush_delay
        
        if self._pending_flush.get(user_name, False):
            logger.debug(f"[FLUSH REQUEST] {user_name} - 이미 예약된 flush 있음 - 병합됨")
            return
        
        self._pending_flush[user_name] = True
        logger.debug(f"[FLUSH REQUEST] {user_name} - flush 예약됨 - {delay:.1f}초 후 실행 예정")
        
        def _delayed_flush():
            try:
                time.sleep(delay)
                with self._flush_lock:
                    logger.debug(f"[FLUSH THREAD] {user_name} - 실행 시작")
                    self.flush_to_excel(user_name)
            except Exception as e:
                logger.error(f"[FLUSH THREAD ERROR] {user_name} - {e}")
                import traceback
                logger.error(traceback.format_exc())
            finally:
                self._pending_flush[user_name] = False
                logger.debug(f"[FLUSH THREAD] {user_name} - 실행 완료")
        
        threading.Thread(target=_delayed_flush, daemon=True).start()
    
    def flush_to_excel(self, user_name: str):
        excel_path = self.get_user_excel_path(user_name)
        
        try:

            logger.info(f"[FLUSH DEBUG] flush 시작 - 전체 버퍼 키: {list(self._buffered_changes.keys())}")
            
            user_buffers = {k: v for k, v in self._buffered_changes.items() if k[0] == user_name}
            
            logger.info(f"[FLUSH DEBUG] {user_name} 버퍼 상태: {[(k, len(v)) for k, v in user_buffers.items()]}")
            
            if not user_buffers:
                logger.debug(f"[FLUSH] {user_name} 버퍼가 비어있음")
                return
            
            excel_file = self.load_user_excel(user_name)
            excel_data = {}
            
            if excel_file:

                for sheet in excel_file.sheet_names:
                    excel_data[sheet] = self.safe_load_sheet(user_name, sheet)
            else:

                for sheet_name in SHEET_SCHEMAS.keys():
                    excel_data[sheet_name] = pd.DataFrame(columns=SHEET_SCHEMAS[sheet_name])
            
            for (uname, sheet_name), records in user_buffers.items():

                logger.debug(f"[FLUSH DEBUG] 처리 중: 시트={sheet_name}, 레코드 수={len(records) if records else 0}")
                
                if not records:
                    logger.debug(f"[FLUSH DEBUG] 건너뜀: 시트={sheet_name} (레코드 없음)")
                    continue
                
                try:
                    schema = SHEET_SCHEMAS.get(sheet_name, SHEET_SCHEMAS["사용자정보KV"])

                    ordered_records = []
                    for record in records:
                        ordered_record = {col: str(record.get(col, "")).strip() if record.get(col) is not None else "" for col in schema}
                        ordered_records.append(ordered_record)
                    
                    df_new = pd.DataFrame(ordered_records, columns=schema)
                    
                    if sheet_name == "물건위치":
                        logger.debug(f"[FLUSH DEBUG] 물건위치 DataFrame:\n{df_new.head()}")
                        logger.debug(f"[FLUSH DEBUG] 물건위치 컬럼 순서: {list(df_new.columns)}")
                except Exception as e:
                    logger.error(f"[FLUSH ERROR] DataFrame 생성 실패: 시트={sheet_name}, 오류={e}, 레코드={records}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
                
                if sheet_name in excel_data:
                    df_existing = excel_data[sheet_name]
                    df_all = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    df_all = df_new

                # ✅ 일정 시트 처리:
                #   - 중복 제거: 날짜(일만) + 제목(공백 정규화) + 시간 + 장소가 같으면 중복
                #   - '날짜' 컬럼은 발화 시간이 아닌 '일정 날짜(YYYY-MM-DD)'만 저장
                if sheet_name == "일정" and not df_all.empty:
                    try:
                        if all(col in df_all.columns for col in ["날짜", "제목", "시간", "장소"]):
                            for col in ["날짜", "제목", "시간", "장소"]:
                                df_all[col] = df_all[col].fillna("").astype(str).str.strip()
                            # 제목 공백 정규화 / 날짜(일만) 비교용 컬럼
                            df_all["_제목_norm"] = df_all["제목"].apply(lambda x: _normalize_schedule_title(str(x)))
                            df_all["_날짜_norm"] = df_all["날짜"].apply(lambda x: _normalize_date_for_compare(str(x)))

                            # 🔹 과거 버그로 인해 같은 제목/시간/장소인데 날짜가 비어 있는 행과 채워진 행이 같이 있는 경우
                            #    → 날짜가 채워진 행만 남기고, 비어 있는 행은 정리
                            try:
                                to_drop_idx = []
                                group_cols = ["_제목_norm", "시간", "장소"]
                                for _, group in df_all.groupby(group_cols, dropna=False):
                                    has_non_empty = (group["_날짜_norm"] != "").any()
                                    has_empty = (group["_날짜_norm"] == "").any()
                                    if has_non_empty and has_empty:
                                        empty_idx = group.index[group["_날짜_norm"] == ""].tolist()
                                        to_drop_idx.extend(empty_idx)
                                if to_drop_idx:
                                    df_all = df_all.drop(index=to_drop_idx)
                            except Exception:
                                pass

                            # 정렬은 datetime으로 하되, 최종 저장은 '_날짜_norm'(YYYY-MM-DD 또는 빈 문자열)만 유지
                            if "_날짜_norm" in df_all.columns:
                                _dt = pd.to_datetime(df_all["_날짜_norm"], errors="coerce")
                                df_all = df_all.assign(_날짜_dt=_dt)
                                df_all = df_all.sort_values("_날짜_dt", na_position="last")

                            df_all = df_all.drop_duplicates(subset=["_날짜_norm", "_제목_norm", "시간", "장소"], keep="last")

                            # 최종적으로 '날짜'는 '_날짜_norm'(YYYY-MM-DD 또는 빈 문자열)로 저장
                            if "날짜" in df_all.columns and "_날짜_norm" in df_all.columns:
                                df_all["날짜"] = df_all["_날짜_norm"]
                            df_all = df_all.drop(columns=["_제목_norm", "_날짜_norm", "_날짜_dt"], errors="ignore")

                            logger.debug(f"[DUPLICATE CHECK] 일정 중복 제거 및 날짜 형식 정리 완료: {len(df_all)}개 레코드")
                    except Exception as e:
                        logger.warning(f"[DUPLICATE CHECK] 일정 중복 제거/날짜 정리 실패: {e}")
                
                if sheet_name == "물건위치" and not df_all.empty:
                    try:
                        # 동일 물건에 대해 가장 최신 위치만 유지
                        # - 동일 물건 + 다른 위치: 기존 행 삭제, 새 행으로 덮어쓰기
                        # - 동일 물건 + 동일 위치: 중복 제거 (keep last)
                        if "물건이름" in df_all.columns:
                            for col in ["물건이름", "장소", "세부위치"]:
                                if col in df_all.columns:
                                    df_all[col] = df_all[col].fillna("").astype(str).str.strip()

                            # 날짜 컬럼이 있으면 datetime으로 변환 후 정렬
                            if "날짜" in df_all.columns:
                                df_all["날짜"] = pd.to_datetime(df_all["날짜"], errors="coerce")
                                df_all = df_all.sort_values("날짜", na_position="last")
                            else:
                                logger.debug("[DUPLICATE CHECK] '날짜' 컬럼 없음 → 정렬 없이 중복 제거 수행")

                            # 동일 물건이름 기준으로 마지막(최신) 행만 남김
                            df_all = df_all.drop_duplicates(subset=["물건이름"], keep="last")

                            logger.debug(f"[DUPLICATE CHECK] 물건위치 중복 제거 후: {len(df_all)}개 레코드")
                    except Exception as e:
                        logger.warning(f"[FLUSH WARN] 물건위치 중복 제거 실패: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())

                if sheet_name == "복약정보" and not df_all.empty:
                    try:

                        if all(col in df_all.columns for col in ["약이름", "시간", "복용방법", "복용기간"]):

                            for col in ["약이름", "시간", "복용방법", "복용기간"]:
                                df_all[col] = df_all[col].fillna("").astype(str).str.strip()
                            
                            df_all = df_all.sort_values("날짜", na_position='last')
                            
                            logger.debug(f"[DUPLICATE CHECK] 복약정보 중복 제거 전: {len(df_all)}개 레코드")
                            logger.debug(f"[DUPLICATE CHECK] 샘플 데이터:\n{df_all[['약이름', '시간', '복용방법', '복용기간']].head()}")
                            
                            df_all = df_all.drop_duplicates(
                                subset=["약이름", "시간", "복용방법", "복용기간"],
                                keep="last"
                            )
                            
                            logger.debug(f"[DUPLICATE CHECK] 복약정보 중복 제거 후: {len(df_all)}개 레코드")
                            logger.debug(f"[FLUSH] 복약정보 중복 제거 완료: {len(df_new)}개 추가 → {len(df_all)}개 최종")
                    except Exception as e:
                        logger.warning(f"[FLUSH WARN] 복약정보 중복 제거 실패: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())

                if sheet_name == "사용자정보KV" and not df_all.empty:
                    try:

                        if "날짜" in df_all.columns and "키" in df_all.columns:
                            df_all = df_all.sort_values("날짜").drop_duplicates(subset=["키"], keep="last")
                    except Exception as _:
                        pass
                
                excel_data[sheet_name] = df_all

                logger.info(f"[FLUSH] {user_name}:{sheet_name} → {len(df_new)}개 레코드 저장 완료")
            
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                for sheet_name, df_data in excel_data.items():
                    df_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            keys_to_remove = [k for k in self._buffered_changes.keys() if k[0] == user_name]
            for k in keys_to_remove:
                del self._buffered_changes[k]
            
            remaining_buffers = [k for k in self._buffered_changes.keys() if k[0] == user_name]
            logger.info(f"[FLUSH SUMMARY] {user_name} 버퍼 상태: {remaining_buffers if remaining_buffers else '비어있음'}")
            logger.info(f"[FLUSH] {user_name} 버퍼 → 엑셀 동기화 완료")
                
        except Exception as e:
            logger.error(f"[ERROR] flush_to_excel 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())

    def save_entity(self, user_name: str, entity_type: str, entity_data: Dict[str, Any]):
        try:
            self.save_entity_data(user_name, entity_type, entity_data)
            logger.info(f"[BUFFER] {user_name}:{self._get_sheet_name(entity_type)} 엔티티 버퍼링됨 ({entity_type})")
        except Exception as e:
            logger.error(f"[ERROR] save_entity 실패: {e}")
