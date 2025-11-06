# user_excel_manager.py
"""
사용자별 개인정보를 엑셀 파일로 저장/관리하는 모듈
"""
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

# 1️⃣ 통합 시트 스키마 정의
SHEET_SCHEMAS = {
    "물건위치": ["날짜", "물건이름", "장소", "세부위치", "출처", "엔티티타입"],
    "복약정보": ["날짜", "약이름", "용량", "단위", "시간", "복용방법", "복용기간", "엔티티타입"],
    "일정": ["날짜", "제목", "시간", "장소", "정보", "엔티티타입"],
    "가족관계": ["날짜", "관계", "이름", "정보", "엔티티타입"],
    "감정기록": ["날짜", "감정", "정보", "엔티티타입"],
    "음식기록": ["날짜", "끼니", "시간", "메뉴", "엔티티타입"],
    "사용자정보KV": ["날짜", "키", "값", "출처", "확신도", "엔티티타입"],
    "대화기록": ["날짜", "시간", "대화요약"],  # 대화 기록은 별도 스키마
}

# 패키지 디렉토리 찾기
def _get_package_dir():
    """life_assist_dm 패키지 디렉토리 반환"""
    current_file = Path(__file__).resolve()
    # life_assist_dm/life_assist_dm/user_excel_manager.py -> life_assist_dm
    package_dir = current_file.parent.parent
    return package_dir

class UserExcelManager:
    """사용자별 엑셀 파일 관리"""
    
    def __init__(self, base_dir: str = None):
        """
        Args:
            base_dir: 엑셀 파일이 저장될 기본 디렉토리 (None이면 패키지 디렉토리/user_information)
        """
        if base_dir is None:
            # 기본값: 패키지 디렉토리/user_information
            package_dir = _get_package_dir()
            self.base_dir = package_dir / "user_information"
        else:
            self.base_dir = Path(os.path.expanduser(base_dir))
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.debug(f"엑셀 파일 저장 경로: {self.base_dir}")
        # 버퍼링: (user_name, sheet_name) -> [records]
        self._buffered_changes = defaultdict(list)
        # ✅ flush 타이밍 경쟁 방지: lock과 pending 플래그
        self._flush_lock = threading.Lock()
        self._pending_flush = {}  # user_name별 중복 방지 플래그: {user_name: bool}
        self._flush_delay = 1.0      # flush 지연 시간 (초)
    
    # -----------------------------
    #  시트 매핑 유틸
    # -----------------------------
    def _get_sheet_name(self, entity_type: str) -> str:
        """엔티티 타입을 시트 이름으로 변환"""
        mapping = {
            "물건": "물건위치",
            "user.물건": "물건위치",  # ✅ 엔티티 키 형식도 지원
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
            "취향": "사용자정보KV",  # ✅ 취향/선호도 사용자 정보이므로 사용자정보KV로 이동
            "선호": "사용자정보KV",  # ✅ 취향/선호도 사용자 정보이므로 사용자정보KV로 이동
            "기념일": "사용자정보KV",  # ✅ 기념일도 사용자 정보이므로 사용자정보KV로 저장
            "취미": "사용자정보KV",  # ✅ 취미도 사용자 정보이므로 사용자정보KV로 저장
        }
        # 매핑되지 않은 엔티티 타입도 모두 사용자정보KV로 저장 (기타 시트 제거)
        sheet_name = mapping.get(entity_type, "사용자정보KV")
        if entity_type not in mapping:
            logger.info(f"[INFO] '{entity_type}' 엔티티 타입이 매핑되지 않아 사용자정보KV로 저장")
        return sheet_name
        
    def get_user_excel_path(self, user_name: str) -> Path:
        """사용자별 엑셀 파일 경로 반환"""
        # 한글 파일명 지원
        file_name = f"{user_name}.xlsx"
        return self.base_dir / file_name
    
    def load_user_excel(self, user_name: str) -> Optional[pd.ExcelFile]:
        """사용자 엑셀 파일 로드"""
        excel_path = self.get_user_excel_path(user_name)
        if not excel_path.exists():
            return None
        try:
            return pd.ExcelFile(excel_path)
        except Exception as e:
            logger.error(f"엑셀 파일 로드 실패: {e}")
            return None
    
    def load_sheet_data(self, user_name: str, sheet_name: str) -> pd.DataFrame:
        """특정 시트 데이터 로드"""
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
    
    # -----------------------------
    # 🧩 안전한 로드 함수 (스키마 정렬)
    # -----------------------------
    def safe_load_sheet(self, user_name: str, sheet_name: str) -> pd.DataFrame:
        """엑셀 시트를 안전하게 로드 + 스키마 일관성 보장"""
        try:
            df = self.load_sheet_data(user_name, sheet_name)
            schema = SHEET_SCHEMAS.get(sheet_name, [])
            if df is None or df.empty:
                return pd.DataFrame(columns=schema)
            # 누락된 컬럼 추가
            for col in schema:
                if col not in df.columns:
                    df[col] = ""
            # 스키마 순서에 맞춰 정렬
            return df[schema]
        except Exception as e:
            logger.error(f"[ERROR] safe_load_sheet 실패: {e}")
            schema = SHEET_SCHEMAS.get(sheet_name, [])
            return pd.DataFrame(columns=schema)
    
    def save_data_to_sheet(self, user_name: str, sheet_name: str, data: List[Dict[str, Any]], 
                           append: bool = True):
        """시트에 데이터 저장"""
        excel_path = self.get_user_excel_path(user_name)
        
        def _cleanup_lockfile(path: Path):
            """openpyxl이 남길 수 있는 임시 .lock 파일을 정리한다."""
            try:
                lock_path = Path(str(path) + ".lock")
                if lock_path.exists():
                    lock_path.unlink(missing_ok=True)
                    logger.debug(f"[LOCK CLEANUP] Lock 파일 제거됨: {lock_path}")
            except Exception as e:
                logger.warning(f"[LOCK CLEANUP 실패] {e}")
        
        # 기존 데이터 로드
        existing_data = []
        if excel_path.exists() and append:
            try:
                df_existing = self.load_sheet_data(user_name, sheet_name)
                existing_data = df_existing.to_dict('records')
            except Exception as e:
                logger.warning(f"기존 데이터 로드 실패: {e}")
        
        # 새 데이터 추가
        if append:
            existing_data.extend(data)
        else:
            existing_data = data
        
        # DataFrame 생성
        df = pd.DataFrame(existing_data)

        # 시트별 표준 스키마에 맞춰 컬럼 순서/존재 강제
        schema = SHEET_SCHEMAS.get(sheet_name, [])
        if schema:
            # 누락 컬럼 추가
            for col in schema:
                if col not in df.columns:
                    df[col] = ""
            # 스키마 순서에 맞춰 정렬
            df = df[schema]
        
        # 엑셀 파일 저장 (시트 단위 교체 저장 최적화)
        try:
            mode = 'a' if excel_path.exists() else 'w'
            with pd.ExcelWriter(
                excel_path, engine='openpyxl', mode=mode, if_sheet_exists='replace'
            ) as writer:
                df.to_excel(writer, sheet_name=sheet_name, index=False)
            # openpyxl이 남긴 임시 잠금 파일 정리
            _cleanup_lockfile(excel_path)
        except TypeError:
            # pandas/openpyxl 구버전 호환: 기존 방식으로 전체 재작성
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
    
    # -----------------------------
    # 🧩 엔티티 표준화 함수
    # -----------------------------
    def _convert_duration_to_date_range(self, duration_str: str) -> str:
        """복용기간을 날짜 범위로 변환 (예: "15일치" → "2025-11-06~2025-11-21")"""
        if not duration_str:
            return ""
        
        import re
        from datetime import datetime, timedelta
        
        # 이미 날짜 범위 형식인 경우 그대로 반환 (예: "2025-11-06~2025-11-21")
        if "~" in duration_str or "-" in duration_str:
            # 날짜 형식이면 그대로 반환
            if re.match(r"\d{4}-\d{2}-\d{2}", duration_str.split("~")[0].strip()):
                return duration_str
        
        # 기간 추출 (예: "15일치", "7일치", "2주일치", "1개월치")
        duration_match = re.search(r"(\d+)\s*(일|주|개월|년)", duration_str)
        if not duration_match:
            # 매칭되지 않으면 원본 반환
            return duration_str
        
        days_to_add = 0
        number = int(duration_match.group(1))
        unit = duration_match.group(2)
        
        if unit == "일":
            days_to_add = number
        elif unit == "주":
            days_to_add = number * 7
        elif unit == "개월":
            days_to_add = number * 30  # 대략 30일
        elif unit == "년":
            days_to_add = number * 365
        
        # 오늘 날짜를 시작일로 설정
        start_date = datetime.now()
        end_date = start_date + timedelta(days=days_to_add - 1)  # -1은 시작일 포함하여 계산
        
        # 날짜 범위 형식으로 반환
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        return f"{start_str}~{end_str}"
    
    def _normalize_entity(self, entity_type: str, data: dict) -> dict:
        """엔티티별 표준 키 이름과 값 정규화"""
        norm = {}
        try:
            if entity_type in ["물건", "user.물건"]:
                # 물건이름: 여러 가능한 키 이름 지원
                norm["물건이름"] = data.get("물건이름") or data.get("이름", "")
                # 장소와 세부위치 분리 처리
                norm["장소"] = str(data.get("장소", "")).strip()
                norm["세부위치"] = str(data.get("세부위치", "")).strip()
                # 하위 호환성: "위치" 필드가 있으면 장소와 세부위치로 분리 시도
                if not norm["장소"] and not norm["세부위치"]:
                    location = str(data.get("위치", "")).strip()
                    if location:
                        # 위치에서 장소 키워드 추출 (긴 키워드부터 체크하여 "내 방", "안방" 같은 복합 키워드 우선 처리)
                        import re
                        # "내 방", "내방" 같은 복합 패턴을 먼저 체크
                        if "내 방" in location or "내방" in location:
                            # "내 방 안에" → 장소="내 방", 세부위치="안에"
                            # "내 방 안" → 장소="내 방", 세부위치="안"
                            if location.startswith("내 방") or location.startswith("내방"):
                                # "내 방 안에" → "안에" 추출
                                remaining = location.replace("내 방", "").replace("내방", "").strip()
                                norm["장소"] = "내 방" if "내 방" in location else "내방"
                                norm["세부위치"] = remaining
                            else:
                                # "내 방"이 중간에 있는 경우
                                norm["장소"] = "내 방" if "내 방" in location else "내방"
                                norm["세부위치"] = location.replace("내 방", "").replace("내방", "").strip()
                        else:
                            # 일반 장소 키워드 체크 (긴 키워드부터)
                            room_keywords = ["안방", "다용도실", "화장실", "주방", "거실", "침실", "현관", "베란다", "방"]
                            room_keywords_sorted = sorted(room_keywords, key=len, reverse=True)
                            for room in room_keywords_sorted:
                                if room in location:
                                    norm["장소"] = room
                                    norm["세부위치"] = location.replace(room, "").strip()
                                    break
                        if not norm["장소"]:
                            # 장소를 찾지 못한 경우 전체를 세부위치로
                            norm["세부위치"] = location
                # 출처: 추출방법 또는 출처 필드 사용
                norm["출처"] = data.get("출처") or data.get("추출방법", "사용자 발화")
                # 엔티티타입은 나중에 추가됨
            elif entity_type in ["약", "user.약"]:
                # 약 필드명 통일: "약명" → "약이름"으로 정규화
                norm["약이름"] = data.get("약이름") or data.get("약명") or data.get("이름", "")
                # 용량과 단위를 별도로 저장 (엑셀 컬럼이 분리되어 있음)
                dose = str(data.get("용량", "")).strip()
                unit = str(data.get("단위", "")).strip()
                norm["용량"] = dose if dose else ""
                norm["단위"] = unit if unit else ""
                # 복용시간: 시간대 또는 시간 필드 사용
                norm["시간"] = data.get("시간대") or data.get("시간") or data.get("복용시간", "")
                # ✅ 복용여부 필드 제거 (대화기록 참고로 변경)
                # ✅ 복용방법 필드 추가 (식후 30분, 공복에 등)
                norm["복용방법"] = data.get("복용방법") or data.get("메모") or ""
                # ✅ 복용기간 필드 추가 (기간을 날짜 범위로 변환)
                복용기간_원본 = data.get("복용기간") or ""
                if 복용기간_원본:
                    # "15일치" 같은 기간 표현을 날짜 범위로 변환
                    norm["복용기간"] = self._convert_duration_to_date_range(복용기간_원본)
                else:
                    norm["복용기간"] = ""
            elif entity_type == "일정":
                norm["제목"] = data.get("제목", "")
                # 날짜 정규화 (어제/오늘/내일 등 → YYYY-MM-DD)
                # support_chains.py에서 이미 정규화된 경우도 있지만, 여기서도 정규화하여 일관성 유지
                date_value = data.get("날짜", "")
                if date_value:
                    try:
                        from life_assist_dm.life_assist_dm.support_chains import _normalize_date_to_iso
                        date_str = str(date_value).strip()
                        if date_str and date_str.lower() not in ("nan", "none", ""):
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
                # 날짜 정규화 (어제/오늘/내일 등 → YYYY-MM-DD)
                # 일정과 동일하게 현재 날짜 기준으로 변환된 날짜 저장
                date_value = data.get("날짜", "")
                if date_value:
                    try:
                        from life_assist_dm.life_assist_dm.support_chains import _normalize_date_to_iso
                        date_str = str(date_value).strip()
                        if date_str and date_str.lower() not in ("nan", "none", ""):
                            norm["날짜"] = _normalize_date_to_iso(date_str)
                        else:
                            # 날짜가 없으면 오늘 날짜로 설정
                            norm["날짜"] = datetime.now().strftime("%Y-%m-%d")
                    except Exception as e:
                        logger.warning(f"날짜 정규화 실패: {e}, 원본 값 사용: {date_value}")
                        norm["날짜"] = str(date_value) if date_value else ""
                else:
                    # 날짜가 없으면 오늘 날짜로 설정
                    norm["날짜"] = datetime.now().strftime("%Y-%m-%d")
            elif entity_type == "정서" or entity_type == "감정":
                norm["감정"] = data.get("감정") or data.get("상태") or data.get("증상", "")
                norm["정보"] = data.get("정보", "") or data.get("원문", "")
            elif entity_type == "가족":
                norm["관계"] = data.get("관계", "")
                norm["이름"] = data.get("이름", "")
                norm["정보"] = data.get("정보", "")
            elif entity_type in ["취향", "선호", "기념일", "취미"]:
                # 이 타입들은 사용자정보KV로 저장되므로 정규화 불필요 (특별 처리됨)
                # 하지만 fallback을 위해 기본 처리
                norm["내용"] = data.get("내용") or json.dumps(data, ensure_ascii=False)
                norm["정보"] = data.get("정보") or entity_type
            else:
                # 알 수 없는 타입도 사용자정보KV로 저장되므로 정규화 불필요 (특별 처리됨)
                norm["내용"] = json.dumps(data, ensure_ascii=False)
                norm["정보"] = ""
        except Exception as e:
            logger.warning(f"엔티티 정규화 중 오류: {e}")
            norm["내용"] = json.dumps(data, ensure_ascii=False)
            norm["정보"] = ""
        return norm
    
    def save_entity_data(self, user_name: str, entity_type: str, data: Dict[str, Any]):
        """안전한 엔티티 저장 (정규화 + 스키마 일관성 보장)"""
        # 사용자 이름 유효성 검증
        if not user_name or not str(user_name).strip() or user_name == "사용자":
            logger.warning(f"[WARN] 잘못된 사용자명으로 저장 시도: {user_name}")
            return
        
        try:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            sheet_name = self._get_sheet_name(entity_type)
            
            # 사용자 관련 엔티티는 모두 특별 처리 (KV 시트 저장)
            # 사용자 기본 정보, 취향/선호, 기념일, 취미 등 모든 사용자 정보
            # 매핑되지 않은 엔티티 타입도 사용자정보KV로 저장 (기타 시트 제거)
            non_user_entity_types = ["물건", "약", "일정", "식사", "음식", "정서", "감정", "가족", "user.물건", "user.약", "user.일정", "user.식사", "user.음식", "user.건강상태", "user.가족"]
            if entity_type not in non_user_entity_types:
                try:
                    from .dialog_manager.config.config_loader import get_excel_sheets
                    sheets = get_excel_sheets()
                    kv_sheet = sheets.get("user_info_kv", "사용자정보KV")
                    
                    # 사용자 정보 정규화
                    normalized_user = {}
                    import re
                    
                    if entity_type == "사용자":
                        # 나이: 숫자 추출 후 '살' 접미사 표준화
                        if "나이" in data and data["나이"]:
                            m = re.search(r"(\d+)", str(data["나이"]))
                            if m:
                                normalized_user["나이"] = f"{m.group(1)}살"
                        # 학교: 발화 전처리 후 '...학교'만 추출
                        if "학교" in data and data["학교"]:
                            raw_school = str(data["학교"]).strip()
                            raw_school = re.sub(r"^(?:나는|난|저는)\s*", "", raw_school)
                            raw_school = re.sub(r"\s*(?:에\s*다녀.*|다녀.*)$", "", raw_school)
                            m = re.search(r"([가-힣A-Za-z\s]+?(?:중학교|고등학교|대학교|초등학교|학교))", raw_school)
                            if m:
                                normalized_user["학교"] = m.group(1).strip()
                        # 이름/별칭/직업/취미/회사/인턴은 그대로
                        for k in ["이름", "별칭", "직업", "취미", "회사", "인턴"]:
                            if k in data and data[k]:
                                normalized_user[k] = data[k]
                    elif entity_type in ["취향", "선호"]:
                        # 취향/선호는 "내용" 필드를 "취향" 키로 저장
                        content = data.get("내용", "") or data.get("값", "") or json.dumps(data, ensure_ascii=False)
                        if content:
                            normalized_user["취향"] = content
                    elif entity_type == "기념일":
                        # 기념일은 "제목"과 "날짜"를 키-값으로 저장
                        if "제목" in data and data["제목"]:
                            normalized_user["기념일"] = f"{data.get('제목', '')} ({data.get('날짜', '')})"
                        elif "날짜" in data and data["날짜"]:
                            normalized_user["기념일"] = data.get("날짜", "")
                    elif entity_type == "취미":
                        # 취미는 "이름" 필드를 "취미" 키로 저장
                        hobby = data.get("이름", "") or data.get("취미", "") or ""
                        if hobby:
                            normalized_user["취미"] = hobby
                    else:
                        # 기타 매핑되지 않은 엔티티 타입은 JSON으로 저장
                        import json
                        entity_json = json.dumps(data, ensure_ascii=False)
                        normalized_user[entity_type] = entity_json
                    
                    # KV 형식으로 변환
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
                        # 취향/선호/기념일/취미/기타 모든 사용자 정보를 키-값으로 저장
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
                        # 버퍼에 추가
                        self._buffered_changes[(user_name, kv_sheet)].extend(kv_rows)
                        logger.info(f"[BUFFER] {user_name}:{kv_sheet} 엔티티 버퍼링됨 ({entity_type})")
                except Exception as e:
                    # 테스트/단독 실행 호환: 상대 임포트 실패 시 기본 시트명 사용
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
            
            # 1️⃣ 데이터 정규화
            normalized = self._normalize_entity(entity_type, data)
            # 날짜는 _normalize_entity에서 이미 정규화되었으므로, 빈 값만 체크
            date_value = normalized.get("날짜", "")
            if not date_value or str(date_value).strip() == "" or str(date_value).lower() in ("nan", "none"):
                normalized["날짜"] = now.split()[0]  # 날짜만 (YYYY-MM-DD)
            else:
                normalized["날짜"] = str(date_value).strip()
            normalized["엔티티타입"] = entity_type
            
            # 2️⃣ 스키마 강제 정렬
            schema = SHEET_SCHEMAS.get(sheet_name, SHEET_SCHEMAS["사용자정보KV"])
            for col in schema:
                if col not in normalized:
                    normalized[col] = ""
            
            # ✅ 디버깅: normalized 딕셔너리 확인
            if entity_type in ["물건", "user.물건"]:
                logger.debug(f"[SAVE DEBUG] 물건 저장 - normalized: {normalized}")
                logger.debug(f"[SAVE DEBUG] 물건 저장 - schema: {schema}")
            
            record = {k: str(normalized[k]) if normalized[k] is not None else "" for k in schema}
            
            # ✅ 디버깅: record 딕셔너리 확인
            if entity_type in ["물건", "user.물건"]:
                logger.debug(f"[SAVE DEBUG] 물건 저장 - record: {record}")
            
            # 3️⃣ 버퍼에 추가 (즉시 저장하지 않음)
            buffer_key = (user_name, sheet_name)
            self._buffered_changes[buffer_key].append(record)
            # ✅ 디버깅: 버퍼링 직후 상태 확인
            logger.info(f"[BUFFER] {user_name}:{sheet_name} 엔티티 버퍼링됨 ({entity_type})")
            logger.debug(f"[BUFFER DEBUG] 버퍼링 직후 - 버퍼 키: {buffer_key}, 레코드 수: {len(self._buffered_changes[buffer_key])}")
            
        except Exception as e:
            logger.error(f"[ERROR] save_entity_data 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def save_conversation_summary(self, user_name: str, summary: str, 
                                 timestamp: Optional[str] = None):
        """대화 요약 저장"""
        if timestamp is None:
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        
        # 대화 기록은 버퍼링 방식으로 저장
        record = {
            "날짜": timestamp.split()[0],  # 날짜만
            "시간": timestamp.split()[1] if len(timestamp.split()) > 1 else "",  # 시간만
            "대화요약": summary,
            "엔티티타입": "대화기록"
        }
        
        # 버퍼에 추가 (즉시 저장하지 않음)
        key = (user_name, "대화기록")
        self._buffered_changes[key].append(record)
        logger.info(f"[BUFFER] 대화 요약 버퍼링됨: {user_name}")

        #  조건부 배치 flush: 대화요약이 3건 이상 누적되면 일괄 저장
        #  request_flush() 사용하여 지연 병합 처리
        try:
            if len(self._buffered_changes.get(key, [])) >= 3:
                self.request_flush(user_name)
                logger.info(f"[FLUSH] 대화요약 누적 3회 → Excel 동기화 예약 ({user_name})")
            else:
                logger.debug(f"[BUFFER] 대화요약 누적 {len(self._buffered_changes.get(key, []))}회 (미flush)")
        except Exception:
            pass
    
    # 제거됨: SQLite 동기화는 더 이상 사용하지 않습니다
    
    def initialize_user_excel(self, user_name: str):
        """새 사용자 엑셀 파일 초기화 (스키마 기반)"""
        excel_path = self.get_user_excel_path(user_name)
        
        # SHEET_SCHEMAS 기반으로 초기화
        with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
            for sheet_name, columns in SHEET_SCHEMAS.items():
                df = pd.DataFrame(columns=columns)
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        logger.info(f"사용자 엑셀 파일 초기화 완료: {user_name}")
    
    def user_exists(self, user_name: str) -> bool:
        """사용자 파일 존재 여부 확인"""
        return self.get_user_excel_path(user_name).exists()

    def cleanup_all_locks(self):
        """사용자 정보 디렉터리 내 잔존 .xlsx.lock 파일 일괄 정리"""
        try:
            for lockfile in self.base_dir.glob("*.xlsx.lock"):
                try:
                    lockfile.unlink(missing_ok=True)
                    logger.debug(f"[LOCK CLEANUP] 세션 종료 전 제거됨: {lockfile}")
                except Exception as e:
                    logger.warning(f"[LOCK CLEANUP 실패] {e}")
        except Exception as e:
            logger.warning(f"[LOCK CLEANUP 스캔 실패] {e}")
    
    # -----------------------------
    # 🧩 flush 메서드 (버퍼 → Excel 반영)
    # -----------------------------
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
        
        # 이미 예약된 flush가 있으면 중복 실행 방지 (user_name별로 관리)
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
        """버퍼 내용을 엑셀로 동기화"""
        excel_path = self.get_user_excel_path(user_name)
        
        try:
            # ✅ 디버깅: flush 시작 시점 버퍼 상태 확인
            logger.info(f"[FLUSH DEBUG] flush 시작 - 전체 버퍼 키: {list(self._buffered_changes.keys())}")
            
            # 버퍼에서 해당 사용자 데이터만 추출
            user_buffers = {k: v for k, v in self._buffered_changes.items() if k[0] == user_name}
            
            # ✅ 디버깅: 사용자별 버퍼 상태 확인
            logger.info(f"[FLUSH DEBUG] {user_name} 버퍼 상태: {[(k, len(v)) for k, v in user_buffers.items()]}")
            
            if not user_buffers:
                logger.debug(f"[FLUSH] {user_name} 버퍼가 비어있음")
                return
            
            # 기존 엑셀 파일 로드 또는 초기화
            excel_file = self.load_user_excel(user_name)
            excel_data = {}
            
            if excel_file:
                # 기존 시트들 로드
                for sheet in excel_file.sheet_names:
                    excel_data[sheet] = self.safe_load_sheet(user_name, sheet)
            else:
                # 새 파일 초기화 (스키마 기반)
                for sheet_name in SHEET_SCHEMAS.keys():
                    excel_data[sheet_name] = pd.DataFrame(columns=SHEET_SCHEMAS[sheet_name])
            
            # 버퍼 데이터를 각 시트에 추가
            for (uname, sheet_name), records in user_buffers.items():
                # ✅ 디버깅: 각 시트별 처리 시작
                logger.debug(f"[FLUSH DEBUG] 처리 중: 시트={sheet_name}, 레코드 수={len(records) if records else 0}")
                
                if not records:
                    logger.debug(f"[FLUSH DEBUG] 건너뜀: 시트={sheet_name} (레코드 없음)")
                    continue
                
                # DataFrame 생성 및 스키마 정렬
                try:
                    schema = SHEET_SCHEMAS.get(sheet_name, SHEET_SCHEMAS["사용자정보KV"])
                    # ✅ 스키마 순서대로 레코드 재정렬 (컬럼 순서 보장)
                    ordered_records = []
                    for record in records:
                        ordered_record = {col: str(record.get(col, "")).strip() if record.get(col) is not None else "" for col in schema}
                        ordered_records.append(ordered_record)
                    
                    df_new = pd.DataFrame(ordered_records, columns=schema)
                    
                    # ✅ 디버깅: 물건위치 시트 저장 시 확인
                    if sheet_name == "물건위치":
                        logger.debug(f"[FLUSH DEBUG] 물건위치 DataFrame:\n{df_new.head()}")
                        logger.debug(f"[FLUSH DEBUG] 물건위치 컬럼 순서: {list(df_new.columns)}")
                except Exception as e:
                    logger.error(f"[FLUSH ERROR] DataFrame 생성 실패: 시트={sheet_name}, 오류={e}, 레코드={records}")
                    import traceback
                    logger.error(traceback.format_exc())
                    continue
                
                # 기존 데이터와 병합
                if sheet_name in excel_data:
                    df_existing = excel_data[sheet_name]
                    df_all = pd.concat([df_existing, df_new], ignore_index=True)
                else:
                    df_all = df_new

                # ✅ 복약정보 시트는 약명+시간+방법+기간 기준으로 중복 제거 (동일한 복용 정보는 한 번만 저장)
                if sheet_name == "복약정보" and not df_all.empty:
                    try:
                        # 중복 기준: 약이름 + 시간 + 복용방법 + 복용기간이 모두 동일한 경우
                        if all(col in df_all.columns for col in ["약이름", "시간", "복용방법", "복용기간"]):
                            # ✅ 데이터 정규화: None/NaN → 빈 문자열, 공백 제거
                            for col in ["약이름", "시간", "복용방법", "복용기간"]:
                                df_all[col] = df_all[col].fillna("").astype(str).str.strip()
                            
                            # 날짜 오름차순 정렬 후 중복 제거 (최신값 유지)
                            df_all = df_all.sort_values("날짜", na_position='last')
                            
                            # ✅ 디버깅: 중복 제거 전 데이터 확인
                            logger.debug(f"[DUPLICATE CHECK] 복약정보 중복 제거 전: {len(df_all)}개 레코드")
                            logger.debug(f"[DUPLICATE CHECK] 샘플 데이터:\n{df_all[['약이름', '시간', '복용방법', '복용기간']].head()}")
                            
                            # 중복 제거: 약이름, 시간, 복용방법, 복용기간이 모두 동일한 경우
                            df_all = df_all.drop_duplicates(
                                subset=["약이름", "시간", "복용방법", "복용기간"],
                                keep="last"  # 최신값 유지
                            )
                            
                            logger.debug(f"[DUPLICATE CHECK] 복약정보 중복 제거 후: {len(df_all)}개 레코드")
                            logger.debug(f"[FLUSH] 복약정보 중복 제거 완료: {len(df_new)}개 추가 → {len(df_all)}개 최종")
                    except Exception as e:
                        logger.warning(f"[FLUSH WARN] 복약정보 중복 제거 실패: {e}")
                        import traceback
                        logger.debug(traceback.format_exc())

                # ✅ 사용자정보KV는 키 기준 최신값으로 update (중복 제거)
                if sheet_name == "사용자정보KV" and not df_all.empty:
                    try:
                        # 날짜 오름차순 정렬 후 같은 키는 마지막(최신)만 유지
                        if "날짜" in df_all.columns and "키" in df_all.columns:
                            df_all = df_all.sort_values("날짜").drop_duplicates(subset=["키"], keep="last")
                    except Exception as _:
                        pass
                
                excel_data[sheet_name] = df_all
                # 시트별 저장 로그를 일관되게 남김 (감정기록 포함)
                logger.info(f"[FLUSH] {user_name}:{sheet_name} → {len(df_new)}개 레코드 저장 완료")
            
            # 전체 엑셀 파일 저장
            with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                for sheet_name, df_data in excel_data.items():
                    df_data.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # 버퍼에서 해당 사용자 데이터 제거
            keys_to_remove = [k for k in self._buffered_changes.keys() if k[0] == user_name]
            for k in keys_to_remove:
                del self._buffered_changes[k]
            
            # ✅ 디버깅: flush 완료 후 버퍼 상태 확인
            remaining_buffers = [k for k in self._buffered_changes.keys() if k[0] == user_name]
            logger.info(f"[FLUSH SUMMARY] {user_name} 버퍼 상태: {remaining_buffers if remaining_buffers else '비어있음'}")
            logger.info(f"[FLUSH] {user_name} 버퍼 → 엑셀 동기화 완료")
                
        except Exception as e:
            logger.error(f"[ERROR] flush_to_excel 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())

    # -----------------------------
    # 편의 저장 함수 (엔티티 단위)
    # -----------------------------
    def save_entity(self, user_name: str, entity_type: str, entity_data: Dict[str, Any]):
        """엔티티 단위 저장: 타입→시트 매핑 및 정규화 포함 (save_entity_data 래퍼)"""
        try:
            self.save_entity_data(user_name, entity_type, entity_data)
            logger.info(f"[BUFFER] {user_name}:{self._get_sheet_name(entity_type)} 엔티티 버퍼링됨 ({entity_type})")
        except Exception as e:
            logger.error(f"[ERROR] save_entity 실패: {e}")
