# support_chains.py
from __future__ import annotations
import os, csv, json, re, random, logging, traceback
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Any
import pandas as pd

# memory.py의 상수 import
try:
    from .memory import TIME_OF_DAY_KEYWORDS, KOREAN_NUMBERS_STR, KOREAN_NUMBERS_INT
except ImportError:
    # fallback: 직접 정의 (memory.py를 import할 수 없는 경우)
    TIME_OF_DAY_KEYWORDS = {
        "아침": ["아침", "조식", "morning", "breakfast", "기상", "일어나자마자", "일어나자 마자", "기상 후", "기상 시"],
        "점심": ["점심", "중식", "lunch"],
        "저녁": ["저녁", "석식", "dinner", "evening"]
    }
    KOREAN_NUMBERS_STR = {
        "한": "1", "두": "2", "세": "3", "네": "4", "다섯": "5",
        "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9", "열": "10"
    }
    KOREAN_NUMBERS_INT = {
        "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
        "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10
    }

# 로그 과다 관리 - rqt 메모리 과부하 방지
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)  # httpx HTTP 요청 로그 억제
logging.getLogger("httpcore").setLevel(logging.WARNING)  # httpcore 로그 억제
# ChromaDB 로깅 제거됨

logger = logging.getLogger("life_assist_physical")
logger.setLevel(logging.DEBUG)

# ===================== 공통 유틸 =====================
CMD_VERBS = (
    r"(찾아줘|찾아와|찾아와줘|찾아봐|찾아봐줘|가져와|가져와줘|갖다줘|갖다\s*줘|"
    r"꺼내와|꺼내줘|정리해|정리해줘|정돈해|정돈해줘|치워줘|치워|가져다\s*놔|놔둬)"
)

NORM_TARGET = {
    "물컵": "컵",
    "핸드 크림": "핸드크림",
    "핸드폰": "핸드폰",
}

# LOC_EN 제거됨 - LOCATION_MAP으로 통합

def _preprocess_for_parsing(text: str) -> str:
    """문장 전처리 - 연속 공백 축소."""
    t = (text or "").strip()
    return re.sub(r"\s+", " ", t)

def _clean_target(tgt: Optional[str]) -> Optional[str]:
    """타깃에서 명령/조사 꼬리 제거 + 정규화."""
    if not tgt:
        return tgt
    t = re.sub(CMD_VERBS + r".*$", "", tgt).strip()
    t = re.sub(r"(?<=[가-힣])[을를은는이가]$", "", t).strip()
    t = NORM_TARGET.get(t, t)
    return t

def _normalize_text(text: str) -> str:
    """텍스트 정규화 - 중복 공백 제거, 간단한 띄어쓰기 보정"""
    if not text:
        return ""
    t = re.sub(r"\s+", " ", str(text)).strip()
    t = t.replace(" 의자 에", " 의자에").replace(" 방 에", " 방에")
    return t

# 중복된 정규화 함수 통합: 과거 사용 호환을 위해 alias 유지
_normalize_utterance = _normalize_text

# 통합된 영문화 매핑 테이블
TARGET_MAP = {
    "핸드폰":"phone","휴대폰":"phone","아이폰":"phone","아이패드":"ipad","아이패드 pro":"ipad",
    "목걸이":"necklace","머리끈":"hair_tie","양말":"socks","볼펜":"pen","펜":"pen","립스틱":"lipstick",
    "리모컨":"remote","안경":"glasses","가위":"scissors","지갑":"wallet","우산":"umbrella",
    "핸드크림":"hand_cream","물컵":"cup","컵":"cup","옷":"clothes","쓰레기":"trash",
    # OBJ_MAP과 통합
    "지갑":"wallet","열쇠":"keys","키":"keys","핸드폰":"phone","휴대폰":"phone",
    "안경":"glasses","컵":"cup","물":"water","리모컨":"remote","서류":"document","문":"door",
    "머리끈":"hair_tie","화장지":"tissue","수건":"towel","책":"book","펜":"pen","지팡이":"cane",
    "사과":"apple","과일":"fruit","음료수":"drink","주스":"juice","우유":"milk",
    "빵":"bread","과자":"snack","음식":"food","식품":"food",
    "가방":"bag","백":"bag","핸드백":"handbag",
    "장난감":"toy","인형":"doll","공":"ball",
    "신발":"shoes","구두":"shoes","양말":"socks",
    "옷":"clothes","셔츠":"shirt","바지":"pants","치마":"skirt",
    "모자":"hat","장갑":"gloves","스카프":"scarf",
    "물건":"item","쓰레기":"trash","휴지":"tissue",
    "신문":"newspaper","핸드크림":"hand_cream",
    "이어폰":"earphone","에어팟":"airpod","담요":"blanket","젤리":"jelly",
    "케이블":"cable","충전선":"charging_cable","실내화":"slippers","슬리퍼":"slippers",
    "약통":"pill_bottle","치실":"dental_floss","치약":"toothpaste","칫솔":"toothbrush","치실컵":"dental_floss_cup",
    "연필":"pencil","접시":"plate","그릇":"bowl","수저":"spoon","포크":"fork","나이프":"knife",
    "티슈":"tissue","물티슈":"wet_tissue","카펫":"carpet","쓰레기봉투":"trash_bag",
    "커피":"coffee","차":"tea","시리얼":"cereal","약":"medicine","립스틱":"lipstick",
    "마스크":"mask","앨범":"album","명함":"business_card","이어폰케이스":"earphone_case",
    "비타민":"vitamin","화분":"plant","타월":"towel","빨래":"laundry","세탁물":"laundry",
    "세제":"detergent","세탁세제":"laundry_detergent","비누":"soap","샴푸":"shampoo","린스":"rinse","컨디셔너":"conditioner",
    "택배":"package","포장":"package","소포":"package","물건":"item","거":"item","것":"item"
}

LOCATION_MAP = {
    "거실":"living room","주방":"kitchen","부엌":"kitchen","현관":"entrance","안방":"master bedroom",
    "내 방":"room","방":"room","프린터":"printer","의자":"chair","소파":"sofa","침대":"bed","식탁":"dining table",
    # LOC_MAP과 통합
    "거실":"living room","부엌":"kitchen","주방":"kitchen","현관":"entrance","침실":"bedroom","방":"room",
    "책상":"desk","테이블":"table","소파":"sofa","신발장":"shoe cabinet","식탁":"dining table",
    "냉장고":"fridge","냉동고":"freezer","식탁":"dining table","식기장":"kitchen cabinet",
    "책꽂이":"bookshelf","서랍":"drawer","옷장":"wardrobe","수납장":"storage",
    "선반":"shelf","장바구니":"shopping_bag","바구니":"basket","빨래대":"drying_rack",
    "빨래 건조대":"drying_rack","건조대":"drying_rack","테이블":"table","식탁테이블":"dining_table",
    "식탁대":"dining_table","선반대":"shelf","화장실":"bathroom","베란다":"balcony",
    "서재":"study","현관문":"entrance_door","베란다문":"balcony_door","현관문앞":"front_of_entrance",
    "현관 앞":"front_of_entrance","문 앞":"front_of_door","문앞":"front_of_door",
    "정수기":"water_purifier","냉장고":"refrigerator","세탁기":"washing_machine","에어컨":"air_conditioner",
    "프린터":"printer","의자":"chair","침대":"bed"
}

def _to_en_target(kor: str) -> str:
    """안전한 타깃 영문화"""
    if not kor: 
        return None
    return TARGET_MAP.get(kor, kor.replace(" ", "_").lower())

def _to_en_location(kor: str) -> str:
    """안전한 위치 영문화"""
    if not kor: 
        return None
    return LOCATION_MAP.get(kor, kor.replace(" ", "_").lower())

# _to_en_location_legacy 함수 제거됨 - _to_en_location으로 통합

# ===== 전처리 & 절 분해 =====
_TAIL_NOISE = re.compile(r"(있던데|있잖아|있지|있으면|좀|그럼|그러면|그렇다면)")
_PUNCT = re.compile(r"[,]+")

def _normalize_utterance(txt: str) -> str:
    """말끝 추임과 구두점 정리"""
    txt = _PUNCT.sub(" ", txt)
    txt = _TAIL_NOISE.sub(" ", txt)
    txt = re.sub(r"\s+", " ", txt).strip()
    return txt

def _split_clauses(txt: str):
    """문장을 절로 분할"""
    return [c.strip() for c in re.split(r"(?:그리고|그래서|그니까|,|\.|/|;)", txt) if c.strip()]

def _normalize_date_to_iso(date_str: str) -> str:
    """
    날짜 문자열을 YYYY-MM-DD 형식으로 변환
    
    지원 형식:
    - "오늘", "내일", "모레"
    - "11월 8일", "11/8", "11-8"
    - "이번주 금요일", "다음주 화요일"
    - "이번 주 금요일", "다음 주 화요일"
    
    Returns:
        YYYY-MM-DD 형식의 날짜 문자열, 변환 실패 시 현재 날짜 반환
    """
    if not date_str:
        return datetime.now().strftime("%Y-%m-%d")
    
    date_str = date_str.strip()
    now = datetime.now()
    
    # 이미 YYYY-MM-DD 형식인 경우 그대로 반환
    ymd_match = re.match(r"(\d{4})-(\d{1,2})-(\d{1,2})$", date_str)
    if ymd_match:
        try:
            year, month, day = int(ymd_match.group(1)), int(ymd_match.group(2)), int(ymd_match.group(3))
            # 유효한 날짜인지 확인
            datetime(year, month, day)
            return date_str  # 이미 올바른 형식이므로 그대로 반환
        except ValueError:
            # 유효하지 않은 날짜면 아래 로직으로 처리
            pass
    
    # 상대적 날짜 처리
    if date_str == "오늘":
        return now.strftime("%Y-%m-%d")
    elif date_str == "내일":
        return (now + timedelta(days=1)).strftime("%Y-%m-%d")
    elif date_str == "모레":
        return (now + timedelta(days=2)).strftime("%Y-%m-%d")
    elif date_str == "어제":
        return (now - timedelta(days=1)).strftime("%Y-%m-%d")
    
    # "이번주/이번 주" + 요일 처리
    week_match = re.search(r"(이번\s*주|이번주|다음\s*주|다음주)", date_str)
    if week_match:
        week_keyword = week_match.group(1).replace(" ", "")
        is_next_week = "다음" in week_keyword
        
        # 요일 추출
        weekdays = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
        weekday_idx = None
        for i, wd in enumerate(weekdays):
            if wd in date_str:
                weekday_idx = i
                break
        
        if weekday_idx is not None:
            # 오늘 요일 기준으로 목표 요일까지의 날짜 계산
            current_weekday = now.weekday()  # 0=월요일, 6=일요일
            days_ahead = weekday_idx - current_weekday
            
            if is_next_week:
                days_ahead += 7
            elif days_ahead < 0:
                days_ahead += 7  # 이번 주가 지났으면 다음 주로
            
            target_date = now + timedelta(days=days_ahead)
            return target_date.strftime("%Y-%m-%d")
    
    # "N월 M일" 형식 처리
    month_day_match = re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", date_str)
    if month_day_match:
        month = int(month_day_match.group(1))
        day = int(month_day_match.group(2))
        year = now.year
        
        # 월이 현재보다 작으면 다음 해
        if month < now.month or (month == now.month and day < now.day):
            year += 1
        
        try:
            target_date = datetime(year, month, day)
            return target_date.strftime("%Y-%m-%d")
        except ValueError:
            pass
    
    # "N/M" 또는 "N-M" 형식 처리
    slash_match = re.search(r"(\d{1,2})[/-](\d{1,2})", date_str)
    if slash_match:
        month = int(slash_match.group(1))
        day = int(slash_match.group(2))
        year = now.year
        
        # 월이 현재보다 작으면 다음 해
        if month < now.month or (month == now.month and day < now.day):
            year += 1
        
        try:
            target_date = datetime(year, month, day)
            return target_date.strftime("%Y-%m-%d")
        except ValueError:
            pass
    
    # 변환 실패 시 현재 날짜 반환
    return now.strftime("%Y-%m-%d")

def _extract_date(text: str) -> str | None:
    """
    텍스트에서 날짜 표현(예: '11월 3일', '11/3', '11-3', '오늘', '내일')을 추출하고 '11월 3일' 형태로 정규화
    """
    import re as _re

    t = (text or "").strip()

    # 1) '11월 3일' 같은 일반 표현
    m1 = _re.search(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일", t)
    if m1:
        month, day = m1.groups()
        return f"{int(month)}월 {int(day)}일"

    # 2) '11/3', '11-3' 형식
    m2 = _re.search(r"(\d{1,2})[/-](\d{1,2})", t)
    if m2:
        month, day = m2.groups()
        return f"{int(month)}월 {int(day)}일"

    # 3) '11/3일', '11-3일' 혼합형도 수용
    m3 = _re.search(r"(\d{1,2})[/-](\d{1,2})\s*일", t)
    if m3:
        month, day = m3.groups()
        return f"{int(month)}월 {int(day)}일"

    if "어제" in t:
        return "어제"
    if "내일" in t:
        return "내일"
    if "오늘" in t:
        return "오늘"
    if "모레" in t:
        return "모레"
    return None

# 감정 단어 리스트 (공통 사용)
EMOTION_POSITIVE_WORDS = ["행복", "좋아", "기뻐", "즐거", "신나", "만족", "뿌듯", "기쁘", "웃음", "즐겁"]
EMOTION_NEGATIVE_WORDS = ["슬퍼", "우울", "힘들", "외로워", "속상해", "짜증", "화나", "답답해", "답답", "괴로워", "아픔", "상처", "실망"]
EMOTION_TIRED_WORDS = ["피곤", "졸려", "지쳐", "무기력", "나른", "졸음"]
EMOTION_ANXIOUS_WORDS = ["불안", "긴장", "걱정", "초조"]

def _extract_emotion_word_and_label(text: str) -> tuple[Optional[str], Optional[str]]:
    """감정 단어와 라벨 추출 (공통 함수)
    
    Args:
        text: 분석할 텍스트
    
    Returns:
        (emotion_word, label): 감정 단어와 라벨 튜플
        - emotion_word: 실제 감정 단어 (예: "속상해", "기뻐") 또는 None
        - label: 감정 라벨 (예: "긍정", "부정", "피로", "불안") 또는 None
    """
    
    # 모든 감정 키워드 리스트
    all_emotion_words = EMOTION_POSITIVE_WORDS + EMOTION_NEGATIVE_WORDS + EMOTION_TIRED_WORDS + EMOTION_ANXIOUS_WORDS
    
    # 실제 감정 단어 찾기 (부분 문자열 매칭)
    emotion_word = None
    for word in all_emotion_words:
        if word in text:
            emotion_word = word
            break
    
    # "속상해", "속상하다" 같은 변형도 체크
    if not emotion_word and "속상" in text:
        emotion_word = "속상해"
    
    # 라벨 설정
    label = None
    if any(k in text for k in EMOTION_POSITIVE_WORDS):
        label = "긍정"
    elif any(k in text for k in EMOTION_TIRED_WORDS):
        label = "피로"
    elif any(k in text for k in EMOTION_ANXIOUS_WORDS):
        label = "불안"
    elif any(k in text for k in EMOTION_NEGATIVE_WORDS):
        label = "부정"
    
    return emotion_word, label

def _summarize_emotion_context_for_save(user_text: str, llm=None) -> str:
    """감정 표현의 원인/상황을 간결하게 요약
    
    예시:
    - "내 남자친구가 연락을 안받아서 너무 속상해" → "남자친구의 연락 문제"
    - "시험에서 떨어져서 너무 슬퍼" → "시험 실패"
    - "오늘 회사에서 상사한테 혼나서 기분이 안좋아" → "직장 문제"
    """
    try:
        if llm:
            prompt = f"""다음 감정 표현에서 감정의 원인이나 상황을 간결하게 요약해주세요.

사용자 발화: "{user_text}"

요약 규칙:
- 감정 단어(속상해, 슬퍼, 기뻐 등)는 제외하고 원인/상황만 추출
- 3-10자 정도로 간결하게 요약
- 예시:
  * "내 남자친구가 연락을 안받아서 너무 속상해" → "남자친구의 연락 문제"
  * "시험에서 떨어져서 너무 슬퍼" → "시험 실패"
  * "오늘 회사에서 상사한테 혼나서 기분이 안좋아" → "직장 문제"
  * "친구가 생일 선물을 안줘서 서운해" → "생일 선물 문제"
  * "오늘 날씨가 좋아서 기분이 좋아" → "날씨 좋음"

요약만 출력하세요 (설명 없이):"""
            
            response = llm.invoke(prompt)
            summary = response.content.strip() if hasattr(response, 'content') else str(response).strip()
            
            # 응답이 너무 길면 제한
            if len(summary) > 30:
                summary = summary[:30]
            
            if summary and summary not in ["", "None", "null"]:
                return summary
    except Exception as e:
        logger.debug(f"감정 상황 요약 실패 (LLM): {e}")
    
    # LLM 실패 시 간단한 규칙 기반 추출
    import re
    # 감정 관련 키워드 제거
    text = user_text
    emotion_patterns = [
        r'\s*(속상해|슬퍼|기뻐|행복해|짜증나|화나|답답해|우울해|불안해|피곤해|기분.*?좋아|기분.*?나빠|너무|정말|진짜|아주)\s*',
        r'\s*(그래서|그래서는|해서|해서는|때문에|때문에는)\s*',
    ]
    for pattern in emotion_patterns:
        text = re.sub(pattern, '', text)
    
    # 간단한 요약 (앞 부분만, 30자 제한)
    text = text.strip()
    if len(text) > 30:
        text = text[:30]
    
    return text if text else user_text[:30]

def _extract_schedule_rule_based(text: str) -> dict:
    """
    규칙 기반 일정 추출 (날짜/시간/제목)
    - 날짜는 _extract_date로 통일 추출
    - 시간은 (오전|오후)? + 숫자 + 시
    - 제목은 날짜 표현 뒤쪽의 핵심 명사를 우선 추출
    """
    import re as _re

    date = _extract_date(text)
    time = None
    title = None

    # 시간 추출
    m_time = _re.search(r"(오전|오후)?\s*(\d{1,2})\s*시", text)
    if m_time:
        time = f"{m_time.group(1) or ''} {m_time.group(2)}시".strip()

    # 제목 추출: 날짜 표현 뒤 또는 '에/에는' 뒤 핵심어
    # 우선 사전 키워드로 빠르게 캡처
    keywords = ["미용실", "치과", "병원", "회의", "약속", "점심", "저녁", "수업", "인턴", "미팅"]
    for kw in keywords:
        if kw in text:
            title = kw
            break

    if not title:
        # 날짜가 있으면 날짜 다음 구간에서 후보 추출
        # 패턴 개선: "11월 3일에 병원 가기로 했어" → 제목=병원, 날짜=11월 3일
        if date:
            # 날짜 패턴을 이스케이프 처리하여 정확한 매칭
            date_pattern = _re.escape(date)
            # "날짜에 제목 가기로/예약" 패턴 우선 매칭
            pattern1 = rf"{date_pattern}\s*(?:에|에는)?\s*([가-힣A-Za-z0-9]+)\s*(?:가야|가기|예약|있어|하기로)"
            m_after = _re.search(pattern1, text)
            if m_after:
                cand = m_after.group(1)
                # 흔한 동사/조사 제거
                cand = _re.sub(r"(예약|가|가기|가기로|있|해야|하기로|함)(?:\s*했어|\s*했어요|\s*함)?", "", cand).strip()
                cand = cand.replace(".", "").replace(" ", "")
                if cand and len(cand) > 0:
                    title = cand
            else:
                # 대체 패턴: "날짜에 제목" 형식
                pattern2 = rf"{date_pattern}\s*(?:에|에는)?\s*([가-힣A-Za-z0-9\s]+)"
                m_after2 = _re.search(pattern2, text)
                if m_after2:
                    cand = m_after2.group(1)
                    # 흔한 동사/조사 제거
                    cand = _re.sub(r"(예약|가|가기|가기로|있|해야|하기로|함)(?:\s*했어|\s*했어요|\s*함)?", "", cand).strip()
                    cand = cand.replace(".", "").replace(" ", "")
                    if cand and len(cand) > 0:
                        title = cand

    # 마지막으로 여전히 없으면 간단한 힌트 기반 추론
    if not title:
        if any(k in text for k in ["머리", "자르", "염색"]):
            title = "미용실"
        elif "치과" in text or "충치" in text:
            title = "치과"
        elif "병원" in text or "진료" in text:
            title = "병원"
        elif "회의" in text or "미팅" in text:
            title = "회의"
        else:
            # 최소 기본값
            title = "일정"

    # 정규화
    if title:
        title = title.strip().replace(" ", "").replace(".", "")
    if date:
        date = date.replace(" ", "") if date not in ("오늘", "내일", "모레") else date

    if date or title or time:
        return {
            "user.일정": [{
                "제목": title or "",
                "날짜": date or "",
                "시간": time or "",
            }]
        }
    return {}

def _extract_robust(utter: str, llm=None):
    """
    1) 'LOC (POS)? 에 TARGET …'  (예: 화장실 선반 위에 치실…)
    2) 'LOC 안/속 에 TARGET …'    (예: 장바구니 안에 사과…)
    3) 앞절: 위치+대상, 뒷절: '그거/그것' 지시어
    """
    text = _normalize_utterance(utter)
    clauses = _split_clauses(text)

    target = location = position = None
    first = clauses[0] if clauses else text

    # --- 1) LLM 우선 추출 (강화된 프롬프트) ---
    if llm:
        try:
            # 강화된 LLM 프롬프트 - 모든 패턴을 이해할 수 있도록
            prompt = f"""다음 문장에서 물건명과 위치를 정확히 추출하세요.

문장: "{text}"

**추출 규칙:**
1. 물건명: 사용자가 말한 정확한 단어 그대로 추출
2. 위치: 구체적인 장소나 위치 정보 추출 (없으면 null)
3. "제자리"는 위치가 아님 (null로 처리)
4. "거", "것" 같은 일반적인 단어는 무시하고 구체적인 물건명 찾기
5. **중요**: "~에 가져다 놔" 패턴에서 "~에" 부분이 위치입니다

**다양한 패턴 예시:**
- "치실 가져와" → target: 치실, location: null
- "화장실 선반 위에 치실 있잖아" → target: 치실, location: 화장실 선반 위  
- "택배 온 거 문 앞에서 가져와" → target: 택배, location: 문 앞
- "냉장고 안에 넣어 우유 가져다줄래" → target: 우유, location: 냉장고 안
- "책상 위에 있는 펜 찾아줘" → target: 펜, location: 책상 위
- "현관에서 소포 가져와" → target: 소포, location: 현관
- "물컵 제자리에 가져다 놔" → target: 물컵, location: null
- "비타민 정수기 옆에 가져다 놔" → target: 비타민, location: 정수기 옆
- "실내화 현관 앞에 가져다 놔라" → target: 실내화, location: 현관 앞
- "내 노트북 충전기 찾아줘. 노트북 충전기는 내 방 책상에 있어." → target: 노트북 충전기, location: 내 방 책상

**출력 형식:**
target: [물건명 또는 null]
location: [위치 또는 null]"""

            resp = llm.invoke(prompt)
            ans = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
            
            # 빠른 파싱
            target = None
            location = None
            
            if 'target:' in ans:
                target_line = [line for line in ans.split('\n') if 'target:' in line][0]
                target = target_line.split('target:')[1].strip()
                if target.lower() in ['null', 'none', '']:
                    target = None
            
            if 'location:' in ans:
                location_line = [line for line in ans.split('\n') if 'location:' in line][0]
                location = location_line.split('location:')[1].strip()
                if location.lower() in ['null', 'none', '']:
                    location = None
            
            if target or location:
                print(f"[DEBUG] LLM 추출: target={target}, location={location}")
                return target, location
                
        except Exception as e:
            print(f"[WARN] LLM 추출 실패: {e}")
    
    # --- LLM만 사용 - 규칙 기반 fallback 제거 ---
    if not target and not location:
        # LLM이 실패한 경우 기본값 반환
        print(f"[WARN] LLM 추출 실패, 기본값 반환")
        return None, None
    
    return target, location

# 상수 정의
ERROR_UNSUPPORTED = "죄송해요, 해당 기능은 아직 지원하지 않아요. 다른 작업을 요청해주시겠어요?"

def _has_batchim(w):
    """받침 여부 확인"""
    try:
        c = ord(w[-1]) - 0xAC00
        return 0 <= c <= 11171 and (c % 28) != 0
    except Exception:
        return False

def josa(topic: str, particle_pair=("은","는")):
    """
    한글 조사 자동 선택 함수 (개선)
    - 받침이 있으면 첫 번째 조사 (은, 이, 을)
    - 받침이 없으면 두 번째 조사 (는, 가, 를)
    """
    if not topic: 
        return particle_pair[1]
    return particle_pair[0] if _has_batchim(topic) else particle_pair[1]

EXPORT_DIR = os.path.expanduser("~/.life_assist_dm/exports")

# append_cognitive_log 함수 제거됨 - 사용되지 않음

# ---------- 정서: 간단 스몰토크 ----------

def build_emotional_reply(text: str, llm=None, user_name_confirmed=False) -> str:
    """
    감정적 대화 응답 생성
    - 날짜/시간/날씨/간단 인사 → 룰 기반 처리
    - 그 외 감정적 문장 → LLM에게 위임
    """
    t = (text or "").strip()

    # ✅ 규칙 기반 (빠른 처리)
    if re.search(r"(오늘|현재).*(날짜|며칠|몇\s*일)", t):
        return f"오늘은 {datetime.now().strftime('%Y-%m-%d')}입니다."
    if re.search(r"(지금|현재).*(시간|몇\s*시)", t):
        return f"지금 시간은 {datetime.now().strftime('%H:%M')}입니다."
    if re.search(r"날씨", t):
        return "날씨 정보를 불러오는 것에 실패했어요. 대신 겉옷이나 우산이 필요할 것 같으면 챙겨가는 걸 추천할게요."
    # 간단 인사/감사는 하드코딩 제거 → LLM 우선 처리

    # ✅ 나머지 감정 표현은 LLM에게 넘김
    if llm:
        prompt = (
            "당신은 사용자의 감정을 깊이 이해하고 공감하는 생활 지원 로봇입니다.\n"
            "사용자의 감정 상태를 파악하고, 그 감정에 맞는 따뜻하고 진심 어린 응답을 해주세요.\n"
            "조언보다는 먼저 공감하고, 사용자가 혼자가 아니라는 것을 느끼도록 해주세요.\n"
            "답변은 1-2문장으로 간결하게, **항상 존댓말로** 응답해주세요.\n\n"
            "**절대 금지사항:**\n"
            "- \"이 정보를 기록하겠습니다\" 같은 시스템 작업을 언급하지 마세요\n"
            "- \"기록된 정보는 다음과 같습니다\" 같은 목록을 만들지 마세요\n"
            "- \"추가로 기록할 사항이나 질문이 있으시면\" 같은 제안을 하지 마세요\n"
            "- 시스템 내부 작업이나 저장 과정을 설명하지 마세요\n"
            "- 단순히 감정에 공감하고 응답하세요\n\n"
            f"사용자: {t}\n"
            "로봇:"
        )
        try:
            response = llm.invoke(prompt)
            # GPT 객체 전체를 찍지 않고, content만 추출
            if hasattr(response, "content"):
                return response.content.strip()
            elif isinstance(response, str):
                return response.strip()
            else:
                return str(response).strip()
        except Exception as e:
            print(f"[WARN] 감정 대화 LLM 호출 실패: {e}")
            # LLM 호출 실패 시 fallback으로 넘어감
            pass

    # ✅ LLM이 실패한 경우 감정별 fallback
    positive = ["좋아", "기뻐", "행복", "신나", "즐거워", "만족", "뿌듯", "기쁘", "웃음", "즐겁"]
    negative = ["슬퍼", "우울", "힘들", "외로워", "속상해", "짜증", "화나", "답답해", "괴로워", "아픔", "상처", "실망"]
    tired    = ["피곤", "졸려", "지쳐", "휴식", "쉬고", "힘빠져", "에너지 없어", "무기력", "나른", "졸음"]

    fallback_map = {
        "positive": [
            "정말 좋은 기분이시군요! 저도 함께 기뻐요.",
            "좋은 일이 있으신 것 같아 저도 기분이 좋아요 🙂",
            "즐거우신 모습이 보기 좋아요!",
            "기쁜 마음이 전해져요. 저도 행복해요!",
            "정말 뿌듯하시겠어요. 축하해요!"
        ],
        "negative": [
            "지금 마음이 많이 힘드실 것 같아요. 제가 옆에 있을게요.",
            "그런 기분이 드시는 게 당연해요. 함께 이겨내봐요.",
            "마음이 무거우시군요. 제가 곁에 있어요.",
            "정말 속상하셨겠어요. 저도 마음이 아파요.",
            "외로우셨겠어요. 혼자가 아니에요, 저도 있어요."
        ],
        "tired": [
            "많이 지치셨군요. 잠깐이라도 쉬세요.",
            "몸과 마음을 잘 챙기셔야 해요.",
            "피곤할 땐 휴식이 최고예요. 같이 잠시 쉬어볼까요?",
            "에너지가 떨어지신 것 같아요. 충전이 필요해요.",
            "무기력하신 기분이 드네요. 천천히 쉬어가세요."
        ],
        "neutral": [
            "네, 듣고 있어요. 계속 말씀해주세요.",
            "그럴 수 있어요. 저도 함께 있어요.",
            "네, 이해해요. 더 이야기해주세요.",
            "말씀해주셔서 고마워요. 계속 들어드릴게요.",
            "네, 알겠어요. 언제든지 말씀해주세요."
        ]
    }

    # 키워드 기반 감정 분류
    category = "neutral"
    if any(k in t for k in positive):
        category = "positive"
    elif any(k in t for k in negative):
        category = "negative"
    elif any(k in t for k in tired):
        category = "tired"

    return random.choice(fallback_map[category])


# ---------- 물리: 로봇 명령(영문) 생성 ----------
# OBJ_MAP과 LOC_MAP은 위의 TARGET_MAP과 LOCATION_MAP으로 통합됨

def to_task_command_en(action: str, target: str, location: str = None, memory_instance=None) -> dict:
    """
    액션/대상/위치를 영어 명령(JSON)으로 변환
    """
    # 기본 매핑
    action_map = {
        "find": "find",
        "deliver": "deliver",
        "organize": "organize"   # ✅ 정리하기를 명확히 분리
    }

    # 안전하게 액션 확인
    if action not in action_map:
        return {
            "action": "unsupported",
            "original": f"Unsupported action for {target} {location or ''}".strip()
        }

    action_en = action_map[action]
    
    # 영문화/매핑 적용 (이미 영문이면 그대로 사용)
    if target:
        if target in TARGET_MAP:                      # 한글 키 매핑
            target_en = TARGET_MAP[target]
        elif target in TARGET_MAP.values():           # 이미 영문 값
            target_en = target
        elif memory_instance:                      # LLM fallback
            target_en = _translate_to_english(target, memory_instance)
        else:
            target_en = target  # ✅ fallback: 원문 그대로 사용
    else:
        target_en = "unknown"

    # location이 dict인 경우 (장소와 세부위치 분리된 경우)
    if isinstance(location, dict):
        place = location.get("장소", "")
        sub_location = location.get("세부위치", "")
        # 장소와 세부위치를 각각 번역
        place_en = None
        sub_en = None
        if place:
            if place in LOCATION_MAP:
                place_en = LOCATION_MAP[place]
            elif place in LOCATION_MAP.values():
                place_en = place
            elif memory_instance:
                place_en = _translate_to_english(place, memory_instance)
            else:
                place_en = place
        if sub_location:
            if sub_location in LOCATION_MAP:
                sub_en = LOCATION_MAP[sub_location]
            elif sub_location in LOCATION_MAP.values():
                sub_en = sub_location
            elif memory_instance:
                sub_en = _translate_to_english(sub_location, memory_instance)
            else:
                sub_en = sub_location
        # 조합: 자연스러운 영어 위치 표현
        if place_en and sub_en:
            # "안에"/"inside" 같은 경우 "in" 중복 방지
            if sub_en.lower() in ["inside", "in", "within"]:
                # "inside room" 형태로 조합 (중복 "in" 제거)
                loc_en = f"{sub_en} {place_en}"
            elif sub_en.lower() in ["on", "above", "on top of"]:
                # "on desk" 형태
                loc_en = f"{sub_en} {place_en}"
            elif sub_en.lower() in ["under", "below", "beneath"]:
                # "under table" 형태
                loc_en = f"{sub_en} {place_en}"
            elif sub_en.lower() in ["beside", "next to", "by"]:
                # "beside desk" 형태
                loc_en = f"{sub_en} {place_en}"
            else:
                # 기타 경우: "drawer in bathroom" 형태
                loc_en = f"{sub_en} in {place_en}"
        elif place_en:
            loc_en = place_en
        elif sub_en:
            loc_en = sub_en
        else:
            loc_en = None
    elif location:
        if location in LOCATION_MAP:                    # 한글 키 매핑
            loc_en = LOCATION_MAP[location]
        elif location in LOCATION_MAP.values():         # 이미 영문 값
            loc_en = location
        elif memory_instance:                      # LLM fallback
            loc_en = _translate_to_english(location, memory_instance)
        else:
            loc_en = location  # ✅ fallback: 원문 그대로 사용
    else:
        loc_en = None

    if action_en == "find":
        cmd = {
            "action": "find",
            "target": target_en,
            "original": f"Please find {target_en}"
        }
    elif action_en == "deliver":
        cmd = {
            "action": "deliver",
            "target": target_en,
            "location": loc_en,
            "original": f"Please deliver {target_en}" + (f" from {loc_en}" if loc_en else "")
        }
    elif action_en == "organize":
        cmd = {
            "action": "organize",
            "target": target_en,
            "location": loc_en,
            "original": f"Please organize {target_en}" + (f" to {loc_en}" if loc_en else "")
        }
    else:
        cmd = {
            "action": "unsupported",
            "original": f"Unsupported action for {target}"
        }

    return cmd


def handle_physical_task(user_input: str, memory_instance, session_id: str, entity_already_saved: bool = False) -> dict:
    """물리적 작업 처리 (찾기, 가져오기, 정리하기)
    
    Args:
        user_input: 사용자 입력
        memory_instance: 메모리 인스턴스
        session_id: 세션 ID
        entity_already_saved: 엔티티가 이미 저장되었는지 여부 (중복 저장 방지)
    """
    try:
        import re
        
        # 0. 세션 상태 준비 및 과도한 pending 초기화 방지
        if not hasattr(memory_instance, "session_state"):
            memory_instance.session_state = {}
        state = memory_instance.session_state.setdefault(session_id, {
            "last_action": None,
            "last_target": None,
            "last_location": None,
            "last_question": None,
        })
        # 이전에는 새 명령 감지 시 바로 pending_question을 삭제했으나,
        # 사용자가 곧바로 위치 등 슬롯을 답변할 수 있도록 유지한다.
        
        # ✅ 해석 전처리
        text = _preprocess_for_parsing(user_input)
        original_text = text
        
        # 0-1. 지시어 1차 치환: 그거/그것/거기 → 직전 엔티티로 보완
        try:
            if state.get("last_target"):
                text = re.sub(r"(그거|그것)", state["last_target"], text)
            if state.get("last_location"):
                text = re.sub(r"(거기)", state["last_location"], text)
        except Exception:
            pass
        
        # 1. 액션 타입 추정 (LLM)
        action = _extract_action_type(text, memory_instance.llm if hasattr(memory_instance, 'llm') else None)
        logger.debug(f"[PHYSICAL] Input={text}, action={action}")
        
        if action == "unsupported":
            return {"success": False, "message": ERROR_UNSUPPORTED, "robot_command": None}
        
        # 2. 물건명과 위치 추출 (LLM)
        target, location = _extract_robust(text, memory_instance.llm if hasattr(memory_instance, 'llm') else None)
        
        # 추출 실패 시 지시어 기반 보강
        if not target and re.search(r"(그거|그것)", original_text) and state.get("last_target"):
            target = state["last_target"]
        if not location and re.search(r"(거기)", original_text) and state.get("last_location"):
            location = state["last_location"]
            
        logger.debug(f"[PHYSICAL] Extracted - target={target}, location={location}")
        
        if not target:
            logger.warning(f"[PHYSICAL] No target extracted from: {text}")
            return {"success": False, "message": "죄송해요, 어떤 물건을 말씀하시는지 모르겠어요.", "robot_command": None}
        
        # ✅ 사용자가 말한 '명시적 위치'가 있으면 무조건 우선
        explicit_location = location is not None
        
        # ✅ 개선된 물리적 지원 로직
        if action == "find":
            # 물건 찾기: 명시적 위치 우선, 없으면 저장된 위치 확인
            if explicit_location:
                # 명시적 위치가 있으면 바로 재질문
                msg = f"찾고 계신 {target}는 {location}에 있어요. 가져다 드릴까요?"
                memory_instance.pending_question[session_id] = {
                    "type": "location_confirmed",
                    "item_name": target,
                    "location": location,
                    "action": "deliver",
                    "question": msg
                }
                memory_instance.current_question[session_id] = msg
                return {"success": True, "message": msg, "robot_command": None}
            else:
                saved_location = _find_saved_location(memory_instance, session_id, target)
                
                if saved_location:
                    # a1. 위치가 저장되어 있음 - 재질문 후 사용자 대답 대기
                    msg = f"찾고 계신 {target}는 {saved_location}에 있어요. 가져다 드릴까요?"
                    memory_instance.pending_question[session_id] = {
                        "type": "location_confirmed",
                        "item_name": target,
                        "location": saved_location,
                        "action": "deliver",
                        "question": msg
                    }
                    memory_instance.current_question[session_id] = msg
                    return {"success": True, "message": msg, "robot_command": None}
                else:
                    # b1. 위치가 저장되어 있지 않음
                    msg = f"{target}의 위치는 알고 있지 않아요. 알려주시면 기억해둘게요."
                    memory_instance.pending_question[session_id] = {
                        "type": "location_unknown",
                        "item_name": target,
                        "action": "deliver",
                        "question": msg
                    }
                    memory_instance.current_question[session_id] = msg
                    return {"success": True, "message": msg, "robot_command": None}
        
        elif action == "deliver":
            # 물건 가져다주기: 위치 확인 후 처리
            logger.debug(f"[PHYSICAL DELIVER] target={target}, location={location}, explicit_location={explicit_location}")
            
            # target이 없으면 지시어 처리
            if not target:
                if re.search(r"(그것|그거|그것|거)", text):
                    target = state.get("last_target")
                else:
                    # 문장에서 물건명 직접 추출
                    target_patterns = [
                        r"내\s*(\w+)\s*는",  # "내 틴트는"
                        r"(\w+)\s*(?:은|는)\s*(?:.*?)\s*있어",  # "티스푼은 책상 위에 있어"
                        r"(\w+)\s*(?:을|를)\s*가져",  # "티스푼을 가져"
                        r"(\w+)\s*침대\s*위에",  # "핸드폰 침대 위에"
                        r"(\w+)\s*책상\s*위에",  # "핸드폰 책상 위에"
                    ]
                    for pattern in target_patterns:
                        match = re.search(pattern, original_text)
                        if match:
                            target = match.group(1)
                            break
            
            if explicit_location and target and location:
                # '~에 있는 ~ 가지고 와' 패턴: 바로 deliver
                
                # ✅ Step 2: 물건 위치 엔티티 추출 및 엑셀 저장 (병행 처리)
                # 단, cognitive 단계에서 이미 저장하지 않은 경우에만 저장
                if not entity_already_saved:
                    user_name = memory_instance.user_names.get(session_id or "default_session", "사용자")
                    if user_name and user_name != "사용자":
                        try:
                            if hasattr(memory_instance, '_rule_based_extract'):
                                # 물건 위치 엔티티 추출 (rule-based)
                                rule_entities = memory_instance._rule_based_extract(user_input, session_id)
                                if rule_entities.get("user.물건"):
                                    logger.debug(f"[PHYSICAL+COGNITIVE] 위치 엔티티 감지됨 → Excel 저장: {rule_entities['user.물건']}")
                                    for item_entity in rule_entities["user.물건"]:
                                        # 엑셀에 엔티티 저장
                                        memory_instance.excel_manager.save_entity_data(
                                            user_name=user_name,
                                            entity_type="user.물건",
                                            data={
                                                "이름": item_entity.get("이름", target),
                                                "위치": item_entity.get("위치", location),
                                                "장소": item_entity.get("장소", ""),
                                                "세부위치": item_entity.get("세부위치", ""),
                                                "추출방법": item_entity.get("추출방법", "rule-based")
                                            }
                                        )
                                    logger.debug(f"[PHYSICAL+COGNITIVE] 물건 위치 엔티티 저장 완료: {user_name}")
                                else:
                                    # rule-based 추출 실패 시, LLM 추출 결과를 직접 저장
                                    logger.debug(f"[PHYSICAL+COGNITIVE] rule-based 추출 없음, LLM 결과 직접 저장")
                                    memory_instance.excel_manager.save_entity_data(
                                        user_name=user_name,
                                        entity_type="user.물건",
                                        data={
                                            "이름": target,
                                            "위치": location,
                                            "장소": "",
                                            "세부위치": "",
                                            "추출방법": "llm"
                                        }
                                    )
                        except Exception as e:
                            logger.warning(f"[PHYSICAL] 위치 엔티티 병행 추출 실패: {e}")
                            import traceback
                            logger.debug(traceback.format_exc())
                else:
                    logger.debug(f"[PHYSICAL+COGNITIVE] 엔티티가 이미 저장되었으므로 중복 저장 건너뜀")
                
                # Step 3: 로봇 명령 생성
                msg = f"'{target}'를 '{location}'에서 가져오겠습니다."
                robot_cmd = to_task_command_en("deliver", target, location, memory_instance)
                logger.debug(f"[PHYSICAL RESULT] message={msg}, robot_command={robot_cmd}")
                return {"success": True, "message": msg, "robot_command": robot_cmd}
            else:
                # 위치를 모름 - 저장된 위치 확인
                saved_location = None
                try:
                    if hasattr(memory_instance, 'get_location'):
                        # dict 형태로 조회 (장소와 세부위치 분리)
                        saved_location = memory_instance.get_location(target, return_dict=True)
                        if not saved_location:
                            # dict 조회 실패 시 문자열로 재시도
                            saved_location = memory_instance.get_location(target, return_dict=False)
                except Exception:
                    pass
                
                if saved_location:
                    # 저장된 위치가 있음 - 바로 deliver
                    # saved_location이 dict인 경우와 문자열인 경우 모두 처리
                    if isinstance(saved_location, dict):
                        place = str(saved_location.get('장소', '') or '').strip()
                        sub_location = str(saved_location.get('세부위치', '') or '').strip()
                        # nan, None, 빈 문자열 필터링
                        if place.lower() in ['nan', 'none', '']:
                            place = ''
                        if sub_location.lower() in ['nan', 'none', '']:
                            sub_location = ''
                        # 둘 다 있으면 공백으로 연결, 하나만 있으면 그것만
                        if place and sub_location:
                            location_msg = f"{place} {sub_location}"
                        elif place:
                            location_msg = place
                        elif sub_location:
                            location_msg = sub_location
                        else:
                            location_msg = str(saved_location) if saved_location else ""
                    else:
                        location_msg = str(saved_location).strip()
                        # nan 체크
                        if location_msg.lower() in ['nan', 'none']:
                            location_msg = ""
                    msg = f"{target}을(를) {location_msg}에서 가져오겠습니다."
                    robot_cmd = to_task_command_en("deliver", target, saved_location, memory_instance)
                    return {"success": True, "message": msg, "robot_command": robot_cmd}
                else:
                    # 위치를 모름 - 재질문
                    msg = f"{target}의 위치는 알고 있지 않아요. 알려주시면 기억해둘게요."
                    memory_instance.pending_question[session_id] = {
                        "type": "location_unknown",
                        "item_name": target,
                        "action": "deliver",
                        "question": msg
                    }
                    memory_instance.current_question[session_id] = msg
                    return {"success": True, "message": msg, "robot_command": None}
        
        elif action == "organize":
            # 물건 정리하기: 명시적 위치 우선, 없으면 저장된 위치 확인
            if explicit_location:
                # 명시적 위치가 있으면 바로 그 위치로 정리
                msg = f"{target}을(를) {location}에 정리해두겠습니다."
                robot_cmd = to_task_command_en("organize", target, location, memory_instance)
                return {"success": True, "message": msg, "robot_command": robot_cmd}
            else:
                saved_location = _find_saved_location(memory_instance, session_id, target)
                
                if saved_location:
                    # 저장된 위치를 알면 그 곳으로 정리
                    msg = f"{target}을(를) {saved_location}에 정리해두겠습니다."
                    robot_cmd = to_task_command_en("organize", target, saved_location, memory_instance)
                    return {"success": True, "message": msg, "robot_command": robot_cmd}
                else:
                    # 위치를 모름 - 재질문
                    msg = f"{target}의 위치는 알고 있지 않아요. 알려주시면 기억해둘게요."
                    memory_instance.pending_question[session_id] = {
                        "type": "location_unknown",
                        "item_name": target,
                        "action": "organize",
                        "question": msg
                    }
                    memory_instance.current_question[session_id] = msg
                    return {"success": True, "message": msg, "robot_command": None}
        
        # 만약 여기까지 왔다면 처리되지 않은 action이므로 에러 반환
        logger.warning(f"[PHYSICAL] Unhandled action: {action}")
        return {"success": False, "message": ERROR_UNSUPPORTED, "robot_command": None}
            
    except Exception as e:
        logger.exception("physical_task_failed: %s\n%s", user_input, traceback.format_exc())
        # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
        return {
            "success": False,
            "message": "명령 처리 중 오류가 발생했어요. 다시 말씀해 주시겠어요?",
            "robot_command": None
        }


def handle_pending_answer(user_input: str, memory_instance, session_id: str) -> dict:
    """물리적 작업 재질문에 대한 답변 처리"""
    try:
        # pending_action은 더 이상 사용하지 않음 - pending_question으로 통합됨
        
        # 2. 기존 pending_question 처리
        if session_id not in memory_instance.pending_question:
            return {
                "success": False,
                "message": "대기 중인 질문이 없습니다.",
                "robot_command": None
            }
            
        question_data = memory_instance.pending_question[session_id]
        question_type = question_data.get("type", "")
        
        if question_type == "organize_meaning_clarification":
            # 정리 의미 구분 재질문 응답 처리
            original_text = question_data.get("original_text", "")
            
            # 청소 의미 응답 확인
            cleaning_keywords = ["청소", "닦", "먼지", "때", "깨끗", "쓸", "빨아", "세척", "소독", "살균", "청소하", "청소해", "청소해줘"]
            is_cleaning = any(keyword in user_input for keyword in cleaning_keywords)
            
            if is_cleaning:
                # 청소 의미로 확인됨 - 미지원 처리
                memory_instance.pending_question.pop(session_id, None)
                if session_id in memory_instance.current_question:
                    del memory_instance.current_question[session_id]
                return {"success": False, 
                        "message": "청소 작업은 아직 지원하지 않아요. 다른 작업을 요청해주시겠어요?",
                        "robot_command": None}
            
            # 제자리에 두기 의미로 확인됨 - organize 액션으로 처리
            memory_instance.pending_question.pop(session_id, None)
            if session_id in memory_instance.current_question:
                del memory_instance.current_question[session_id]
            
            # 원래 텍스트에서 물건과 위치 추출하여 organize 처리
            target, location = _extract_robust(original_text, memory_instance.llm if hasattr(memory_instance, 'llm') else None)
            
            if not target:
                target = "물건"  # 기본값
            
            saved_location = _find_saved_location(memory_instance, session_id, target)
            
            if saved_location:
                # 저장된 위치를 알면 그 곳으로 정리
                msg = f"{target}을(를) {saved_location}에 정리해두겠습니다."
                robot_cmd = to_task_command_en("organize", target, saved_location, memory_instance)
                return {"success": True, "message": msg, "robot_command": robot_cmd}
            else:
                # 위치를 모름 - 재질문
                msg = f"{target}의 위치는 알고 있지 않아요. 알려주시면 기억해둘게요."
                memory_instance.pending_question[session_id] = {
                    "type": "location_unknown",
                    "item_name": target,
                    "action": "organize",
                    "question": msg
                }
                memory_instance.current_question[session_id] = msg
                return {"success": True, "message": msg, "robot_command": None}
            
        elif question_type == "location_unknown":
            # 사용자가 위치를 알려줬을 때는 항상 저장/갱신(덮어쓰기 질문 없음)
            item = question_data.get("item_name")
            loc  = _extract_location_from_input(user_input) or user_input.strip()
            if not item or not loc:
                return {"success": False, "message": "위치를 잘 이해하지 못했어요. 다시 한 번만 알려주세요.", "robot_command": None}
            memory_instance.save_location(item, loc, overwrite=True)
            action = question_data.get("action")
            if action == "deliver":
                en_loc = _to_en_location(loc)
                cmd = to_task_command_en("deliver", item, en_loc or loc, memory_instance)
                memory_instance.pending_question.pop(session_id, None)
                return {"success": True, "message": f"{loc}의 {item}을 가져오겠습니다.", "robot_command": cmd}
            if action == "organize":
                en_loc = _to_en_location(loc)
                cmd = to_task_command_en("organize", item, en_loc or loc, memory_instance)
                memory_instance.pending_question.pop(session_id, None)
                return {"success": True, "message": f"{item}을(를) {loc}에 정리해둘게요.", "robot_command": cmd}
            memory_instance.pending_question.pop(session_id, None)
            return {"success": True, "message": f"'{item}'의 위치를 '{loc}'(으)로 저장했어요.", "robot_command": None}
        elif question_type == "location_confirmed":
            # 위치 확인 후 작업 실행 여부 결정
            res = _handle_task_execution_response(user_input, question_data, memory_instance, session_id)
            return res if isinstance(res, dict) else {"success": True, "message": res, "robot_command": None}
        else:
            # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
            return {
                "success": False,
                "message": "알 수 없는 질문 유형이에요. 다시 말씀해 주시겠어요?",
                "robot_command": None
            }
            
    except Exception as e:
        import traceback
        print(f"[ERROR] handle_pending_answer 실패: {traceback.format_exc()}")
        # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
        return {
            "success": False,
            "message": "죄송해요, 답변 처리 중 오류가 발생했어요. 다시 말씀해 주시겠어요?",
            "robot_command": None
        }


# handle_user_confirmation 함수 제거됨 - _handle_task_execution_response로 통합


# _handle_location_confirmation 함수 제거됨 - pending_question으로 통합


def _generate_physical_response(action: str, target: str, location: Optional[str], user_input: str, memory_instance=None) -> str:
    """
    사용자에게 보여줄 응답을 생성한다.
    - 로봇 명령(JSON)은 영어로 유지하되, 사용자에게는 한국어 자연스럽게 출력
    """
    # 영어 물건명을 한국어로 변환
    korean_target = _translate_to_korean(target)
    
    # ✅ 쓰레기 전용 slot-filling
    if korean_target in ["쓰레기", "휴지", "trash", "garbage"]:
        if location:
            return f"{korean_target}를 {location}에 버릴까요?"
        else:
            return "쓰레기를 어디에 버릴까요? (예: 쓰레기통)"
    
    if action == "find":
        if not target:
            return "죄송해요, 어떤 물건을 말씀하시는지 모르겠어요."
        if location:
            return f"{korean_target}{josa(korean_target, ('은','는'))} {location}에 있어요. 찾아드릴까요?"
        else:
            return f"{korean_target}{josa(korean_target, ('의','의'))} 위치는 알고 있지 않아요. 알려주시면 기억해둘게요."

    if action == "deliver":
        if location:
            return f"{korean_target}{josa(korean_target, ('은','는'))} {location}{josa(location,('에','에'))} 있어요. 가져다드릴까요?"
        else:
            return f"{korean_target}의 위치는 아직 몰라요. 알려주시면 기억해둘게요."

    if action == "organize":
        if not target:
            return "죄송해요, 어떤 물건을 말씀하시는지 모르겠어요."
        if location and location != "제자리":
            return f"{korean_target}{josa(korean_target, ('은','는'))} {location}{josa(location,('에','에'))} 있어요. 제자리에 가져다둘까요?"
        else:
            # 제자리 정리 요청 - 저장된 위치 확인
            saved_loc = _find_saved_location(memory_instance, None if not memory_instance else getattr(memory_instance, 'current_session_id', ''), target)
            
            if saved_loc:
                return f"{korean_target}의 제자리는 {saved_loc}에 있어요. 그곳에 가져다둘까요?"
            else:
                return f"{korean_target}의 제자리 위치는 기억하지 못해요. 알려주시면 그곳에 가져다둘게요."

    return ERROR_UNSUPPORTED


def _translate_to_korean(english_word: str) -> str:
    """
    영어 물건명을 한국어로 변환
    """
    korean_map = {
        "cup": "컵",
        "book": "책",
        "phone": "핸드폰",
        "keys": "열쇠",
        "wallet": "지갑",
        "glasses": "안경",
        "water": "물",
        "remote": "리모컨",
        "document": "서류",
        "door": "문",
        "hair_tie": "머리끈",
        "tissue": "휴지",
        "towel": "수건",
        "pen": "펜",
        "cane": "지팡이",
        "apple": "사과",
        "fruit": "과일",
        "drink": "음료수",
        "juice": "주스",
        "milk": "우유",
        "bread": "빵",
        "snack": "과자",
        "food": "음식",
        "bag": "가방",
        "handbag": "핸드백",
        "toy": "장난감",
        "doll": "인형",
        "ball": "공",
        "shoes": "신발",
        "socks": "양말",
        "clothes": "옷",
        "shirt": "셔츠",
        "pants": "바지",
        "skirt": "치마",
        "hat": "모자",
        "gloves": "장갑",
        "scarf": "스카프",
        "trash": "쓰레기",
        "garbage": "쓰레기",
        "waste": "쓰레기",
        "item": "물건",
        "magazine": "잡지",
        "newspaper": "신문"
    }
    return korean_map.get(english_word, english_word)




# _extract_target_and_location 함수 제거 - _extract_robust로 통합


# _preprocess_input 함수 제거됨 - _preprocess_for_parsing으로 통합

def _extract_action_type(user_input: str, llm=None) -> str:
    """
    LLM 우선 액션 추론:
      1) LLM 직접 분류 → 2) 간단한 Rule 가드 → 3) Embedding fallback
    """
    import re

    text = _preprocess_for_parsing(user_input)
    if not text:
        return "unsupported"

    # --- 1) LLM 우선 분류 ---
    if llm:
        try:
            prompt = f"""다음 한국어 명령을 분석해서 액션 타입을 분류해주세요.

명령: "{text}"

가능한 액션 타입:
- find: 물건을 찾아달라는 요청 (찾아줘, 어디있어, 위치 알려줘 등)
- deliver: 물건을 가져다달라는 요청 (가져와, 갖다줘, 가지고 와, 꺼내와 등)  
- organize: 물건을 정리해달라는 요청 (정리해, 정돈해, 제자리에 놔, 치워, 가져다 놔, 놔둬 등)
- clean: 청소를 해달라는 요청 (청소해, 닦아줘, 깨끗하게 해 등)
- unsupported: 지원하지 않는 요청 (스마트홈 제어, 복잡한 작업 등)

**중요 구분:**
- "가져와" = deliver (물건을 나에게 가져오기)
- "가져다 놔" = organize (물건을 특정 위치에 정리하기)

예시:
- "비타민 가져와" → deliver
- "리모컨 어디있어" → find  
- "책상 정리해줘" → organize
- "비타민 정수기 옆에 가져다 놔" → organize
- "실내화 현관 앞에 가져다 놔라" → organize
- "방 청소해줘" → clean
- "TV 켜줘" → unsupported

답변은 반드시 다음 중 하나의 단어만 출력하세요: find, deliver, organize, clean, unsupported"""

            resp = llm.invoke(prompt)
            ans = resp.content.strip().lower() if hasattr(resp, 'content') else str(resp).strip().lower()
            
            # 답변 정제 (불필요한 텍스트 제거)
            for action in ["find", "deliver", "organize", "clean", "unsupported"]:
                if action in ans:
                    return action
                    
        except Exception as e:
            print(f"[WARN] LLM 분류 실패: {e}")

    # --- LLM만 사용 - 규칙 기반 fallback 제거 ---
    # LLM이 실패한 경우 기본값 반환
    return "unsupported"


# ---------- [NEW] 간단 영어 변환 fallback ----------
def _to_english(word: str | None) -> str | None:
    """간단한 영어 변환 fallback"""
    if not word:
        return None
    # 사전이 있으면 우선 사용, 없으면 원문 유지
    try:
        # 통합 맵 우선, 없으면 간단 정규화(소문자+언더스코어)
        return TARGET_MAP.get(word, LOCATION_MAP.get(word, word.replace(" ", "_").lower()))
    except Exception:
        return word


# _extract_item_name 함수 제거 - _extract_robust로 통합


def _extract_location_from_input(user_input: str) -> str:
    """
    위치 표현을 최대한 포착:
    - "~에 있는", "~에서", "~(으)로" 패턴
    - 사전 LOC_MAP 키(한글) 최장일치
    """
    text = (user_input or "").strip()
    if not text:
        return None

    # ✅ 0) 명시적 위치 표현 우선 처리 (침대 옆, 소파 밑 등)
    explicit_patterns = [
        r"(침대\s*옆|침대\s*위|침대\s*밑|침대\s*머리맡)",
        r"(소파\s*밑|소파\s*위|소파\s*옆)",
        r"(책상\s*위|책상\s*밑|책상\s*옆)",
        r"(식탁\s*위|식탁\s*밑)",
        r"(테이블\s*위|테이블\s*밑)",
        r"(바닥\s*에|바닥\s*에서)",
        r"(현관\s*쪽|현관\s*앞|현관\s*에)",
        r"(베란다\s*에|베란다\s*에서)",
        # 신규 보강: 복합 위치 표현
        r"(식탁\s*(밑|위))",
        r"(책꽂이\s*(위|맨\s*위|맨\s*아래|칸))",
        r"(문\s*앞|문\s*옆)",
        r"(바닥\s*(에|위))",
        r"(세탁기\s*(위|옆|안))",
        r"(에어컨\s*밑)",
        r"(장바구니\s*안|장바구니\s*속)",
        r"(정수기\s*옆|정수기\s*위|정수기\s*밑)",
        r"(냉장고\s*옆|냉장고\s*위|냉장고\s*밑)",
    ]
    
    for pattern in explicit_patterns:
        m = re.search(pattern, text)
        if m:
            return m.group(1).strip()

    # 1) "~에 있는" 패턴
    m = re.search(r"([가-힣A-Za-z0-9\s]+?)\s*에\s*있는", text)
    if m:
        cand = m.group(1).strip()
        # 최장일치로 세분화 (예: "부엌 식탁")
        tokens = [t.strip() for t in re.split(r"\s+", cand) if t.strip()]
        if tokens:
            # 마지막 토큰이 위치 표지어면 바로 앞 토큰과 결합
            if tokens[-1] in {"위","옆","밑"} and len(tokens) > 1:
                return f"{tokens[-2]} {tokens[-1]}"
            # 사전 키 우선
            for k in reversed([" ".join(tokens[:i]) for i in range(len(tokens),0,-1)]):
                if k in LOCATION_MAP:
                    return k
            # 사전에 없어도 전체 cand 반환 ("옆" 단독 반환 방지)
            return cand

    # 2) "~에서" 패턴
    m = re.search(r"([가-힣A-Za-z0-9\s]+?)\s*에서", text)
    if m:
        cand = m.group(1).strip()
        # 사전 최장일치
        best = None
        for key in LOCATION_MAP.keys():
            if key in cand and (best is None or len(key) > len(best)):
                best = key
        return best or cand

    # 3) 사전 LOCATION_MAP 키 직접 포함 (최장일치)
    best = None
    for key in LOCATION_MAP.keys():
        if key in text and (best is None or len(key) > len(best)):
            best = key
    return best


def _translate_to_english(korean_text: str, memory_instance=None) -> str:
    """한국어를 영어로 변환 (사전 매핑 + LLM fallback)"""
    # ✅ Fallback: memory_instance가 없으면 원문 반환
    if not memory_instance:
        return korean_text
        
    try:
        # 간단한 캐시 체크 (같은 텍스트에 대해 반복 변환 방지)
        if not hasattr(memory_instance, '_translation_cache'):
            memory_instance._translation_cache = {}
        
        if korean_text in memory_instance._translation_cache:
            return memory_instance._translation_cache[korean_text]
        
        # 1. 기본 사전 매핑 시도
        basic_mapping = {
            "열쇠": "key",
            "지갑": "wallet", 
            "핸드폰": "phone",
            "책": "book",
            "펜": "pen",
            "컵": "cup",
            "물": "water",
            "책상": "desk",
            "침대": "bed",
            "의자": "chair",
            "소파": "sofa",
            "테이블": "table",
            "냉장고": "refrigerator",
            "정수기": "water purifier",
            "세탁기": "washing machine",
            "에어컨": "air conditioner",
            "화장실": "bathroom",
            "방": "room",
            "거실": "living room",
            "부엌": "kitchen",
            "침실": "bedroom",
            "위에": "on",
            "아래": "under",
            "안에": "inside",
            "옆에": "beside",
            "뒤에": "behind",
            "앞에": "in front of"
        }
        
        if korean_text in basic_mapping:
            english_text = basic_mapping[korean_text]
            memory_instance._translation_cache[korean_text] = english_text
            print(f"[DEBUG] 사전 매핑 사용: '{korean_text}' -> '{english_text}'")
            return english_text
        
        # 2. LLM을 사용한 번역 (fallback)
        # ✅ 위치 표현 번역 개선: "정수기 옆에" → "next to water purifier" 형태로 번역
        prompt = f"""
다음 한국어 위치 표현을 자연스러운 영어로 번역하세요.

**번역 규칙:**
1. 위치 표현 전체를 번역하세요 (단어 하나만이 아닌 전체 구문)
2. "~ 옆에" → "next to ~" 또는 "beside ~"
3. "~ 위에" → "on ~" 또는 "on top of ~"
4. "~ 아래" → "under ~" 또는 "below ~"
5. "~ 안에" → "inside ~" 또는 "in ~"
6. "~ 앞에" → "in front of ~"
7. "~ 뒤에" → "behind ~"
8. 장소명도 함께 번역하세요 (예: "정수기" → "water purifier", "냉장고" → "refrigerator")

**예시:**
- "정수기 옆에" → "next to water purifier"
- "냉장고 위에" → "on refrigerator"
- "책상 위" → "on desk"
- "화장실 서랍" → "drawer in bathroom"
- "내 방 안" → "in my room" 또는 "inside my room"
- "내 방 안에" → "in my room" 또는 "inside my room"
- "방 안에" → "in the room" 또는 "inside the room"
- "옆에" → "beside" (단독 사용 시)

**중요:**
- 문장, 예문, 설명, 따옴표 없이 **번역 결과만** 출력하세요.
- 자연스러운 영어 위치 표현으로 번역하세요.

한국어: {korean_text}
영어:
"""
        
        response = memory_instance.llm.invoke(prompt)
        # AIMessage 객체를 문자열로 변환
        if hasattr(response, 'content'):
            english_text = response.content.strip()
        else:
            english_text = str(response).strip()
        
        # 결과 검증 (영어인지 확인)
        if english_text and len(english_text) < 50 and not any(ord(char) >= 0xAC00 and ord(char) <= 0xD7A3 for char in english_text):
            memory_instance._translation_cache[korean_text] = english_text
            print(f"[DEBUG] LLM 번역 사용: '{korean_text}' -> '{english_text}'")
            return english_text
        else:
            # 번역 결과가 이상하면 원문 반환
            print(f"[WARNING] 번역 결과 이상: '{english_text}', 원문 사용")
            memory_instance._translation_cache[korean_text] = korean_text
            return korean_text
        
    except Exception as e:
        print(f"[ERROR] 영어 변환 실패: {e}")
        # 변환 실패 시 원문 반환
        memory_instance._translation_cache[korean_text] = korean_text
        return korean_text


def _handle_direct_location_command(user_input: str, item_name: str, location: str, action: str, memory_instance, session_id: str) -> dict:
    """위치가 직접 언급된 명령 처리"""
    # 위치 정보를 VectorStore에 저장
    memory_instance._add_to_vstore(
        "user.물건", 
        {"이름": item_name, "위치": location},
        {"session_id": session_id, "entity_key": "user.물건", "type": "entity"}
    )
    
    # ✅ 새로운 to_task_command_en 함수 사용
    cmd = to_task_command_en(action, item_name, location, memory_instance)
    
    # ✅ 항상 dict로 반환 (message + robot_command)
    return {
        "success": True,
        "message": f"{item_name}{josa(item_name, ('의','의'))} 위치를 기억해뒀습니다. 로봇에게 명령을 전달했어요.",
        "robot_command": cmd
    }


# _handle_location_lookup_command 함수 제거됨 - handle_physical_task로 통합


def _handle_location_save_response(user_input: str, question_data: dict, memory_instance, session_id: str) -> str:
    """위치 정보 저장 응답 처리"""
    item_name = question_data["item_name"]
    action = question_data["action"]
    
    # 위치 정보 추출
    location = _extract_location_from_input(user_input)
    if not location:
        return f"죄송해요, {item_name}의 위치를 명확히 알려주세요. (예: 거실, 부엌, 현관 등)"
    
    # VectorStore에 위치 정보 저장
    memory_instance._add_to_vstore(
        "user.물건",
        {"이름": item_name, "위치": location},
        {"session_id": session_id, "entity_key": "user.물건", "type": "entity"}
    )
    
    # 재질문 상태 초기화
    del memory_instance.pending_question[session_id]
    if session_id in memory_instance.current_question:
        del memory_instance.current_question[session_id]
    
    # 후속 질문
    if action == "organize":
        question = f"{item_name}{josa(item_name, ('의','의'))} 위치를 기억해뒀습니다. {item_name}을 {location}에 정리해둘까요?"
    else:
        question = f"{item_name}{josa(item_name, ('의','의'))} 위치를 기억해뒀습니다. 가져다 드릴까요?"
    
    # 새로운 재질문 상태 저장
    memory_instance.pending_question[session_id] = {
        "type": "location_confirmed",
        "item_name": item_name,
        "location": location,
        "action": action,
        "question": question
    }
    memory_instance.current_question[session_id] = question
    
    return question


def _handle_task_execution_response(user_input: str, question_data: dict, memory_instance, session_id: str) -> str:
    """작업 실행 여부 응답 처리"""
    item_name = question_data["item_name"]
    location = question_data["location"]
    action = question_data["action"]
    
    # 부정 응답 확인 (우선 처리)
    no_keywords = ["아니", "괜찮", "됐어", "그만", "취소"]
    is_no = any(keyword in user_input for keyword in no_keywords)
    
    if is_no:
        # 재질문 상태 초기화
        del memory_instance.pending_question[session_id]
        if session_id in memory_instance.current_question:
            del memory_instance.current_question[session_id]
        return "알겠습니다. 다른 도움이 필요하시면 말씀해주세요."
    
    # 긍정 응답 확인 (강화)
    yes_keywords = ["네", "응", "그래", "맞아", "좋아", "해줘", "해주세요", "가져다", "가져", "가지고", "정리해", "가져와", "부탁해", "고마워", "감사", "고맙"]
    is_yes = any(keyword in user_input for keyword in yes_keywords)
    
    # 재질문 상태 초기화
    del memory_instance.pending_question[session_id]
    if session_id in memory_instance.current_question:
        del memory_instance.current_question[session_id]
    
    if is_yes:
        # ✅ 새로운 to_task_command_en 함수 사용
        cmd = to_task_command_en(action, item_name, location, memory_instance)
        
        return {
            "success": True,
            "message": f"로봇에게 명령을 전달했어요: {json.dumps(cmd, ensure_ascii=False)}",
            "robot_command": cmd
        }
    else:
        return "알겠습니다. 다른 도움이 필요하시면 말씀해주세요."


# ========== LCEL History 기반 핸들러 ==========

def handle_query_with_lcel(user_input: str, memory_instance, session_id: str) -> str:
    """
    Query 요청 처리:
    1. LCEL 메모리(history) 우선 참고 (방금 대화)
    2. 대화 맥락 기반 답변
    3. 엑셀 데이터 검색
    4. fallback
    """
    import re  # 함수 내부에서 re 사용 시 명시적으로 import 필요
    try:
        # 0️⃣ LLM 분류 기반: 사용자 프로필 질의 우선 처리
        text = (user_input or "").strip()
        from life_assist_dm.life_assist_dm.dialog_manager.config.config_loader import (
            get_personal_info_config,
            get_excel_sheets,
        )

        def _classify_profile_query_llm(text: str):
            try:
                llm = getattr(memory_instance, 'llm', None)
                if not llm:
                    return {"type": "other", "fields": []}
                prompt = (
                    "다음 질문이 사용자 프로필(나이/학교/직업/회사/취미/인턴)에 관한 것인지 분류하고, 필요한 필드를 나열하세요.\n"
                    "가능한 필드: name, age, school, job, company, hobby, intern\n"
                    "출력은 JSON 한 줄: {\"type\": \"user_profile|item_location|other\", \"fields\": [..]}\n"
                    f"질문: {text}"
                )
                resp = memory_instance.llm.invoke(prompt)
                raw = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
                import json as _json
                data = _json.loads(raw)
                if not isinstance(data, dict):
                    return {"type": "other", "fields": []}
                data['fields'] = [f for f in (data.get('fields') or []) if f in ["name","age","school","job","company","hobby","intern"]]
                return data
            except Exception:
                return {"type": "other", "fields": []}

        classification = _classify_profile_query_llm(text)

        if classification.get("type") == "user_profile":
            from life_assist_dm.life_assist_dm.user_excel_manager import UserExcelManager
            user_name = memory_instance.user_names.get(session_id or "default")
            if not user_name:
                return "사용자 이름을 먼저 알려주세요."

            # 캐시 우선
            user_rows = memory_instance.get_excel_data(session_id, "사용자") or []
            # 없으면 엑셀에서 직접 로드
            if not user_rows:
                # ✅ memory_instance의 excel_manager 사용 (새 인스턴스 생성 금지)
                excel = memory_instance.excel_manager
                sheets = get_excel_sheets()
                df_userinfo = excel.load_sheet_data(user_name, sheets.get("user_info", "사용자정보"))
                if not df_userinfo.empty:
                    for _, row in df_userinfo.iterrows():
                        user_rows.append({
                            "이름": row.get("이름", ""),
                            "나이": row.get("나이", ""),
                            "학교": row.get("학교", ""),
                            "직업": row.get("직업", ""),
                            "취미": row.get("취미", ""),
                            "날짜": row.get("날짜", "")
                        })

            context_map = {}
            if user_rows:
                for k in ["이름","나이","학교","직업","취미","회사","인턴"]:
                    if user_rows[-1].get(k):
                        context_map[k] = user_rows[-1].get(k)
            # LCEL 메모리(history)에서 최근 발화로 보강 추출
            try:
                mem_vars = memory_instance.conversation_memory.load_memory_variables({})
                history = mem_vars.get("history", "") or ""
                text_hist = str(history)
                import re
                if not context_map.get("나이"):
                    m = re.search(r"(\d+)\s*살", text_hist)
                    if m:
                        context_map["나이"] = f"{m.group(1)}살"
                if not context_map.get("학교"):
                    m = re.search(r"([가-힣\s]+학교)", text_hist)
                    if m:
                        context_map["학교"] = m.group(1).strip()
                if not context_map.get("직업"):
                    m = re.search(r"직업\s*은\s*([가-힣A-Za-z0-9\s]+)", text_hist)
                    if m:
                        context_map["직업"] = m.group(1).strip()
                if not context_map.get("취미"):
                    m = re.search(r"취미\s*는\s*([가-힣A-Za-z0-9\s]+)", text_hist)
                    if m:
                        context_map["취미"] = m.group(1).strip()
            except Exception:
                pass

            try:
                # ✅ memory_instance의 excel_manager 사용 (새 인스턴스 생성 금지)
                excel = memory_instance.excel_manager
                sheets = get_excel_sheets()
                df_kv = excel.load_sheet_data(user_name, sheets.get("user_info_kv", "사용자정보KV"))
                if df_kv is not None and not df_kv.empty:
                    # 사용자정보 시트 값보다 KV 최신값을 우선 적용하기 위해 먼저 관련 키 제거
                    _allowed = ["이름","나이","학교","직업","취미","회사","인턴"]
                    for _k in _allowed:
                        if _k in context_map:
                            context_map.pop(_k, None)
                    # 역순 순회로 최신값부터 채우고, 한 번 채운 키는 더 이상 덮어쓰지 않음
                    for _, row in df_kv.iloc[::-1].iterrows():
                        k = str(row.get("키", "")).strip()
                        v = str(row.get("값", "")).strip()
                        if k in _allowed and v and k not in context_map:
                            context_map[k] = v
            except Exception:
                pass

            # 최종 응답은 LLM에게 위임 (하드코딩 최소화)
            try:
                from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm
                llm = get_llm()
                kv_lines = [f"- {k}: {v}" for k, v in context_map.items()]
                context_text = "\n".join(kv_lines) if kv_lines else "없음"
                prompt = (
                    f"사용자 정보(KV):\n{context_text}\n\n"
                    f"질문: {user_input}\n\n"
                    "위 사용자 정보와 질문 의도를 고려해 간단히 한국어로 답하세요.\n모르면 정직하게 모른다고 답하세요. 불필요한 항목은 말하지 마세요."
                )
                resp = llm.invoke(prompt)
                return resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
            except Exception:
                return "아직 그 정보는 기억해 둔 게 없어요. 한 번 알려주시면 다음부터 답해드릴게요."

        # 0.5️⃣ 복약정보 조회 우선: "내가 먹는 약 뭐였지?" 등 (통합 검색 함수 사용)
        # ✅ "약속"은 일정이므로 약물로 처리하지 않음
        try:
            medicine_triggers = ["먹는", "복용", "드시는", "먹었던", "기억"]
            has_medicine_trigger = any(k in user_input for k in medicine_triggers)
            # "약" 키워드 체크 (단, "약속"은 제외 - 약속은 일정)
            if not has_medicine_trigger and "약" in user_input and "약속" not in user_input:
                # "~약" 패턴 (혈압약, 감기약 등) 또는 "약 먹" 패턴 체크
                if re.search(r"[가-힣A-Za-z]+약|약\s*[먹드]", user_input):
                    has_medicine_trigger = True
            if has_medicine_trigger:
                #  통합 검색 함수 사용: 1) 복약정보 시트 → 2) 대화 기록 시트 → 3) LCEL
                def _search_medicine(df_medicine, query_text):
                    """복약정보 시트에서 약 정보 검색"""
                    df_medicine = df_medicine.fillna("")
                    valid_meds = []
                    for _, row in df_medicine.iterrows():
                        drug_name = str(row.get("약이름", "")).strip()
                        if drug_name.lower() not in ("nan", "none", ""):
                            dose_time = str(row.get("시간", "")).strip()
                            if dose_time.lower() in ("nan", "none", ""):
                                dose_time = ""
                            valid_meds.append({
                                "약이름": drug_name,
                                "시간": dose_time
                            })
                    
                    if valid_meds:
                        # 모든 약을 나열
                        med_texts = []
                        for med in valid_meds:
                            if med["시간"]:
                                med_texts.append(f"{med['시간']}에 {med['약이름']}")
                            else:
                                med_texts.append(med['약이름'])
                        
                        if len(med_texts) == 1:
                            return f"네, 기억하고 있어요! {med_texts[0]} 드시고 계시죠."
                        else:
                            med_list = ", ".join(med_texts[:-1]) + f", {med_texts[-1]}"
                            return f"네, 기억하고 있어요! {med_list} 드시고 계시죠."
                    return None
                
                result = _query_with_fallback(
                    user_input=user_input,
                    memory_instance=memory_instance,
                    session_id=session_id,
                    sheet_name="복약정보",
                    primary_search_func=_search_medicine,
                    query_type="약"
                )
                if result:
                    return result
        except Exception as e:
            print(f"[DEBUG] 복약정보 조회 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

        try:
            family_triggers = ["가족", "동생", "형", "누나", "언니", "엄마", "아빠", "부모", "이름", "누구", "뭐라고"]
            if any(k in user_input for k in family_triggers):
                # ✅ 통합 검색 함수 사용: 1) 가족관계 시트 → 2) 대화 기록 시트 → 3) LCEL
                def _search_family(df_family, query_text):
                    """가족관계 시트에서 가족 정보 검색"""
                    dfq = df_family.fillna("")
                    # 관계 키워드 추출
                    relations = ["동생", "형", "누나", "언니", "엄마", "아빠", "부모"]
                    target_rel = next((r for r in relations if r in query_text), None)
                    if target_rel and "관계" in dfq.columns:
                        dfq = dfq[dfq["관계"].astype(str) == target_rel]
                    if not dfq.empty:
                        row = dfq.iloc[-1]
                        rel = str(row.get("관계", "")).strip()
                        name = str(row.get("이름", "")).strip()
                        if name and name.lower() not in ("nan", "none", ""):
                            if rel:
                                return f"{rel}의 이름은 {name}이에요."
                            else:
                                return f"이름은 {name}이에요."
                    return None
                
                result = _query_with_fallback(
                    user_input=user_input,
                    memory_instance=memory_instance,
                    session_id=session_id,
                    sheet_name="가족관계",
                    primary_search_func=_search_family,
                    query_type="가족"
                )
                if result:
                    return result
        except Exception:
            pass

        # 2️⃣ 취향/선호 질의 우선 처리 (물건 위치 질의보다 먼저)
        # 🔧 취향/선호 관련 키워드가 하나라도 있으면 취향 질의로 간주
        pref_q_triggers = ["취향", "선호", "좋아하는", "좋아해", "제일 좋아", "좋아했던", "싫어해"]
        pref_ans_triggers = ["알아", "있어", "말해", "뭐였지", "기억", "저장", "알고"]
        # 취향 키워드가 있으면 무조건 취향 질의로 처리
        if any(k in user_input for k in pref_q_triggers):
            # 조회 의도가 있을 때만 조회, 없으면 저장 로직으로 넘어감
            if any(k in user_input for k in pref_ans_triggers):
                try:
                    user_name = memory_instance.user_names.get(session_id or "default")
                    if not user_name:
                        return "사용자 이름을 먼저 알려주세요."
                    # ✅ memory_instance의 excel_manager 사용 (새 인스턴스 생성 금지)
                    excel = memory_instance.excel_manager
                    # 1) 사용자정보KV에서 취미 키 조회
                    df_kv = excel.safe_load_sheet(user_name, "사용자정보KV")
                    prefs: list[str] = []
                    if df_kv is not None and not df_kv.empty:
                        for _, row in df_kv.iloc[::-1].iterrows():
                            if str(row.get("키", "")).strip() in ["취미", "좋아하는 음식", "좋아하는 것"]:
                                val = str(row.get("값", "")).strip()
                                if val:
                                    prefs.append(val)
                                    if len(prefs) >= 3:
                                        break
                    # 2) 사용자정보KV 시트에서 취향 검색
                    if len(prefs) < 3:
                        from life_assist_dm.life_assist_dm.dialog_manager.config.config_loader import get_excel_sheets
                        sheets = get_excel_sheets()
                        df_kv = excel.safe_load_sheet(user_name, sheets.get("user_info_kv", "사용자정보KV"))
                        if df_kv is not None and not df_kv.empty:
                            for _, row in df_kv.iloc[::-1].iterrows():
                                key = str(row.get("키", "")).strip()
                                value = str(row.get("값", "")).strip()
                                # "취향" 키로 저장된 값 찾기
                                if key == "취향" and value and value.lower() not in ("nan", "none"):
                                    if value not in prefs:
                                        prefs.append(value)
                                    if len(prefs) >= 3:
                                        break
                    if prefs:
                        return "최근 제가 기억하는 취향은 " + ", ".join(prefs[:3]) + " 등이 있어요."
                    return "아직 명확한 취향 정보는 없어요. 한 번 알려주시면 기억해둘게요!"
                except Exception:
                    pass

        # 3️⃣ 물건 위치 조회 - 대화 맥락이 없을 때만 (명시적 물건 위치 질문)
        # LLM이 user_profile로 분류하지 않은 경우에만 물건 위치 해석
        # 🔧 취향/선호 관련 키워드가 포함되면 물건 조회로 라우팅하지 않음
        pref_exclude_keywords = ["취향", "선호", "좋아하는", "좋아해", "제일 좋아", "좋아했던", "싫어"]
        is_pref_query = any(k in user_input for k in pref_exclude_keywords)
        
        if (classification.get("type") != "user_profile" and 
            not is_pref_query and
            any(word in user_input for word in ["어디", "위치", "있어"]) and 
            not any(word in user_input for word in ["어떻게", "왜", "무엇", "뭐", "해결", "싸움", "갈등", "문제"])):
            print(f"[DEBUG] 물건 위치 조회 시작: {user_input}")
            # ✅ 통합 검색 함수 사용: 1) 물건위치 시트 → 2) 대화 기록 시트 → 3) LCEL
            def _search_item_location(df_items, query_text):
                """물건위치 시트에서 물건 위치 검색"""
                df_items = df_items.fillna("")
                # 사용자 입력에서 물건 이름 추출 시도
                for _, row in df_items.iterrows():
                    item_name = str(row.get("물건이름", "")).strip() or str(row.get("이름", "")).strip()
                    if item_name and item_name.lower() not in ("nan", "none", ""):
                        # 물건 이름이 질문에 포함되어 있거나, 질문에 물건 이름이 명시되지 않은 경우 모든 물건 반환
                        if item_name in query_text or ("어디" in query_text and "위치" in query_text):
                            # 위치 정보 조회: "위치" 필드 우선, 없으면 "장소"와 "세부위치" 조합
                            location = str(row.get("위치", "")).strip()
                            if not location or location.lower() in ("nan", "none", ""):
                                # "위치" 필드가 없으면 "장소"와 "세부위치" 조합
                                place = str(row.get("장소", "")).strip()
                                sub_location = str(row.get("세부위치", "")).strip()
                                # nan, None 필터링
                                if place.lower() in ['nan', 'none', '']:
                                    place = ''
                                if sub_location.lower() in ['nan', 'none', '']:
                                    sub_location = ''
                                # 조합
                                if place and sub_location:
                                    location = f"{place} {sub_location}"
                                elif place:
                                    location = place
                                elif sub_location:
                                    location = sub_location
                            
                            if location and location.lower() not in ("nan", "none", ""):
                                return f"{item_name}{josa(item_name, ('은','는'))} {location}에 있어요."
                return None
            
            result = _query_with_fallback(
                user_input=user_input,
                memory_instance=memory_instance,
                session_id=session_id,
                sheet_name="물건위치",
                primary_search_func=_search_item_location,
                query_type="물건"
            )
            if result:
                return result
            return "해당 물건의 위치는 아직 기록되어 있지 않아요."
        
        # 3.5️⃣ 식사 조회: "오늘 아침 뭐 먹었지?" 등 최근 식사 기록 확인
        try:
            # 식사 조회 트리거 키워드 확장 (공백 유무 모두 포함)
            meal_triggers = [
                "뭐 먹었", "뭐먹었", "먹었지", "먹었더라", "먹었어", "먹었어요",
                "먹은 게", "먹은거", "먹은것", "식사", "뭐 드셨어", "뭐드셨어",
                "뭐 드셨", "뭐드셨", "아침으로", "점심으로", "저녁으로"
            ]
            # 트리거 키워드가 하나라도 포함되어 있는지 확인
            has_meal_trigger = any(k in user_input for k in meal_triggers)
            
            if has_meal_trigger:
                meal_time = None
                if "아침" in user_input or "아침으로" in user_input:
                    meal_time = "아침"
                elif "점심" in user_input or "점심으로" in user_input:
                    meal_time = "점심"
                elif "저녁" in user_input or "밤" in user_input or "저녁으로" in user_input:
                    meal_time = "저녁"

                # 날짜 정규화 - _normalize_date_to_iso 함수 사용
                extracted_date = _extract_date(user_input)
                if extracted_date:
                    target_date = _normalize_date_to_iso(extracted_date)
                else:
                    # 날짜가 명시되지 않으면 오늘 날짜로 기본 설정 (식사 조회는 일반적으로 최근 기록을 찾는 것이므로)
                    target_date = _normalize_date_to_iso("오늘")

                user_name = memory_instance.user_names.get(session_id or "default")
                if user_name:
                    excel = memory_instance.excel_manager
                    df_meal = excel.safe_load_sheet(user_name, "음식기록")
                    if df_meal is not None and not df_meal.empty:
                        df_meal = df_meal.fillna("")
                        df_meal["날짜_str"] = df_meal["날짜"].astype(str)
                        
                        # 날짜 형식 정규화: "2025-11-03" 형태로 통일하여 비교
                        # 엑셀의 날짜 형식이 다양할 수 있으므로 유연하게 처리
                        cond = True
                        if meal_time:
                            cond = cond & (df_meal["끼니"] == meal_time)
                        if target_date:
                            # 날짜 문자열 매칭 개선: 다양한 날짜 형식 지원
                            date_cond = (
                                df_meal["날짜_str"].str.contains(target_date, na=False) |
                                df_meal["날짜_str"].str.contains(target_date.replace("-", "/"), na=False) |
                                df_meal["날짜_str"].str.contains(target_date.replace("-", "."), na=False)
                            )
                            cond = cond & date_cond
                        
                        matched = df_meal[cond]
                        
                        # 매칭 결과가 없으면 끼니만으로 최근 기록 검색
                        if matched.empty and meal_time:
                            meal_only_cond = (df_meal["끼니"] == meal_time)
                            matched = df_meal[meal_only_cond]
                        
                        # 여전히 없으면 전체 중 최근 기록 검색
                        if matched.empty:
                            matched = df_meal.tail(5)  # 최근 5개 기록
                        
                        if not matched.empty:
                            row = matched.iloc[-1]  # 가장 최근 기록
                            menu = row.get("메뉴", "")
                            
                            # 메뉴 처리: 리스트 또는 문자열 형식 모두 지원
                            if isinstance(menu, list):
                                menu_text = ", ".join([str(m) for m in menu if str(m).strip()])
                            elif isinstance(menu, str):
                                # JSON 문자열인 경우 파싱 시도
                                try:
                                    import json
                                    menu_list = json.loads(menu)
                                    if isinstance(menu_list, list):
                                        menu_text = ", ".join([str(m) for m in menu_list if str(m).strip()])
                                    else:
                                        menu_text = str(menu).strip()
                                except:
                                    menu_text = str(menu).strip()
                            else:
                                menu_text = str(menu).strip() if menu else ""
                            
                            if menu_text:
                                # 날짜 표시 개선
                                record_date = row.get("날짜_str", "").strip()
                                record_meal_time = row.get("끼니", "").strip()
                                record_actual_time = row.get("시간", "").strip()
                                
                                # 날짜가 오늘과 같으면 "오늘"로 표시
                                if record_date and target_date in record_date:
                                    date_display = "오늘"
                                else:
                                    date_display = record_date if record_date else "최근"
                                
                                # 끼니와 실제 시간 모두 표시
                                meal_time_display = record_meal_time if record_meal_time else meal_time or ""
                                if record_actual_time and record_actual_time.lower() not in ("nan", "none", ""):
                                    if meal_time_display:
                                        return f"{date_display} {meal_time_display} {record_actual_time}에 {menu_text}를 드셨어요.".strip()
                                    else:
                                        return f"{date_display} {record_actual_time}에 {menu_text}를 드셨어요.".strip()
                                else:
                                    # 시간이 없으면 끼니만 표시
                                    if meal_time_display:
                                        return f"{date_display} {meal_time_display}에는 {menu_text}를 드셨어요.".strip()
                                    else:
                                        return f"{date_display} {menu_text}를 드셨어요.".strip()
                            else:
                                return "기록된 식사 정보를 찾았지만 메뉴 정보가 없어요."
                        else:
                            # 조회 실패 시 LCEL Fallback: 최근 대화에서 식사 언급 추출
                            try:
                                mem_vars = memory_instance.conversation_memory.load_memory_variables({})
                                history = mem_vars.get("history", "") or ""
                                if history:
                                    prompt = (
                                        "최근 대화 기록에서 사용자가 무엇을 먹었다고 언급했는지 한 문장으로 답하세요.\n"
                                        "없으면 '최근 식사 기록을 찾지 못했어요.'라고 답하세요.\n\n"
                                        f"최근 대화 기록:\n{history}\n"
                                    )
                                    resp = memory_instance.llm.invoke(prompt)
                                    text = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
                                    if text:
                                        return text
                            except Exception:
                                pass
                            # 조회 실패 시 명확한 메시지 반환
                            if meal_time:
                                return f"{meal_time}에 드신 식사 기록을 찾지 못했어요."
                            else:
                                return "최근 식사 기록을 찾지 못했어요."
                # 사용자 이름이 없으면 조용히 다음 단계로
        except Exception as e:
            # 디버깅을 위한 예외 정보 출력
            print(f"[DEBUG] 식사 조회 중 오류 발생: {e}")
            import traceback
            traceback.print_exc()

        # 3️⃣ 감정 상태 조회 - 통합 검색 함수 사용 (엑셀 시트 → 대화 기록 → LCEL)
        emotion_query_triggers = ["감정상태", "감정 상태", "기분", "감정", "요즘.*감정", "어떤.*감정", "최근.*감정", 
                                  "피곤", "힘들", "좋아", "힘들다고", "피곤하다고", "어떤 줄 알아"]
        if any(re.search(pattern, user_input) for pattern in emotion_query_triggers):
            # ✅ 통합 검색 함수 사용: 1) 감정기록 시트 → 2) 대화 기록 시트 → 3) LCEL
            def _search_emotion(df_emotion, query_text):
                """감정기록 시트에서 감정 상태 검색"""
                df_emotion = df_emotion.fillna("")
                if df_emotion.empty:
                    return None
                
                # 최근 감정 기록 추출 (최근 5개)
                recent_emotions = []
                for _, row in df_emotion.tail(5).iterrows():
                    emotion = str(row.get("감정", "")).strip()
                    date = str(row.get("날짜", "")).strip()
                    info = str(row.get("정보", "")).strip()
                    
                    if emotion and emotion.lower() not in ("nan", "none", ""):
                        emotion_entry = {"감정": emotion}
                        if date and date.lower() not in ("nan", "none", ""):
                            emotion_entry["날짜"] = date
                        if info and info.lower() not in ("nan", "none", ""):
                            emotion_entry["정보"] = info
                        recent_emotions.append(emotion_entry)
                
                if recent_emotions:
                    # 감정 상태 요약 생성
                    emotion_summary = []
                    for e in recent_emotions:
                        if e.get("정보"):
                            emotion_summary.append(f"{e['감정']} ({e.get('정보', '')})")
                        else:
                            emotion_summary.append(e['감정'])
                    
                    if len(emotion_summary) == 1:
                        return f"최근에 {emotion_summary[0]} 상태를 느끼셨다고 기록되어 있어요."
                    else:
                        summary_text = ", ".join(emotion_summary[:-1]) + f", {emotion_summary[-1]}"
                        return f"최근에 {summary_text} 상태를 느끼셨다고 기록되어 있어요."
                
                return None
            
            result = _query_with_fallback(
                user_input=user_input,
                memory_instance=memory_instance,
                session_id=session_id,
                sheet_name="감정기록",
                primary_search_func=_search_emotion,
                query_type="감정"
            )
            if result:
                return result
        
        # 3.5️⃣ 복약정보 조회는 위의 0.5️⃣에서 이미 처리됨 (중복 제거)
        
        # 4️⃣ 일반적인 엑셀/사용자 KV 기반 답변 (LLM 컨텍스트로 전달)
        user_name = memory_instance.user_names.get(session_id or "default")
        if user_name:
            # 🔎 일정 조회: 통합 검색 함수 사용
            try:
                if any(k in user_input for k in ["일정", "스케줄", "예약"]):
                    # ✅ 통합 검색 함수 사용: 1) 일정 시트 → 2) 대화 기록 시트 → 3) LCEL
                    def _search_schedule(df_sched, query_text):
                        """일정 시트에서 일정 정보 검색"""
                        dfq = df_sched.fillna("")
                        # 질의 날짜 파싱 및 정규화 (YYYY-MM-DD 형식으로 변환)
                        qdate = _extract_date(query_text)
                        if qdate:
                            # 질의 날짜를 YYYY-MM-DD 형식으로 정규화
                            normalized_qdate = _normalize_date_to_iso(qdate)
                            # 엑셀의 날짜 컬럼과 정확히 일치하는지 확인 (YYYY-MM-DD 형식)
                            dfq = dfq[dfq["날짜"].astype(str).str.strip() == normalized_qdate]
                        # 제목/날짜가 유효한 행만
                        if not dfq.empty:
                            # 여러 건일 때 최근/마지막 우선
                            row = dfq.iloc[-1]
                            title = str(row.get("제목", "")).strip()
                            # nan/None 필터링
                            if title.lower() in ("nan", "none", ""):
                                title = str(row.get("내용", "")).strip()
                            if title.lower() in ("nan", "none", ""):
                                title = "일정"
                            
                            date = str(row.get("날짜", "")).strip()
                            time = str(row.get("시간", "")).strip()
                            
                            # nan/None 필터링 개선
                            import pandas as pd
                            if pd.isna(row.get("시간")) or time.lower() in ("nan", "none", ""):
                                time = ""
                            if pd.isna(row.get("날짜")) or date.lower() in ("nan", "none", ""):
                                date = ""
                            if pd.isna(row.get("제목")):
                                title = "일정" if not title or title.lower() in ("nan", "none", "") else title
                            
                            # 날짜와 시간 조합
                            if date and time:
                                parts = f"{date} {time}"
                            elif date:
                                parts = date
                            elif time:
                                parts = time
                            else:
                                parts = ""
                            
                            # 응답 생성
                            if parts:
                                return f"{parts}에 {title} 일정이 있어요."
                            else:
                                return f"{title} 일정이 있어요."
                        return None
                    
                    result = _query_with_fallback(
                        user_input=user_input,
                        memory_instance=memory_instance,
                        session_id=session_id,
                        sheet_name="일정",
                        primary_search_func=_search_schedule,
                        query_type="일정"
                    )
                    if result:
                        return result
            except Exception:
                pass
            # 사용자 KV 최신 맵 구성
            kv_map = {}
            try:
                from life_assist_dm.life_assist_dm.dialog_manager.config.config_loader import get_excel_sheets
                sheets = get_excel_sheets()
                # ✅ memory_instance의 excel_manager 사용 (새 인스턴스 생성 금지)
                excel = memory_instance.excel_manager
                df_kv = excel.load_sheet_data(user_name, sheets.get("user_info_kv", "사용자정보KV"))
                if df_kv is not None and not df_kv.empty:
                    for _, row in df_kv.iloc[::-1].iterrows():
                        k = str(row.get("키", "")).strip()
                        v = str(row.get("값", "")).strip()
                        if not k or v.lower() in ("nan", "none") or v == "":
                            continue
                        if k not in kv_map:
                            kv_map[k] = v
            except Exception:
                kv_map = {}

            # LLM에 KV와 함께 질문
            try:
                from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm
                llm = get_llm()
                context_lines = [f"- {k}: {v}" for k, v in kv_map.items()]
                context_text = "\n".join(context_lines)
                prompt = (
                    f"사용자 정보(KV):\n{context_text if context_text else '없음'}\n\n"
                    f"질문: {user_input}\n\n"
                    "위 KV와 일반 상식을 참고해 간단히 한국어로 답하세요. "
                    "모르면 정직하게 모른다고 답하세요."
                )
                resp = llm.invoke(prompt)
                return resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
            except Exception:
                pass

            # 기타 시트 대신 사용자정보KV 시트에서 조회
            excel_data = memory_instance.get_excel_data(session_id, "사용자정보KV")
            if excel_data:
                context = "\n".join([str(item) for item in excel_data[-5:]])
                prompt = f"""질문: {user_input}

저장된 정보: {context}

자연스럽게 답변하세요."""
                response = memory_instance.llm.invoke(prompt)
                return response.content.strip()

        # 5️⃣ 엑셀 대화 기록 확인 (우선)
        if user_name:
            conversations = memory_instance.get_excel_data(session_id, "대화")
            if conversations:
                recent_conversations = conversations[-3:]  # 최근 3개
                conv_text = "\n".join([f"- {conv['대화요약']}" for conv in recent_conversations])
                prompt = f"""최근 대화 기록:
                    {conv_text}

                    사용자 질문: {user_input}

                    위 대화 기록을 바탕으로 사용자의 질문에 답변하세요. 
                    대화 기록과 관련 없는 일반적인 정보 조회는 하지 마세요.
                    """
                response = memory_instance.llm.invoke(prompt)
                return response.content.strip()

        # 6️⃣ LCEL 메모리 (대화 기록이 없을 때만)
        mem_vars = memory_instance.conversation_memory.load_memory_variables({})
        history = mem_vars.get("history", "")
        if history:
            context = memory_instance._build_context_for_llm(user_input, session_id) if hasattr(memory_instance, '_build_context_for_llm') else ""
            prompt = f"""{context}대화 맥락:
                    {history}

                    사용자 질문: {user_input}

                    위 대화 맥락을 바탕으로 사용자의 질문에 답변하세요. 
                    대화 맥락과 관련 없는 일반적인 정보 조회는 하지 마세요.
                    대화의 흐름에 맞는 답변을 해주세요.
                    """
            response = memory_instance.llm.invoke(prompt)
            return response.content.strip()

        # 7️⃣ fallback - 대화 맥락이 없을 때만
        return "아직 기록된 정보가 없어요. 알려주시면 기억해둘게요!"

    except Exception as e:
        import traceback
        traceback.print_exc()
        # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
        return "정보 조회 중 오류가 발생했어요. 다시 시도해 주세요."


def handle_cognitive_task_with_lcel(user_input: str, memory_instance, session_id: str) -> str:
    """
    Cognitive 요청 처리:
    1. 중복 응답 처리 체크
    2. 엔티티 추출 및 저장
    3. LCEL 메모리(history) 참고
    4. 엑셀 저장
    5. fallback
    """
    try:
        text_norm = (user_input or "").strip()
        q_trigger = (
            text_norm.endswith("?") or
            any(k in text_norm for k in ["어디", "어딨어", "위치", "기억", "알고", "알아"]) or
            "몇 살" in text_norm or "몇살" in text_norm
        )
        if q_trigger:
            return handle_query_with_lcel(user_input, memory_instance, session_id)

        # 0.5️⃣ 감정(정서) 직접 감지 및 저장 (질문이 아닐 때만)
        # 주의: 다른 엔티티(약, 일정 등)가 함께 있을 수 있으므로 바로 return하지 않고 플래그 사용
        emotion_saved = False
        medicine_saved = False
        try:
            em_text = (user_input or "").strip()
            
            # ✅ 공통 함수 사용하여 감정 단어와 라벨 추출
            emotion_word, label = _extract_emotion_word_and_label(em_text)

            if label or emotion_word:
                try:
                    user_name = memory_instance.user_names.get(session_id or "default")
                except Exception:
                    user_name = None
                if user_name and user_name != "사용자":
                    # 순수 감정 표현만 있는 경우(다른 정보 없음)에만 바로 저장하고 return
                    # 다른 엔티티가 있을 수 있는 경우는 아래 엔티티 처리 단계에서 처리
                    has_other_keywords = any(k in em_text for k in [
                        "약", "일정", "약속", "병원", "치과", "미용실",
                        "식사", "밥", "음식", "약물", "복용"
                    ])
                    
                    if not has_other_keywords:
                        # ✅ 실제 감정 단어가 있으면 그것을 저장, 없으면 라벨 저장
                        emotion_to_save = emotion_word if emotion_word else label
                        # 순수 감정 표현만 있는 경우
                        # ✅ 감정의 원인/상황을 간결하게 요약
                        info_summary = _summarize_emotion_context_for_save(em_text, memory_instance.llm if hasattr(memory_instance, 'llm') else None)
                        
                        memory_instance.excel_manager.save_entity_data(user_name, "정서", {
                            "감정": emotion_to_save,
                            "정보": info_summary,
                        })
                        return f"그 마음 이해해요. {emotion_to_save}하게 느끼신 걸 기록해둘게요."
                    # 다른 정보도 있을 수 있으므로 엔티티 추출 후 처리 (플래그는 이미 False로 초기화됨)
        except Exception:
            pass

        # 1️⃣ LLM 기반 사용자 정보 추출 (질문이 아닌 서술형에 한해 저장)
        def _extract_user_profile_llm(text: str) -> dict:
            try:
                llm = getattr(memory_instance, 'llm', None)
                if not llm:
                    return {}
                prompt = (
                    "아래 문장에서 사용자 프로필 정보를 JSON으로 추출하세요. 존재하지 않으면 null.\n"
                    "반드시 키는 다음만 사용: name, age, school, job, company, hobby, intern.\n"
                    "age는 숫자만, school은 학교명만(조사/접사 제거), company는 기관/회사명만.\n"
                    "응답은 JSON 한 줄만 출력. 예: {\"age\":22,\"school\":\"경희대학교\",\"company\":\"KETI\",\"intern\":true}\n\n"
                    f"문장: {text}"
                )
                resp = llm.invoke(prompt)
                raw = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
                import json as _json
                data = _json.loads(raw)
                if isinstance(data, dict):
                    return {k: v for k, v in data.items() if v not in (None, "", [])}
                return {}
            except Exception:
                return {}

        if not text_norm.endswith("?"):
            # 일정/시간/약/물건 문맥이면 사용자 프로필 저장을 시도하지 않음
            schedule_like = any(k in text_norm for k in ["오늘", "내일", "모레", "이번주", "다음주", "요일", "시", "약속", "미팅", "회의", "예약"]) \
                or bool(re.search(r"(오전|오후|저녁|아침)?\s*\d{1,2}\s*시", text_norm))
            medication_like = any(k in text_norm for k in ["약", "복용", "알", "정", "캡슐"]) and ("약속" not in text_norm)
            item_like = any(k in text_norm for k in ["있어", "위에", "안에", "서랍", "찬장", "가져와", "가져다", "놓여", "보관"]) 

            # 명시적 프로필 신호(직업/회사/학교/나이/취미 표현)가 있는 경우에만 허용
            explicit_profile = any(k in text_norm for k in ["직업", "회사", "직장", "다녀", "다닙", "근무", "소속", "학교", "대학교", "중학교", "고등학교", "나이", "살", "취미"]) 

            extracted = {} if (schedule_like or medication_like or item_like or not explicit_profile) else _extract_user_profile_llm(text_norm)
            if extracted:
                # 후처리: age 숫자→"n살", intern true→job/company 보강 X (그대로 저장)
                if 'age' in extracted and extracted['age'] is not None:
                    try:
                        extracted['나이'] = f"{int(extracted['age'])}살"
                    except Exception:
                        pass
                if 'school' in extracted and extracted['school']:
                    extracted['학교'] = str(extracted['school']).strip()
                if 'job' in extracted and extracted['job']:
                    extracted['직업'] = str(extracted['job']).strip()
                if 'company' in extracted and extracted['company']:
                    extracted['회사'] = str(extracted['company']).strip()
                if 'hobby' in extracted and extracted['hobby']:
                    extracted['취미'] = str(extracted['hobby']).strip()
                # intern은 bool/문자 모두 허용하여 KV에 '인턴' 키로 저장
                if 'intern' in extracted and extracted['intern']:
                    extracted['인턴'] = "true" if str(extracted['intern']).lower() in ("true","1","yes") else str(extracted['intern'])

                # 저장용 필터링
                save_map = {k: v for k, v in extracted.items() if k in ("이름","나이","학교","직업","취미","회사","인턴") and v}
                if save_map:
                    user_name = memory_instance.user_names.get(session_id or "default")
                    if user_name:
                        memory_instance.excel_manager.save_entity_data(user_name, "사용자", save_map)
                        fields = list(save_map.keys())
                        if len(fields) == 1:
                            return f"사용자의 {fields[0]} 정보를 저장했어요."
                        else:
                            return f"사용자의 {', '.join(fields)} 정보를 저장했어요."

        # 인턴 기관 추출 및 저장
        intern_pat = r"(?:나는|난|저는)?\s*([가-힣A-Za-z0-9\s]+?)\s*(?:에서|에)\s*인턴(?:을)?\s*(?:하고\s*있어(?:요)?|중이야|중이에요|중입니다|해(?:요)?)"
        m_intern = re.search(intern_pat, text_norm)
        if m_intern and not text_norm.endswith("?"):
            org = m_intern.group(1).strip()
            user_name = memory_instance.user_names.get(session_id or "default")
            if user_name:
                memory_instance.excel_manager.save_entity_data(user_name, "사용자", {"직업": "인턴", "회사": org})
                print(f"[DEBUG] 인턴 기관 직접 저장: 회사={org} (사용자: {user_name})")
                return f"인턴 정보를 저장했어요: {org}에서 인턴 중"

        # 1️⃣ 중복 응답 처리 체크
        if hasattr(memory_instance, 'pending_question') and memory_instance.pending_question.get(session_id):
            pending_data = memory_instance.pending_question[session_id]
            print(f"[DEBUG] 중복 응답 처리: {user_input}")
            result = memory_instance.handle_duplicate_answer(user_input, pending_data)
            
            if session_id in memory_instance.pending_question:
                del memory_instance.pending_question[session_id]
            
            return result["message"]
        
        
        
        entities = memory_instance._pre_extract_entities(user_input, session_id)
        print(f"[DEBUG] handle_cognitive_task_with_lcel에서 추출된 엔티티: {entities}")
        
        
        has_valid_entities = False
        if entities and isinstance(entities, dict):
            for entity_list in entities.values():
                if entity_list and isinstance(entity_list, list) and len(entity_list) > 0:
                    # 리스트에 실제 값이 있는 엔티티가 하나라도 있으면 유효
                    has_valid_entities = True
                    break
        
        
        if not entities or not has_valid_entities:
            print(f"[DEBUG] 엔티티 추출 실패 또는 유효한 엔티티 없음, 직접 패턴 매칭 시도")
            
            
            item_location_keywords = ["있어", "위에", "안에", "보관", "놓여", "위치", "서랍", "찬장", "테이블", "식탁", "책상", "방", "주방", "화장실", "거실", "가져"]
            has_item_keywords = any(keyword in user_input for keyword in item_location_keywords)
            
            if has_item_keywords and (not entities or "user.물건" not in entities):
                try:
                    logger.debug(f"[FALLBACK] 물건 위치 키워드 감지 → rule-based 추출 재시도: {user_input}")
                    rule_entities = memory_instance._rule_based_extract(user_input, session_id)
                    if rule_entities.get("user.물건"):
                        if not entities:
                            entities = {}
                        entities["user.물건"] = rule_entities["user.물건"]
                        logger.debug(f"[FALLBACK] 물건 위치 추출 성공: {rule_entities['user.물건']}")
                        has_valid_entities = True
                except Exception as e:
                    logger.warning(f"[FALLBACK] 물건 위치 추출 재시도 실패: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            age_match = re.search(r"내\s*나이(?:는|가)?\s*(\d+)(?:살|세)", user_input)
            if age_match:
                age = f"{age_match.group(1)}살"
                if not entities:
                    entities = {}
                if "user.사용자" not in entities:
                    entities["user.사용자"] = []
                entities["user.사용자"].append({"나이": age})
                print(f"[DEBUG] 직접 나이 추출: {age}")
            
            # 학교 추출
            school_match = re.search(r"나는\s*([가-힣\s]+(?:중학교|고등학교|대학교|초등학교|학교))에?\s*다녀", user_input)
            if school_match:
                school = school_match.group(1).strip()
                if not entities:
                    entities = {}
                if "user.사용자" not in entities:
                    entities["user.사용자"] = []
                if not entities["user.사용자"]:
                    entities["user.사용자"].append({})
                entities["user.사용자"][0]["학교"] = school
                print(f"[DEBUG] 직접 학교 추출: {school}")
            

            try:
                pref_pat = r"(?:나는|난|저는)?\s*([가-힣A-Za-z0-9\s]+?)\s*(?:을|를)?\s*(?:제일\s*)?좋아해"
                m_pref = re.search(pref_pat, user_input)
                if m_pref and not user_input.strip().endswith("?"):
                    pref_item = m_pref.group(1).strip()
                    user_name = memory_instance.user_names.get(session_id or "default")
                    if user_name and user_name != "사용자":
                        memory_instance.excel_manager.save_entity_data(user_name, "취향", {"내용": f"{pref_item} 좋아함"})
                        print(f"[DEBUG] 취향 직접 저장: {pref_item}")
                        return f"취향 정보를 저장했어요: {pref_item}을(를) 좋아하시는 것으로 기록할게요."
            except Exception:
                pass

            # 약(복용) 규칙 우선 추출: 비타민/영양제/오메가3 등은 식사보다 약으로 인식
            # 주의: 엔티티 추출(_pre_extract_entities)에서 약을 추출했으면 건너뛰기
            if not medicine_saved:
                try:
                    # 약/식사 구분 강화: '먹' + 약/보충제 키워드가 있으면 무조건 약으로 처리
                    # ✅ "약속"은 일정이므로 약물로 처리하지 않음: "약" 뒤에 "속"이 오지 않는 경우만 매칭
                    med_kw = re.search(r"(약(?!속)|비타민|영양제|오메가\s*3|오메가3|철분제|프로틴|보충제|[가-힣A-Za-z]+약)", user_input)
                    if ("먹" in user_input) and med_kw:
                        # 다른 복잡한 정보(일정, 사용자 정보 등)가 함께 있는지 확인
                        has_other_info = any(k in user_input for k in [
                            "일정", "약속", "병원", "치과", "미용실", 
                            "나이", "학교", "직업", "인턴"
                        ])
                        
                        if not has_other_info:
                            # 순수 약 복용 정보만 있는 경우 바로 저장
                            when = None
                            m_when = re.search(r"(아침|점심|저녁|밤|자기\s*전)", user_input)
                            if m_when:
                                when = m_when.group(1)
                            user_name = memory_instance.user_names.get(session_id or "default")
                            if user_name and user_name != "사용자":
                                memory_instance.excel_manager.save_entity_data(user_name, "약", {
                                    "약이름": med_kw.group(1),
                                    "시간": when or "",
                                })
                                print(f"[DEBUG] 약 엔티티 규칙 기반 저장: {med_kw.group(1)} / {when or ''}")
                                return f"{med_kw.group(1)} 복용 정보를 저장했어요."
                except Exception:
                    pass

            
            # ✅ 가족 관계 키워드가 있으면 일정으로 fallback하지 않음
            # ✅ 엔티티 추출 실패 시 가족 관계 직접 추출 시도
            # ✅ task_classifier의 FAMILY_RELATION_KEYWORDS 재사용
            from life_assist_dm.life_assist_dm.task_classifier import FAMILY_RELATION_KEYWORDS
            has_family_keywords = any(keyword in user_input for keyword in FAMILY_RELATION_KEYWORDS)
            
            # ✅ 가족 관계 정보가 있고 엔티티 추출이 실패했을 때 직접 추출 시도
            if has_family_keywords and (not entities or "user.가족" not in entities):
                try:
                    logger.debug(f"[FALLBACK] 가족 관계 키워드 감지 → rule-based 추출 재시도: {user_input}")
                    rule_entities = memory_instance._rule_based_extract(user_input, session_id)
                    if rule_entities.get("user.가족"):
                        if not entities:
                            entities = {}
                        entities["user.가족"] = rule_entities["user.가족"]
                        logger.debug(f"[FALLBACK] 가족 관계 추출 성공: {rule_entities['user.가족']}")
                        has_valid_entities = True
                except Exception as e:
                    logger.warning(f"[FALLBACK] 가족 관계 추출 재시도 실패: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            has_schedule_entity = False
            if entities and isinstance(entities, dict):
                has_schedule_entity = (
                    "user.일정" in entities and 
                    entities["user.일정"] and 
                    any(e.get("제목") for e in entities["user.일정"])
                )
            
            # 물건 위치가 이미 추출되었으면 일정으로 fallback하지 않음
            has_item_entity = False
            if entities and isinstance(entities, dict):
                has_item_entity = (
                    "user.물건" in entities and 
                    entities["user.물건"] and 
                    len(entities["user.물건"]) > 0
                )
            
            # ✅ 가족 관계 키워드가 있으면 일정 fallback 건너뛰기
            if not has_schedule_entity and not has_item_entity and not has_family_keywords:
                try:
                    extracted = _extract_schedule_rule_based(user_input)
                    if extracted and extracted.get("user.일정"):
                        ent = extracted["user.일정"][0]
                        title = ent.get("제목", "")
                        date_str = ent.get("날짜", "")
                        time_str = ent.get("시간", "")
                        if title or date_str:  # 제목 또는 날짜가 있어야 저장
                            # 날짜를 YYYY-MM-DD 형식으로 정규화
                            normalized_date = _normalize_date_to_iso(date_str) if date_str else ""
                            user_name = memory_instance.user_names.get(session_id or "default")
                            if user_name and user_name != "사용자":
                                memory_instance.excel_manager.save_entity_data(user_name, "일정", {
                                    "제목": title,
                                    "날짜": normalized_date,
                                    "시간": time_str,
                                    "장소": ""
                                })
                                print(f"[DEBUG] 일정 규칙 기반 저장: {date_str} {time_str} {title}")
                                parts = " ".join(p for p in [date_str, time_str] if p)
                                return f"{parts} {title} 예약(일정)으로 기록했어요.".strip()
                except Exception:
                    pass
        
        print(f"[DEBUG] 최종 엔티티: {entities}")
        
        # 2.5️⃣ Slot-filling 응답 처리
        if isinstance(entities, dict) and entities.get("success") == False and entities.get("incomplete"):
            print(f"[DEBUG] Slot-filling 필요: {entities['message']}")
            # pending_question에 저장
            memory_instance.pending_question[session_id] = entities.get("pending_data", {})
            return entities["message"]
        
        # 3️⃣ VectorStore 저장/조회 (엔티티 기반) - 먼저 처리
        print(f"[DEBUG] 엔티티 처리 시작: entities={entities}")
        
        # 주의: 엔티티 추출은 이미 위의 2️⃣ 단계에서 완료됨
        # 패턴 매칭 fallback은 line 1722-1753에서 entities에 추가하는 방식으로 처리되므로
        # 여기서는 추가 추출 없이 entities가 있으면 처리, 없으면 LCEL fallback으로 진행
        
        if entities and isinstance(entities, dict):
            print(f"[DEBUG] 엔티티가 있어서 처리 시작")
            results = []
            
            # 엔티티를 VectorStore에 저장 (JSON 구조로 통일)
            for entity_key, entity_list in entities.items():
                for entity in entity_list:
                    if entity_key == "user.물건":
                        # 물건 엔티티는 JSON 구조로 저장
                        name = entity.get("이름", "")
                        location = entity.get("위치", "")
                        place = entity.get("장소", "")
                        sub_location = entity.get("세부위치", "")
                        # 위치 모호/빈값 가드
                        ambiguous = {None, "", "어딘가", "모름", "몰라", "unknown", "Unknown", "UNKNOWN"}
                        if name and (location or place or sub_location) and location not in ambiguous:
                            # 엑셀에 직접 저장
                            user_name = memory_instance.user_names.get(session_id or "default")
                            if user_name and user_name != "사용자":
                            
                                excel_manager = memory_instance.excel_manager
                                
                                excel_manager.save_entity_data(user_name, "user.물건", {
                                    "이름": name, 
                                    "위치": location, 
                                    "장소": place,
                                    "세부위치": sub_location,
                                    "추출방법": entity.get("추출방법", "rule-based")  
                                })
                                if place and sub_location:
                                    location_msg = f"{place} {sub_location}"
                                elif place:
                                    location_msg = place
                                elif sub_location:
                                    location_msg = sub_location
                                else:
                                    location_msg = location
                                results.append(f"'{name}'의 위치를 '{location_msg}'로 저장했어요.")
                                # 세션 캐시에 즉시 반영하여 후속 물리요청에서 즉시 사용 가능
                                try:
                                    if not hasattr(memory_instance, 'excel_cache'):
                                        memory_instance.excel_cache = {}
                                    session_cache = memory_instance.excel_cache.setdefault(session_id, {})
                                    items = session_cache.setdefault("물건", [])
                                    items.append({"이름": name, "위치": location, "장소": place, "세부위치": sub_location})
                                except Exception:
                                    pass
                            else:
                                results.append(f"'{name}'의 위치를 '{location or place or sub_location}'로 저장했어요.")
                    elif entity_key == "user.건강상태":
                        # 감정 엔티티 처리 (단, cognitive 0.5단계에서 이미 감정이 처리된 경우 제외)
                        # 건강상태는 감정과 별도로 처리하되, 이미 감정이 저장된 경우 중복 저장 방지
                        emotion = entity.get("증상", "")
                        # 감정 키워드가 있고 0.5단계에서 이미 처리된 경우 중복 저장 방지
                        # ✅ 공통 감정 단어 리스트 사용
                        all_emotion_keywords = EMOTION_POSITIVE_WORDS + EMOTION_NEGATIVE_WORDS + EMOTION_TIRED_WORDS + EMOTION_ANXIOUS_WORDS
                        em_kw_check = any(k in user_input for k in all_emotion_keywords)
                        # emotion_saved 플래그 확인 (0.5단계에서 이미 저장됨)
                        if emotion and not emotion_saved and not em_kw_check:
                            # 건강 상태 관련 증상만 별도 처리 (예: 두통, 복통 등)
                            user_name = memory_instance.user_names.get(session_id or "default")
                            if user_name and user_name != "사용자":
                                excel_manager = memory_instance.excel_manager
                                # ✅ 감정의 원인/상황을 간결하게 요약
                                info_summary = _summarize_emotion_context_for_save(user_input, memory_instance.llm if hasattr(memory_instance, 'llm') else None)
                                
                                excel_manager.save_entity_data(user_name, "정서", {
                                    "감정": emotion,
                                    "정보": info_summary
                                })
                                results.append(f"'{emotion}' 상태를 기록했어요.")
                            else:
                                results.append(f"'{emotion}' 상태를 기록했어요.")
                    elif entity_key == "user.사용자":
                        # 사용자 엔티티는 JSON 구조로 저장 (개인정보 포함)
                        print(f"[DEBUG] 사용자 엔티티 저장 시도: {entity}")
                        
                        # 엑셀에 직접 저장
                        user_name = memory_instance.user_names.get(session_id or "default")
                        if user_name and user_name != "사용자":
                            excel_manager = memory_instance.excel_manager
                            
                            # 사용자 개인정보 저장
                            user_data = {}
                            if entity.get("이름"):
                                user_data["이름"] = entity.get("이름")
                            # 별명/별칭 처리 (둘 다 지원, "별명"은 "별칭"으로 정규화)
                            nickname = entity.get("별명") or entity.get("별칭") or entity.get("alias")
                            if nickname:
                                user_data["별칭"] = nickname
                            if entity.get("나이"):
                                user_data["나이"] = entity.get("나이")
                            if entity.get("학교"):
                                user_data["학교"] = entity.get("학교")
                            if entity.get("직업"):
                                user_data["직업"] = entity.get("직업")
                            if entity.get("취미"):
                                user_data["취미"] = entity.get("취미")
                            
                            if user_data:
                                excel_manager.save_entity_data(user_name, "사용자", user_data)
                                saved_fields = list(user_data.keys())
                                if len(saved_fields) == 1:
                                    results.append(f"사용자의 {saved_fields[0]} 정보를 저장했어요.")
                                else:
                                    results.append(f"사용자의 {', '.join(saved_fields)} 정보를 저장했어요.")
                                # 세션 캐시에 즉시 반영
                                try:
                                    if not hasattr(memory_instance, 'excel_cache'):
                                        memory_instance.excel_cache = {}
                                    session_cache = memory_instance.excel_cache.setdefault(session_id, {})
                                    user_rows = session_cache.setdefault("사용자", [])
                                    # 부분 업데이트 형태라도 행으로 추가해 최신값 가용
                                    row = {**user_data}
                                    row["날짜"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                    user_rows.append(row)
                                except Exception:
                                    pass
                            else:
                                results.append("사용자 정보를 저장했어요.")
                        else:
                            results.append("사용자 정보를 저장했어요.")
                    elif entity_key == "user.가족":
                        # ✅ 가족 관계 엔티티 저장
                        print(f"[DEBUG] 가족 관계 엔티티 저장 시도: {entity}")
                        relation = entity.get("관계", "")
                        name = entity.get("이름", "")
                        
                        if relation and name:
                            user_name = memory_instance.user_names.get(session_id or "default")
                            if user_name and user_name != "사용자":
                                excel_manager = memory_instance.excel_manager
                                excel_manager.save_entity_data(user_name, "가족", {
                                    "관계": relation,
                                    "이름": name
                                })
                                results.append(f"'{relation}' '{name}'의 정보를 저장했어요.")
                                # 세션 캐시에 즉시 반영
                                try:
                                    if not hasattr(memory_instance, 'excel_cache'):
                                        memory_instance.excel_cache = {}
                                    session_cache = memory_instance.excel_cache.setdefault(session_id or "default_session", {})
                                    family_list = session_cache.setdefault("가족", [])
                                    family_list.append({"관계": relation, "이름": name})
                                except Exception:
                                    pass
                            else:
                                results.append(f"'{relation}' '{name}'의 정보를 저장했어요.")
                        elif relation:
                            # 관계만 있는 경우
                            results.append(f"{relation} 정보를 저장했어요.")
                        else:
                            results.append("가족 정보를 저장했어요.")
                    elif entity_key == "user.일정":
                        # 일정 엔티티는 JSON 구조로 저장
                        print(f"[DEBUG] 일정 엔티티 저장 시도: {entity}")
                        title = entity.get("제목", "")
                        date = entity.get("날짜", "")
                        time = entity.get("시간", "")
                        location = entity.get("장소", "")
                        
                        # nan/None 값 필터링 및 정규화
                        if str(title).lower() in ("nan", "none", ""):
                            title = ""
                        if str(date).lower() in ("nan", "none", ""):
                            date = ""
                        if str(time).lower() in ("nan", "none", ""):
                            time = ""
                        if str(location).lower() in ("nan", "none", ""):
                            location = ""
                        
                        # 날짜를 YYYY-MM-DD 형식으로 정규화
                        normalized_date = _normalize_date_to_iso(date) if date else ""
                        
                        # 제목 또는 날짜가 있어야 저장
                        if title or normalized_date:
                            print(f"[DEBUG] 일정 저장: 제목={title}, 날짜={normalized_date}, 시간={time}, 장소={location}")
                            # 엑셀에 직접 저장
                            user_name = memory_instance.user_names.get(session_id or "default")
                            if user_name and user_name != "사용자":
                                memory_instance.excel_manager.save_entity_data(user_name, "일정", {"제목": title or "", "날짜": normalized_date or "", "시간": time or "", "장소": location or ""})
                                if title:
                                    results.append(f"{title} 일정을 저장했어요.")
                                else:
                                    results.append(f"일정을 저장했어요.")
                            else:
                                if title:
                                    results.append(f"{title} 일정을 저장했어요.")
                                else:
                                    results.append(f"일정을 저장했어요.")
                        else:
                            print(f"[DEBUG] 일정 제목과 날짜가 모두 없어서 저장하지 않음")
                    elif entity_key == "user.약":
                        # 약 복용 정보 저장 (엑셀 표준 스키마 사용)
                        # 주의: 이미 위에서 약을 저장한 경우 중복 저장 방지
                        print(f"[DEBUG] 약 엔티티 저장 시도: {entity}")
                        drug_name = entity.get("약이름") or entity.get("약명") or entity.get("이름") or ""
                        # ✅ 시간대 필드 우선 확인 (rule-based에서 추출된 경우)
                        dose_time = entity.get("시간대", "") or entity.get("시간", "")
                        
                        # 용량과 단위 필드
                        용량_값 = entity.get("용량", "")
                        단위_값 = entity.get("단위", "")
                        
                        # 복용방법과 복용기간 필드
                        복용방법_값 = entity.get("복용방법", "")
                        복용기간_값 = entity.get("복용기간", "")
                        
                        # ✅ 복용 필드가 리스트 형태인 경우 파싱 (예: [{'원문': '하루 3번'}], [{'원문': '아침'}, {'원문': '저녁'}], [{'원문': '1알'}])
                        if isinstance(entity.get("복용"), list):
                            time_list = []
                            for 복용항목 in entity.get("복용", []):
                                if isinstance(복용항목, dict) and "원문" in 복용항목:
                                    원문 = 복용항목.get("원문", "")
                                    
                                    # ✅ 용량+단위 패턴 파싱 (예: "1알", "2정", "3캡슐")
                                    if not 용량_값 and not 단위_값:
                                        dose_match = re.search(r"(\d+|[한두세네다섯여섯일곱여덟아홉열])\s*(알|정|캡슐|포|mg|ml|병|씩)", 원문)
                                        if dose_match:
                                            dose_str = dose_match.group(1)
                                            unit_str = dose_match.group(2)
                                            # 한글 숫자 변환
                                            if dose_str in KOREAN_NUMBERS_STR:
                                                용량_값 = KOREAN_NUMBERS_STR[dose_str]
                                            else:
                                                용량_값 = dose_str
                                            단위_값 = unit_str.replace("씩", "").strip()
                                            continue  # 용량/단위를 찾았으면 시간대 체크는 건너뛰기
                                    
                                    # 직접적인 시간대 표현 (아침, 저녁, 점심 등)
                                    if 원문 in ["아침", "점심", "저녁", "밤", "오전", "오후"]:
                                        if 원문 not in time_list:
                                            time_list.append(원문)
                                        continue
                                    
                                    # "하루 X번" 패턴을 시간대로 변환
                                    frequency_match = re.search(r"하루\s*(?:에\s*)?(\d+|[한두세네다섯])\s*번", 원문)
                                    if frequency_match:
                                        freq_str = frequency_match.group(1)
                                        if freq_str.isdigit():
                                            frequency = int(freq_str)
                                        else:
                                            frequency = KOREAN_NUMBERS_INT.get(freq_str, 0)
                                        
                                        # 빈도에 따라 시간대 설정
                                        if frequency == 2:
                                            time_list = ["아침", "저녁"]
                                        elif frequency == 3:
                                            time_list = ["아침", "점심", "저녁"]
                                        elif frequency >= 4:
                                            time_list = ["아침", "점심", "저녁", "밤"]
                                        # frequency == 1은 시간대 설정 안 함
                                        break
                            
                            # 시간대 목록이 있으면 "/"로 구분하여 저장
                            if time_list:
                                dose_time = "/".join(time_list)
                        
                        # ✅ user_input에서 시간대 키워드 직접 추출 (LLM이 추출하지 못한 경우)
                        if not dose_time:
                            # 시간대 키워드 매핑 (memory.py의 TIME_OF_DAY_KEYWORDS 사용)
                            for time_key, keywords in TIME_OF_DAY_KEYWORDS.items():
                                if any(k in user_input for k in keywords):
                                    dose_time = time_key
                                    break
                        
                        # ✅ 식사와의 관계가 있으면 복용방법으로 변환
                        if not 복용방법_값:
                            식사와의관계 = entity.get("식사와의 관계", "")
                            if 식사와의관계:
                                # 원문에서 "30분" 같은 정보도 추출
                                method_with_time = re.search(r"식후\s*(\d+)\s*분", user_input)
                                if method_with_time:
                                    복용방법_값 = f"식후 {method_with_time.group(1)}분"
                                else:
                                    복용방법_값 = 식사와의관계
                        
                        # ✅ 복용기간 추출 (사용자 입력에서)
                        if not 복용기간_값:
                            # "일주일치", "일주일 동안" 등 복용기간 패턴 추출
                            period_patterns = [
                                r"복용\s*기간.*?일주일",
                                r"복용\s*기간.*?(\d+)\s*일(?:\s*(?:이야|이에요|입니다|동안|치))?",
                                r"복용\s*기간.*?(\d+)\s*주(?:\s*(?:이야|이에요|입니다|동안|치))?",
                                r"복용\s*기간.*?(\d+)\s*개월(?:\s*(?:이야|이에요|입니다|동안|치))?",
                                r"기간.*?일주일치",
                                r"일주일치",
                                r"일주일\s*동안",
                            ]
                            for pattern in period_patterns:
                                period_match = re.search(pattern, user_input)
                                if period_match:
                                    if "일주일" in pattern:
                                        복용기간_값 = "7일"
                                        break
                                    elif period_match.lastindex and period_match.group(1):
                                        unit = "일"
                                        if "주" in pattern:
                                            unit = "주"
                                        elif "개월" in pattern:
                                            unit = "개월"
                                        복용기간_값 = f"{period_match.group(1)}{unit}"
                                        break
                        
                        # 약 이름이 없으면 입력에서 직접 추출 시도
                        if not drug_name:
                            # "~약" 패턴 우선 추출 (비염약, 혈압약 등)
                            med_pattern = re.search(r"([가-힣A-Za-z]+약)", user_input)
                            if med_pattern:
                                drug_name = med_pattern.group(1).strip()
                                print(f"[DEBUG] 약 이름 추출 (fallback, ~약 패턴): {drug_name}")
                            else:
                                # "약명 + 약" 또는 "약명을/를 먹어" 패턴 추출
                                med_pattern = re.search(r"([가-힣A-Za-z]+)\s*(?:약|을|를)", entity.get("정보", "") or user_input)
                                if med_pattern:
                                    drug_name = med_pattern.group(1).strip()
                                    print(f"[DEBUG] 약 이름 추출 (fallback): {drug_name}")
                        
                        if drug_name:
                            # 약 이름이 있으면 저장
                            # ✅ 저장 전에 중복 감지하여 응답 메시지 제어
                            user_name = memory_instance.user_names.get(session_id or "default")
                            if user_name and user_name != "사용자":
                                # ✅ 중복 감지: 기존 엑셀 데이터 확인
                                sheet_name = memory_instance.excel_manager._get_sheet_name("약")
                                existing_df = memory_instance.excel_manager.safe_load_sheet(user_name, sheet_name)
                                
                                # 정규화된 비교 기준 (공백 제거, None → 빈 문자열)
                                약명_정규화 = str(drug_name).strip()
                                시간_정규화 = str(dose_time).strip() if dose_time else ""
                                복용방법_정규화 = str(복용방법_값).strip() if 복용방법_값 else ""
                                복용기간_정규화 = str(복용기간_값).strip() if 복용기간_값 else ""
                                
                                # 중복 확인: 동일한 약명+시간+복용방법+복용기간 존재 여부
                                is_duplicate = False
                                if not existing_df.empty and all(col in existing_df.columns for col in ["약이름", "시간", "복용방법", "복용기간"]):
                                    # 기존 데이터도 정규화하여 비교
                                    existing_df_normalized = existing_df.copy()
                                    for col in ["약이름", "시간", "복용방법", "복용기간"]:
                                        existing_df_normalized[col] = existing_df_normalized[col].fillna("").astype(str).str.strip()
                                    
                                    # 중복 확인
                                    duplicate_mask = (
                                        (existing_df_normalized["약이름"] == 약명_정규화) &
                                        (existing_df_normalized["시간"] == 시간_정규화) &
                                        (existing_df_normalized["복용방법"] == 복용방법_정규화) &
                                        (existing_df_normalized["복용기간"] == 복용기간_정규화)
                                    )
                                    is_duplicate = duplicate_mask.any()
                                
                                if is_duplicate:
                                    # 중복이면 저장하지 않고 응답만 변경
                                    logger.debug(f"[DUPLICATE] 약 복용 정보 중복 감지: {약명_정규화}, {시간_정규화}, {복용방법_정규화}, {복용기간_정규화}")
                                    medicine_saved = True  # 플래그는 설정하여 중복 저장 방지
                                    results.append(f"'{drug_name}' 복용 정보는 이미 기록되어 있어요.")
                                else:
                                    # 중복이 아니면 저장
                                    memory_instance.excel_manager.save_entity_data(user_name, "약", {
                                        "약이름": drug_name,
                                        "용량": 용량_값,
                                        "단위": 단위_값,
                                        "시간": dose_time,
                                        "복용방법": 복용방법_값,
                                        "복용기간": 복용기간_값
                                    })
                                    medicine_saved = True
                                    results.append(f"'{drug_name}' 복용 정보를 저장했어요.")
                            else:
                                results.append(f"'{drug_name}' 복용 정보를 저장했어요.")
                        elif dose_time:
                            # 약 이름은 없지만 시간대가 있으면 저장 (약 이름은 추후 보완 가능)
                            if not medicine_saved:
                                user_name = memory_instance.user_names.get(session_id or "default")
                                if user_name and user_name != "사용자":
                                    memory_instance.excel_manager.save_entity_data(user_name, "약", {
                                        "약이름": "알 수 없음",
                                        "용량": 용량_값,
                                        "단위": 단위_값,
                                        "시간": dose_time,
                                        "복용방법": 복용방법_값,
                                        "복용기간": 복용기간_값
                                    })
                                    results.append("복용 정보를 기록해두었어요.")
                            else:
                                results.append("복용 정보를 기록해두었어요.")
                        else:
                            # 약 이름도 시간대도 없으면 저장하지 않음
                            print(f"[DEBUG] 약 엔티티 저장 건너뜀: 약 이름과 시간대 모두 없음")
                            results.append("복용 정보를 기록해두었어요.")
                    else:
                        # 다른 엔티티는 기존 방식 유지
                        memory_instance._add_to_vstore(
                            entity_key=entity_key,
                            value=entity,
                            metadata={"session_id": session_id, "type": "entity"},
                            user_input=user_input
                        )
            
            # 엔티티 저장 후 물리 명령 체크
            final_msg = "\n".join(results) if results else "말씀하신 내용을 기억해뒀습니다!"
            
            # ✅ LLM fallback 재질문: 약 정보가 누락된 경우
            # 일정, 식사, 물건 등 다른 엔티티가 저장된 경우에는 약 재질문하지 않음
            다른_엔티티_저장됨 = bool(entities.get("user.일정") or entities.get("user.식사") or entities.get("user.물건") or entities.get("user.사용자"))
            
            # LLM이 약 정보를 못 찾았고, rule-based도 실패했으며, 사용자가 약 관련 키워드를 사용한 경우
            약_키워드_존재 = any(k in user_input for k in ["약", "복용", "복약", "비타민", "영양제", "약물"])
            약_정보_저장_요청 = any(k in user_input for k in ["저장", "기록", "기억", "알려줘"])
            
            # 다른 엔티티가 저장되지 않았고, 약 키워드가 있고, 약 정보 저장 요청이 있고, 약 엔티티가 없는 경우에만 재질문
            if not 다른_엔티티_저장됨 and 약_키워드_존재 and 약_정보_저장_요청 and not entities.get("user.약"):
                logger.debug("[LLM REASK] 복약정보 누락 → 보강 질문 실행")
                return "복약 정보를 더 명확히 알려주세요. 약 이름, 복용 시간, 복용 방법(식후 30분 등), 복용 기간을 알려주시면 기록할게요."
            
            # 물리 명령이 있으면 바로 처리
            # ✅ 엔티티가 이미 저장되었는지 확인 (중복 저장 방지)
            entity_already_saved = bool(entities and isinstance(entities, dict) and entities.get("user.물건"))
            if any(keyword in user_input for keyword in ["가져", "갖다", "와", "가지고 와", "꺼내", "정리", "열어"]):
                from life_assist_dm.life_assist_dm.support_chains import handle_physical_task
                print("[CHAIN] 복합 명령 감지됨 → 물리 행동 연결 수행")
                result = handle_physical_task(user_input, memory_instance, session_id, entity_already_saved=entity_already_saved)
                return result
            
            return final_msg

        # 4️⃣ LCEL 메모리 (방금 대화라도 우선 참고)
        mem_vars = memory_instance.conversation_memory.load_memory_variables({})
        history = mem_vars.get("history", "")
        if history:
            # 통합 맥락 구성
            context = memory_instance._build_context_for_llm(user_input, session_id)
            prompt = f"""{context}대화 맥락:
                    {history}

                    사용자 입력: {user_input}

                    위 맥락과 저장된 정보를 바탕으로 간단하고 자연스럽게 답변하세요.
                    """
            response = memory_instance.llm.invoke(prompt)
            return response.content.strip()

        
        # cognitive 처리 완료 후, 물리 명령이 같이 포함된 경우 이어서 실행
        if any(keyword in user_input for keyword in ["가져", "갖다", "와", "가지고 와", "꺼내", "정리", "열어"]):
            from life_assist_dm.life_assist_dm.support_chains import handle_physical_task
            print("[CHAIN] 복합 명령 감지됨 → 물리 행동 연결 수행")
            result = handle_physical_task(user_input, memory_instance, session_id)
            return result
        
        return "말씀하신 내용을 기억해둘게요!"

    except Exception as e:
        # Broken pipe 등의 오류는 무시하고 조용히 처리
        import traceback
        print(f"[ERROR] cognitive 처리 중 오류 발생: {e}")
        traceback.print_exc()
        # 오류가 나도 물리 명령이 있으면 계속 진행
        if any(keyword in user_input for keyword in ["가져", "갖다", "와", "가지고 와", "꺼내", "정리", "열어"]):
            from life_assist_dm.life_assist_dm.support_chains import handle_physical_task
            print("[CHAIN] cognitive 오류 후 물리 행동 처리")
            try:
                result = handle_physical_task(user_input, memory_instance, session_id)
                return result
            except Exception as e:
                import traceback
                traceback.print_exc()
                # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
                return "처리 중 오류가 발생했어요. 다시 시도해 주세요."
        # 물리 명령이 없으면 조용히 실패
        return ""


# ===================== 공통 조회 유틸 =====================
def _query_with_fallback(user_input: str, memory_instance, session_id: str, 
                          sheet_name: str, 
                          primary_search_func: callable,
                          query_type: str = "일반",
                          lcel_prompt_template: str = None) -> Optional[str]:
    """
    통합 쿼리 검색 함수: 모든 쿼리 타입에 공통으로 적용
    
    검색 순서:
    1. 올바른 시트에서 찾기 (primary_search_func 호출)
    2. 없으면 대화 기록 시트에서 찾기
    3. 없으면 LCEL 참고하기
    
    Args:
        user_input: 사용자 입력
        memory_instance: 메모리 인스턴스
        session_id: 세션 ID
        sheet_name: 검색할 엑셀 시트 이름 (예: "복약정보", "일정", "물건위치")
        primary_search_func: 1순위 검색 함수 (시트 데이터를 받아서 결과 반환)
        query_type: 쿼리 타입 ("약", "일정", "물건", "가족", "식사", "감정" 등)
        lcel_prompt_template: LCEL 검색용 프롬프트 템플릿 (None이면 기본 템플릿 사용)
    
    Returns:
        검색 결과 문자열 또는 None (없으면)
    """
    try:
        user_name = memory_instance.user_names.get(session_id or "default")
        if not user_name:
            return None
        
        excel = memory_instance.excel_manager
        
        # ✅ 1순위: 올바른 시트에서 찾기
        try:
            df_sheet = excel.safe_load_sheet(user_name, sheet_name)
            if df_sheet is not None and not df_sheet.empty:
                result = primary_search_func(df_sheet, user_input)
                if result:
                    logger.debug(f"[QUERY] {query_type} 조회 성공 (시트: {sheet_name})")
                    return result
        except Exception as e:
            logger.debug(f"[QUERY] {query_type} 시트 검색 실패: {e}")
        
        # ✅ 2순위: 대화 기록 시트에서 찾기
        try:
            df_conversation = excel.safe_load_sheet(user_name, "대화기록")
            if df_conversation is not None and not df_conversation.empty:
                # 최근 대화 기록에서 관련 정보 추출
                conversation_texts = []
                for _, row in df_conversation.tail(10).iterrows():  # 최근 10개만
                    summary = str(row.get("대화요약", "")).strip()
                    if summary and summary.lower() not in ("nan", "none", ""):
                        conversation_texts.append(summary)
                
                if conversation_texts:
                    # LLM으로 대화 기록에서 관련 정보 추출
                    conversation_context = "\n".join([f"- {text}" for text in conversation_texts])
                    
                    # 기본 프롬프트 템플릿이 없으면 생성
                    if not lcel_prompt_template:
                        if query_type == "약":
                            lcel_prompt_template = "다음 대화 기록에서 사용자가 복용한다고 말한 약 이름과 시간대만 간단히 한 문장으로 한국어로 답하세요.\n없으면 '기록된 복약 정보가 없어요.'라고 답하세요."
                        elif query_type == "일정":
                            lcel_prompt_template = "다음 대화 기록에서 일정이나 예약(병원/치과/미용실 등) 언급이 있는지 확인하고, 있으면 한 문장으로 답하세요.\n없으면 '기록된 일정이 없어요.'라고 답하세요."
                        elif query_type == "물건":
                            lcel_prompt_template = "다음 대화 기록에서 특정 물건의 위치를 단정적으로 말한 문장이 있는지 확인하고, 있으면 한 문장으로 답하세요.\n없으면 '해당 물건의 위치는 아직 기록되어 있지 않아요.'라고 답하세요."
                        elif query_type == "가족":
                            lcel_prompt_template = "다음 대화 기록에서 가족(동생/형/누나/언니/엄마/아빠) 이름/관계를 찾을 수 있다면 한 문장으로 한국어로 답하세요.\n없으면 '기록된 가족 정보가 없어요.'라고 답하세요."
                        elif query_type == "식사":
                            lcel_prompt_template = "다음 대화 기록에서 식사(아침/점심/저녁) 정보를 찾을 수 있다면 한 문장으로 한국어로 답하세요.\n없으면 '기록된 식사 정보가 없어요.'라고 답하세요."
                        elif query_type == "감정":
                            lcel_prompt_template = "다음 대화 기록에서 사용자가 표현한 감정 상태(기분, 감정, 힘듦, 피로, 행복 등)를 찾아서 요약해주세요. 최근 감정 상태를 한 문장으로 한국어로 답하세요.\n예: '최근에 힘들고 피곤한 상태를 느끼셨다고 기록되어 있어요.' 또는 '학교에서 친구가 괴롭혀서 힘들다고 하셨어요.'\n없으면 '기록된 감정 정보가 없어요.'라고 답하세요."
                        else:
                            lcel_prompt_template = f"다음 대화 기록에서 {query_type} 관련 정보를 찾을 수 있다면 한 문장으로 한국어로 답하세요.\n없으면 '기록된 정보가 없어요.'라고 답하세요."
                    
                    prompt = f"{lcel_prompt_template}\n\n대화 기록:\n{conversation_context}\n\n질문: {user_input}"
                    resp = memory_instance.llm.invoke(prompt)
                    text = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
                    if text and "없어요" not in text and "없어" not in text and "기록" not in text:
                        logger.debug(f"[QUERY] {query_type} 조회 성공 (대화 기록 시트)")
                        return text
        except Exception as e:
            logger.debug(f"[QUERY] {query_type} 대화 기록 검색 실패: {e}")
        
        # ✅ 3순위: LCEL 참고하기
        try:
            mem_vars = memory_instance.conversation_memory.load_memory_variables({})
            history = mem_vars.get("history", "") or ""
            if history:
                if not lcel_prompt_template:
                    if query_type == "약":
                        lcel_prompt_template = "다음 최근 대화 기록에서 사용자가 복용한다고 말한 약 이름과 시간대만 간단히 한 문장으로 한국어로 답하세요.\n없으면 '기록된 복약 정보가 없어요.'라고 답하세요."
                    elif query_type == "일정":
                        lcel_prompt_template = "최근 대화 기록에서 일정이나 예약(병원/치과/미용실 등) 언급이 있는지 확인하고, 있으면 한 문장으로 답하세요.\n없으면 '기록된 일정이 없어요.'라고 답하세요."
                    elif query_type == "물건":
                        lcel_prompt_template = "최근 대화 기록에서 특정 물건의 위치를 단정적으로 말한 문장이 있는지 확인하고, 있으면 한 문장으로 답하세요.\n없으면 '해당 물건의 위치는 아직 기록되어 있지 않아요.'라고 답하세요."
                    elif query_type == "가족":
                        lcel_prompt_template = "다음 최근 대화 기록에서 가족(동생/형/누나/언니/엄마/아빠) 이름/관계를 찾을 수 있다면 한 문장으로 한국어로 답하세요.\n없으면 '기록된 가족 정보가 없어요.'라고 답하세요."
                    elif query_type == "식사":
                        lcel_prompt_template = "최근 대화 기록에서 식사(아침/점심/저녁) 정보를 찾을 수 있다면 한 문장으로 한국어로 답하세요.\n없으면 '기록된 식사 정보가 없어요.'라고 답하세요."
                    elif query_type == "감정":
                        lcel_prompt_template = "최근 대화 기록에서 사용자가 표현한 감정 상태(기분, 감정, 힘듦, 피로, 행복 등)를 찾아서 요약해주세요. 최근 감정 상태를 한 문장으로 한국어로 답하세요.\n예: '최근에 힘들고 피곤한 상태를 느끼셨다고 기록되어 있어요.' 또는 '학교에서 친구가 괴롭혀서 힘들다고 하셨어요.'\n없으면 '기록된 감정 정보가 없어요.'라고 답하세요."
                    else:
                        lcel_prompt_template = f"최근 대화 기록에서 {query_type} 관련 정보를 찾을 수 있다면 한 문장으로 한국어로 답하세요.\n없으면 '기록된 정보가 없어요.'라고 답하세요."
                
                prompt = f"{lcel_prompt_template}\n\n최근 대화 기록:\n{history}\n\n질문: {user_input}"
                resp = memory_instance.llm.invoke(prompt)
                text = resp.content.strip() if hasattr(resp, 'content') else str(resp).strip()
                if text and "없어요" not in text and "없어" not in text and "기록" not in text:
                    logger.debug(f"[QUERY] {query_type} 조회 성공 (LCEL)")
                    return text
        except Exception as e:
            logger.debug(f"[QUERY] {query_type} LCEL 검색 실패: {e}")
        
        return None
    except Exception as e:
        logger.debug(f"[QUERY] {query_type} 통합 검색 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

def _find_saved_location(memory_instance, session_id: Optional[str], target: Optional[str]) -> Optional[str]:
    """세션 캐시 → 엑셀 순으로 물건 위치를 조회한다."""
    try:
        if not target:
            return None
        # 1) 세션 캐시
        if hasattr(memory_instance, 'excel_cache') and session_id:
            items = memory_instance.excel_cache.get(session_id, {}).get("물건", [])
            for it in reversed(items):
                if it.get("이름") == target and it.get("위치"):
                    return it.get("위치")
        # 2) 엑셀
        user_name = None
        try:
            user_name = memory_instance.user_names.get(session_id or "default")
        except Exception:
            user_name = None
        if user_name:
            excel = memory_instance.excel_manager
            df = excel.load_sheet_data(user_name, "물건위치")
            if df is not None and not df.empty:
                for _, row in df.iloc[::-1].iterrows():
                    if str(row.get("물건이름", "")) == target or str(row.get("이름", "")) == target:
                        loc = row.get("위치", "")
                        if str(loc).strip() != "":
                            return str(loc).strip()
    except Exception:
        return None
    return None

        
