# support_chains.py
from __future__ import annotations
import os, csv, json, re, random, logging, traceback
from datetime import datetime
from typing import Optional, Dict, List, Any
import pandas as pd

# ✅ 디버그 로깅 설정
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

LOCATION_TOKENS = r"(침대\s*(옆|위|밑)|침대\s*머리맡|소파\s*(옆|위|밑)|책상\s*(옆|위|밑)|거실|주방|현관|문\s*앞|식탁|테이블|베란다|냉장고|책꽂이|서랍|옷장)"

def _preprocess_for_parsing(text: str) -> str:
    """두 문장 이상일 때 첫 문장만 해석. 연속 공백 축소."""
    t = (text or "").strip()
    for sep in [".", "?", "!", "\n"]:
        if sep in t:
            t = t.split(sep)[0].strip()
            break
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
    """텍스트 정규화 - 중복 공백 제거, 조사 띄어쓰기 보정"""
    if not text:
        return ""
    # 중복 공백 제거, 조사의 잘못된 띄어쓰기 보정
    t = re.sub(r"\s+", " ", text).strip()
    t = t.replace(" 의자 에", " 의자에").replace(" 방 에", " 방에")
    return t

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

# 견고한 위치/대상 추출 (에/에 있는/옆/밑/뒤/안/속/근처 패턴 포함)
LOC_PAT = r"(?:내\s*방|안방|거실|주방|부엌|현관|화장실|베란다|현관문|서재|침실|현관문|베란다문|현관문앞|현관\s*앞|문\s*앞|문앞|방|부엌|정수기|냉장고|세탁기|에어컨|식탁|책상|의자|소파|침대|프린터|신발장|서랍|책꽂이|선반|장바구니|바구니|빨래대|빨래\s*건조대|건조대|테이블|식탁테이블|식탁대|선반대)"
POS_PAT = r"(?:위|옆|밑|아래|뒤|뒤쪽|안|안쪽|속|근처|머리맡|쪽|앞|바로앞)"
# 간소화된 TARGET_PAT (LLM이 주력이므로 최소한만)
TARGET_PAT = r"(?:책|펜|컵|핸드폰|리모컨|충전기|노트북|지갑|가방|우산|키|안경|휴지|수건|담요|베개|이불|쓰레기|신문|사과|과일|음료|물|커피|우유|빵|약|치실|칫솔|마스크|장갑|신발|앨범|이어폰|양말|볼펜|물컵|옷|비타민|모자|타월|샴푸|비누|화장품|시계|반지|귀걸이|목걸이|팔찌|선글라스|속옷|티셔츠|바지|치마|원피스|자켓|코트|운동화|구두|부츠|샌들|슬리퍼|실내화|장화|캡|헬멧|목도리|인형|화분|빨래|세탁물|세제|린스|컨디셔너|택배|포장|소포|물건|거|것)"

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
    
    # --- 2) 간소화된 Rule 기반 fallback (최소한만) ---
    if not target:
        # 가장 기본적인 패턴들만 유지
        patterns = [
            # 1) "[LOC] [POS]? 에 [TARGET]" 패턴
            rf"({LOC_PAT})(?:\s*({POS_PAT}))?\s*에\s*(?:있는\s*)?({TARGET_PAT})",
            # 2) "[LOC] [POS]? 에서 [TARGET] 가져와" 패턴  
            rf"({LOC_PAT})(?:\s*({POS_PAT}))?\s*에서\s*({TARGET_PAT})\s*(?:가져와|갖다줘|찾아줘)",
            # 3) "[TARGET] 가져와" 패턴
            rf"({TARGET_PAT})\s*(?:가져와|갖다줘|찾아줘|정리해|치워줘)",
            # 4) 택배 특수 패턴
            r"(택배|포장|소포).*?(문\s*앞|현관).*?(?:가져와|갖다줘|찾아줘)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                if len(match.groups()) >= 3:  # location, position, target
                    location, position, target = match.group(1), (match.group(2) or "").strip() or None, match.group(3)
                elif len(match.groups()) == 2:  # location, target 또는 target, location
                    if "가져와" in pattern:  # target, location 패턴
                        target, location = match.group(1), match.group(2)
                    else:  # location, target 패턴
                        location, target = match.group(1), match.group(2)
                else:  # target only
                    target = match.group(1)
                break

    # 위치 후처리: '화장실 선반' + '위' → '화장실 선반 위' 등 자연스럽게 결합
    if location and position:
        location = f"{location} {position}"
    
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

def build_emotional_reply(text: str, llm=None) -> str:
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
    if re.search(r"안녕", t):
        return "안녕하세요! 😊"
    if re.search(r"(고마워|수고|좋아)", t):
        return "네! 언제든지요 🙂"

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
            return llm.invoke(prompt)
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

    if location:
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


def handle_physical_task(user_input: str, memory_instance, session_id: str) -> dict:
    """물리적 작업 처리 (찾기, 가져오기, 정리하기)"""
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
        
        # '주워/집어/꺼내' 즉시 deliver (위치 없어도 OK)
        if re.search(r"(주워|집어|꺼내)\w*", text):
            target, location = _extract_robust(text, memory_instance.llm if hasattr(memory_instance, 'llm') else None)
            if not target: 
                target = "물건"
            msg = f"{target}{josa(target, ('을','를'))} 가져오겠습니다."
            robot_cmd = to_task_command_en("deliver", target, location, memory_instance)
            return {"success": True, "message": msg, "robot_command": robot_cmd}
        
        # 버리기(슬롯 채우기) 처리
        if re.search(r"(버려|처리해)", text):
            target, location = _extract_robust(text, memory_instance.llm if hasattr(memory_instance, 'llm') else None)
            if not target: 
                target = "쓰레기"
            bin_loc = memory_instance.get_location("쓰레기통") if hasattr(memory_instance, 'get_location') else None
            if not bin_loc:
                return {"success": True, "message": "어디에 버릴까요? (쓰레기통 위치를 알려주시면 기억해둘게요)", "robot_command": None}
            # dispose 액션이 있다면 여기서 명령 생성
            return {"success": True, "message": f"{target}을(를) {bin_loc}에 버리겠습니다.", "robot_command": None}
        
        # '제자리'가 포함되면 organize 강제 (하지만 위치는 추출하지 않음)
        if re.search(r"제자리(에)?", text):
            action = "organize"
            # 제자리는 실제 위치가 아니므로 location을 None으로 설정
            location = None
        
        # 1. 청소 가드 (가장 먼저) - 단독 청소 요청만
        if re.search(r"(청소|닦|먼지|때|깨끗|쓸)\s*(해|해줘|해주세요)", text) and not re.search(r"(가져와|갖다줘|가져다줘|가지고\s*와)", text):
            return {"success": False, "message": ERROR_UNSUPPORTED, "robot_command": None}
        
        # 2. 로봇 제어류 가드 (켜/꺼/밝기/온도/볼륨 등) - "꺼내" 제외, '리모컨' 제거, 잠금 추가
        # 제어 동사와 디바이스 토큰이 함께 있을 때만 제어로 판단 (오탐 완화)
        if re.search(r"(켜|끄|꺼|열|닫|높이|낮추|올리|내리)", text) and re.search(r"(전등|불|조명|커튼|블라인드|에어컨|보일러|난방|히터|선풍기|TV|티비|창문)", text):
            return {"success": False, "message": ERROR_UNSUPPORTED, "robot_command": None}
        
        # 1. 액션 타입 추정 (LLM 전달)
        action = _extract_action_type(text, memory_instance.llm if hasattr(memory_instance, 'llm') else None)
        logger.debug(f"[PHYSICAL] Input={text}, action={action}")
        
        # 지원하지 않는 작업 처리
        if action == "unsupported":
            logger.debug(f"[PHYSICAL] Unsupported action: {text}")
            return {"success": False, "message": ERROR_UNSUPPORTED, "robot_command": None}
        
        # 정리/정돈/치워 → 의미 구분 후 처리 (CMD_VERBS와 중복 제거)
        # 단, "가져와"가 포함된 경우는 deliver로 우선 처리
        if re.search(r"(정리|정돈|치워)\s*(좀|조금|제발|주세요|줘)?", text) and not re.search(r"(가져와|갖다줘|가져다줘|가지고\s*와)", text):
            # 청소 의미 키워드가 있으면 미지원 처리
            if re.search(r"(청소|닦|먼지|때|깨끗|쓸|빨아|세척|소독|살균)", text):
                return {"success": False,
                        "message": "청소 작업은 아직 지원하지 않아요. 다른 작업을 요청해주시겠어요?",
                        "robot_command": None}
            
            # 의미 재질문 (타깃 추출 전에 의미 확인)
            msg = "제자리에 가져다 두라는 뜻인가요, 아니면 공간을 청소하라는 의미이신가요? (청소는 미지원)"
            memory_instance.pending_question[session_id] = {
                "type": "organize_meaning_clarification",
                "original_text": text,
                "question": msg
            }
            memory_instance.current_question[session_id] = msg
            return {"success": True, "message": msg, "robot_command": None}
        
        # 2. 물건명과 위치 추출 (LLM 우선)
        target, location = _extract_robust(text, memory_instance.llm if hasattr(memory_instance, 'llm') else None)
        # 2-1. 추출 실패 시 지시어 기반 보강 (원문 기준)
        if not target and re.search(r"(그거|그것)", original_text) and state.get("last_target"):
            target = state["last_target"]
        if not location and re.search(r"(거기)", original_text) and state.get("last_location"):
            location = state["last_location"]
        logger.debug(f"[PHYSICAL] Extracted - target={target}, location={location}")
        
        if not target:
            logger.warning(f"[PHYSICAL] No target extracted from: {text}")
            return {"success": False, "message": "죄송해요, 어떤 물건을 말씀하시는지 모르겠어요.", "robot_command": None}
        
        # 타깃 추출 후 청소 관련 체크
        if re.search(r"(청소|닦|먼지|때|깨끗|쓸|빨아|세척)", text):
            return {"success": False,
                    "message": "청소 작업은 아직 지원하지 않아요. 다른 작업을 요청해주시겠어요?",
                    "robot_command": None}
        
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
                # 명시적 위치가 없으면 저장된 위치 확인
                saved_location = None
                try:
                    saved_location = memory_instance.get_location(target) if hasattr(memory_instance, 'get_location') else None
                except Exception:
                    pass
                
                if saved_location:
                    # a1. 위치가 저장되어 있음
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
            if explicit_location:
                # 위치를 바로 알 수 있음 - 엔티티 저장 후 deliver
                # 물건 위치 정보를 VectorStore에 저장
                try:
                    save_result = memory_instance.save_entity_to_vectorstore(
                        entity_type="물건",
                        data={"이름": target, "위치": location},
                        session_id=session_id
                    )
                    print(f"[DEBUG] 물건 위치 저장: {target} -> {location}")
                except Exception as e:
                    print(f"[WARN] 물건 위치 저장 실패: {e}")
                
                msg = f"{target}을(를) {location}에서 가져오겠습니다."
                robot_cmd = to_task_command_en("deliver", target, location, memory_instance)
                return {"success": True, "message": msg, "robot_command": robot_cmd}
            else:
                # 위치를 모름 - 저장된 위치 확인
                saved_location = None
                try:
                    saved_location = memory_instance.get_location(target) if hasattr(memory_instance, 'get_location') else None
                except Exception:
                    pass
                
                if saved_location:
                    # 저장된 위치가 있음 - 바로 deliver
                    msg = f"{target}을(를) {saved_location}에서 가져오겠습니다."
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
                # 명시적 위치가 없으면 저장된 위치 확인
                saved_location = None
                try:
                    saved_location = memory_instance.get_location(target) if hasattr(memory_instance, 'get_location') else None
                except Exception:
                    pass
                
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
        
        # 중복된 액션 처리 로직 제거됨 - 위의 액션별 처리로 통합
        
        # 3. 사용자 응답 생성 (fallback)
        response = _generate_physical_response(action, target, location, user_input, memory_instance)
        
        # 4. 로봇 명령 JSON 생성 (디버깅용)
        robot_cmd = to_task_command_en(action, target, location, memory_instance) if action in ("find","deliver","organize") else None
        if robot_cmd:
            print(f"[DEBUG] 로봇 명령 전달: {robot_cmd}")
        
        # ✅ 항상 dict 반환 + 세션 상태 갱신
        state["last_target"] = target or state.get("last_target")
        if location:
            state["last_location"] = location
        state["last_action"] = action
        return {
            "success": True,
            "message": response,
            "robot_command": robot_cmd
        }
            
    except Exception as e:
        logger.exception("physical_task_failed: %s\n%s", user_input, traceback.format_exc())
        return {
            "success": False,
            "message": f"파싱 오류: {e.__class__.__name__}",
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
            
            # 저장된 위치 확인
            saved_location = None
            try:
                saved_location = memory_instance.get_location(target) if hasattr(memory_instance, 'get_location') else None
            except Exception:
                pass
            
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
            return {
                "success": False,
                "message": "알 수 없는 질문 유형입니다.",
                "robot_command": None
            }
            
    except Exception as e:
        print(f"[ERROR] handle_pending_answer 실패: {e}")
        return {
            "success": False,
            "message": "죄송해요, 답변 처리 중 오류가 발생했어요.",
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
            saved_loc = None
            try:
                saved_loc = memory_instance.get_location(target) if hasattr(memory_instance, 'get_location') else None
            except Exception:
                pass
            
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
                    print(f"[DEBUG] LLM 분류: '{text}' -> {action}")
                    return action
                    
        except Exception as e:
            print(f"[WARN] LLM 분류 실패: {e}")

    # --- 2) 간단한 Rule 가드 (LLM 실패 시) ---
    # 스마트홈 제어 가드
    home_ctrl = r"(불|전등|조명|커튼|블라인드|에어컨|히터|난방|보일러|선풍기|환기|창문|문|커버|TV|티비|볼륨|밝기|온도)\s*(켜|꺼|열|닫|높|낮|올리|내리)"
    if re.search(home_ctrl, text):
        return "unsupported"

    # 청소 가드 (단독 청소 요청만)
    if re.search(r"(청소|닦아|깨끗|먼지|쓸)\s*(해|해줘|해주세요)", text) and not re.search(r"(가져와|갖다줘|가지고\s*와)", text):
        return "clean"

    # 기본 패턴 매칭 (fallback)
    if re.search(r"(찾아줘|찾아와|찾아봐|어디있어|위치)", text):
        return "find"
    if re.search(r"(가져와|갖다줘|가져다줘|가지고\s*와|꺼내와|주워줘)", text):
        return "deliver"
    # 정리 패턴 (CMD_VERBS와 중복 제거)
    # if re.search(r"(정리|정돈|치워|제자리|가져다\s*놔|갖다\s*놔|다시\s*가져다\s*놔)", text):
    #     return "organize"

    return "unsupported"


# ---------- [NEW] 간단 영어 변환 fallback ----------
def _to_english(word: str | None) -> str | None:
    """간단한 영어 변환 fallback"""
    if not word:
        return None
    # 사전이 있으면 우선 사용, 없으면 원문 유지
    try:
        return TARGET_MAP.get(word, LOCATION_MAP.get(word, word))  # 통합된 맵 사용
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
        prompt = f"""
한국어 단어를 영어로 단어 하나만 번역하세요.
문장, 예문, 설명, 따옴표 없이 **단어 하나만** 출력하세요.

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
    yes_keywords = ["네", "응", "그래", "맞아", "좋아", "해줘", "해주세요", "가져다", "정리해", "가져와", "부탁해"]
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
    2. VectorStore 검색
    3. SQLite 요약본 확인
    4. fallback
    """
    try:
        # 1️⃣ LCEL 메모리 (최근 대화 맥락 우선)
        mem_vars = memory_instance.conversation_memory.load_memory_variables({})
        history = mem_vars.get("history", "")
        
        # "방금", "지금", "오늘" 같은 지시어가 있으면 LCEL buffer 우선 조회
        if any(word in user_input for word in ["방금", "지금", "오늘", "최근", "아까"]):
            if history:
                # 통합 맥락 구성
                context = memory_instance._build_context_for_llm(user_input, session_id)
                prompt = f"""{context}방금 대화 맥락:
                        {history}

                        사용자 질문: {user_input}

                        위 대화 맥락과 저장된 정보를 바탕으로 답변하세요. 새로운 사실을 지어내면 안 됩니다.
                        """
                response = memory_instance.llm.invoke(prompt)
                return response.content.strip()
        
        # 일반적인 LCEL history 조회
        if history:
            prompt = f"""대화 맥락:
                    {history}

                    사용자 질문: {user_input}

                    반드시 사용자 발화를 토대로 저장된 맥락만 기반으로 답변하세요. 새로운 내용을 가짜로 지어내면 절대로 안 됩니다.
                    """
            response = memory_instance.llm.invoke(prompt)
            return response.content.strip()

        # 2️⃣ VectorStore 검색 (구조화된 엔티티 조회)
        docs = memory_instance.vectorstore.similarity_search(user_input, k=5)
        
        # 물건 위치 조회
        if any(word in user_input for word in ["어디", "위치", "있어"]):
            for d in docs:
                try:
                    import json
                    content = json.loads(d.page_content)
                    if content.get("type") == "물건":
                        name = content.get("이름", "")
                        location = content.get("위치", "")
                        if name and location:
                            return f"{name}{josa(name, ('은','는'))} {location}에 있어요."
                except Exception:
                    continue
            return "해당 물건의 위치는 아직 기록되어 있지 않아요."
        
        # 감정 기록 조회 (정서 타입으로 통일)
        if any(word in user_input for word in ["기분", "감정", "최근", "피곤", "힘들", "좋아", "힘들다고", "피곤하다고"]):
            emotions = []
            for d in docs:
                try:
                    import json
                    content = json.loads(d.page_content)
                    if content.get("type") == "정서":
                        emotion = content.get("감정", "")
                        date = content.get("날짜", "")
                        if emotion:
                            emotions.append(f"{emotion}({date})" if date else emotion)
                except Exception:
                    continue
            if emotions:
                return f"최근에 말씀하신 감정은 {', '.join(emotions)} 등이 있어요."
            return "아직 저장된 감정 기록이 없어요."
        
        # 일반적인 VectorStore 검색
        if docs:
            context = "\n".join([d.page_content for d in docs])
            prompt = f"""질문: {user_input}

저장된 정보: {context}

자연스럽게 답변하세요. 예시: "점심을 드신 뒤에는 처방 받으신 혈압약을 드셔야 해요!"
"""
            response = memory_instance.llm.invoke(prompt)
            return response.content.strip()

        # 3️⃣ SQLite 요약 확인
        summaries = memory_instance._get_recent_conversation_summary(session_id)
        if summaries:
            prompt = f"""최근 대화 요약:
                    {summaries}

                    사용자 질문: {user_input}
                    """
            response = memory_instance.llm.invoke(prompt)
            return response.content.strip()

        # 4️⃣ fallback
        return "아직 기록된 정보가 없어요. 알려주시면 기억해둘게요!"

    except Exception as e:
        return f"[ERROR] query 처리 중 오류 발생: {e}"


def handle_cognitive_task_with_lcel(user_input: str, memory_instance, session_id: str) -> str:
    """
    Cognitive 요청 처리:
    1. 중복 응답 처리 체크
    2. 요약 요청 체크 (query로 위임)
    3. 엔티티 추출 및 저장
    4. LCEL 메모리(history) 참고
    5. VectorStore 저장/조회
    6. fallback
    """
    try:
        # 1️⃣ 중복 응답 처리 체크
        if hasattr(memory_instance, 'pending_question') and memory_instance.pending_question.get(session_id):
            pending_data = memory_instance.pending_question[session_id]
            print(f"[DEBUG] 중복 응답 처리: {user_input}")
            result = memory_instance.handle_duplicate_answer(user_input, pending_data)
            
            # 응답 처리 완료 후 pending_question 제거
            if session_id in memory_instance.pending_question:
                del memory_instance.pending_question[session_id]
            
            return result["message"]
        
        # 2️⃣ 요약 요청은 query로 위임 (중복 제거)
        # if re.search(r"(요약|정리해줘|대화.*정리|지난.*요약)", user_input):
        #     return handle_query_with_lcel(user_input, memory_instance, session_id)
        
        # 2️⃣ 엔티티 추출 (Slot-filling 체크 포함)
        entities = memory_instance._pre_extract_entities(user_input, session_id)
        print(f"[DEBUG] handle_cognitive_task_with_lcel에서 추출된 엔티티: {entities}")
        
        # 2.5️⃣ Slot-filling 응답 처리
        if isinstance(entities, dict) and entities.get("success") == False and entities.get("incomplete"):
            print(f"[DEBUG] Slot-filling 필요: {entities['message']}")
            # pending_question에 저장
            memory_instance.pending_question[session_id] = entities.get("pending_data", {})
            return entities["message"]
        
        # 3️⃣ VectorStore 저장/조회 (엔티티 기반) - 먼저 처리
        print(f"[DEBUG] 엔티티 처리 시작: entities={entities}")
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
                        if name and location:
                            save_result = memory_instance.save_entity_to_vectorstore(
                                entity_type="물건",
                                data={"이름": name, "위치": location},
                                session_id=session_id
                            )
                            if save_result.get("duplicate"):
                                # 중복 발견 시 pending_question에 저장
                                memory_instance.pending_question[session_id] = save_result.get("pending_data", {})
                                return save_result["message"]
                            else:
                                results.append(f"'{name}'의 위치를 '{location}'로 저장했어요.")
                    elif entity_key == "user.건강상태":
                        # 감정 엔티티도 JSON 구조로 저장 (정서 타입으로 통일)
                        emotion = entity.get("증상", "")
                        if emotion:
                            save_result = memory_instance.save_entity_to_vectorstore(
                                entity_type="정서",
                                data={"감정": emotion, "강도": entity.get("정도", "보통")},
                                session_id=session_id
                            )
                            if save_result.get("duplicate"):
                                # 중복 발견 시 pending_question에 저장
                                memory_instance.pending_question[session_id] = save_result.get("pending_data", {})
                                return save_result["message"]
                            else:
                                results.append(f"'{emotion}' 감정을 기록했어요.")
                    elif entity_key == "user.사용자":
                        # 사용자 엔티티는 JSON 구조로 저장
                        name = entity.get("이름", "")
                        if name:
                            save_result = memory_instance.save_entity_to_vectorstore(
                                entity_type="사용자",
                                data={"이름": name, "확인됨": entity.get("확인됨", True)},
                                session_id=session_id
                            )
                            if save_result.get("duplicate"):
                                # 중복 발견 시 pending_question에 저장
                                memory_instance.pending_question[session_id] = save_result.get("pending_data", {})
                                return save_result["message"]
                            else:
                                results.append(f"'{name}'님의 이름을 저장했어요.")
                    elif entity_key == "user.일정":
                        # 일정 엔티티는 JSON 구조로 저장
                        print(f"[DEBUG] 일정 엔티티 저장 시도: {entity}")
                        title = entity.get("제목", "")
                        date = entity.get("날짜", "")
                        time = entity.get("시간", "")
                        if title:
                            print(f"[DEBUG] 일정 저장: 제목={title}, 날짜={date}, 시간={time}")
                            save_result = memory_instance.save_entity_to_vectorstore(
                                entity_type="일정",
                                data={"제목": title, "날짜": date, "시간": time},
                                session_id=session_id
                            )
                            if save_result.get("duplicate"):
                                # 중복 발견 시 pending_question에 저장
                                memory_instance.pending_question[session_id] = save_result.get("pending_data", {})
                                return save_result["message"]
                            else:
                                results.append(f"'{title}' 일정을 저장했어요.")
                        else:
                            print(f"[DEBUG] 일정 제목이 없어서 저장하지 않음")
                    else:
                        # 다른 엔티티는 기존 방식 유지
                        memory_instance._add_to_vstore(
                            entity_key=entity_key,
                            value=entity,
                            metadata={"session_id": session_id, "type": "entity"},
                            user_input=user_input
                        )
            
            return "\n".join(results) if results else "말씀하신 내용을 기억해뒀습니다!"

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

        
        return "말씀하신 내용을 기억해둘게요!"

    except Exception as e:
        return f"[ERROR] cognitive 처리 중 오류 발생: {e}"
