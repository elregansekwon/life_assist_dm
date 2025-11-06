# task_classifier.py - 하드코딩 복원
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any
import re
import json

@dataclass
class ClassificationResult:
    category: str
    confidence: float
    probabilities: Optional[Dict[str, float]] = None
    method: str = "unknown"

# ✅ 가족 관계 키워드 (정규식 패턴과 리스트 모두 제공)
FAMILY_RELATION_KEYWORDS_PATTERN = r"(동생|형|누나|언니|오빠|엄마|아빠|어머니|아버지|부모|할머니|할아버지|아들|딸|손주|손녀|며느리|사위)"
FAMILY_RELATION_KEYWORDS = ["동생", "형", "누나", "언니", "오빠", "엄마", "아빠", "어머니", "아버지", "부모", 
                            "할머니", "할아버지", "아들", "딸", "손주", "손녀", "며느리", "사위", "남편", "아내"]

# 하드코딩 패턴 복원
COGNITIVE_PATTERNS = [
    r"(예약|추가|기억|기록|저장|넣어).*해",
    r"(치과|병원|미용실|회의|미팅).*(예약|잡아|넣어)",
    r"일정.*넣어|일정.*추가|일정.*등록",
    r".*(먹었어|했다|했어|봤어|읽었어|들었어|갔어|왔어|만났어|만났다)",
    r".*(기억해|기록해|저장해|넣어줘|추가해줘)",
    r"(오늘|어제|지난주|이번주|다음주).*(먹었|봤|읽었|갔|왔|만났|했다|했어)",
    r"(점심|아침|저녁|식사).*(먹었|했어)",
    r"(영화|책|음악|게임).*(봤|읽었|들었|했어)",
    # 위치 정보 저장 패턴 추가
    r".*(에\s*있어|에\s*두었어|에\s*놨어|에\s*있고|에\s*있습니다)",
    r".*(위치|장소).*(기억|저장|알려)",
    rf"{FAMILY_RELATION_KEYWORDS_PATTERN}.*(이름|이름은|이름이).*(이야|이에요|입니다|야|다)",
    rf"(내|우리|제)\s*{FAMILY_RELATION_KEYWORDS_PATTERN}.*(이름|이름은|이름이).*([가-힣]{{2,}})",
]

PREFERENCE_PATTERNS = [
    r"좋아해|좋아하는|좋아|선호|싫어해|싫어|취향|제일 좋아|즐겨|좋아하던",
]

# ✅ 감정 단어 리스트 (support_chains.py와 공통 사용)
try:
    from life_assist_dm.life_assist_dm.support_chains import (
        EMOTION_POSITIVE_WORDS, EMOTION_NEGATIVE_WORDS, 
        EMOTION_TIRED_WORDS, EMOTION_ANXIOUS_WORDS
    )
    # 감정 단어 패턴 생성
    _all_emotion_words = EMOTION_POSITIVE_WORDS + EMOTION_NEGATIVE_WORDS + EMOTION_TIRED_WORDS + EMOTION_ANXIOUS_WORDS
    _emotion_words_pattern = "|".join([re.escape(word) for word in _all_emotion_words])
except ImportError:
    # fallback: 기본 감정 단어만
    _all_emotion_words = ["행복", "좋아", "기뻐", "슬퍼", "우울", "힘들", "속상해", "짜증", "화나", "피곤", "지쳐", "불안", "긴장"]
    _emotion_words_pattern = "|".join([re.escape(word) for word in _all_emotion_words])

EMOTIONAL_PATTERNS = [
    r"안녕|고마워|힘들|피곤|기분|행복|슬퍼|화나",
    r"(오늘|현재).*(날짜|시간|몇\s*시|며칠)",
    r"(날씨|온도).*",
    r"기분.*어때|기분.*어떤|기분.*어떠|어떤.*기분",
    rf"({_emotion_words_pattern})",  # ✅ 공통 감정 단어 패턴 추가
]

PHYSICAL_PATTERNS = [
    r"(가져와|갖다|찾아|어디\s*있|치워|꺼내|찾아와|가져다)",
    r"켜|꺼|밝기|온도|볼륨|작동|재생|멈춰|시작|정지|청소 시작|돌려|전등|불|조명|보일러|에어컨|TV|커튼|쓰레기통|비워",
    r"정리해|정돈해|치워",
]

QUERY_PATTERNS = [
    r"(알려줘|보여줘|궁금|무엇|뭐야|뭐라고)",
    r"(오늘|내일|이번주|다음주).*(일정|스케줄|약속).*알려",
    # 질문 패턴 강화
    r".*\?$",  # 문장 끝에 ?가 있는 경우
    r"(어디|언제|무엇|몇|왜|어떻게|어떤)",
    # 해결책/조언 요청 패턴 추가
    r"(어떻게|어떤).*(해결|좋을|할까|해야|하면)",
    r"(해결|좋을|할까|해야|하면).*(어떻게|어떤)",
    r"(싸움|갈등|문제|고민|어려움).*(해결|좋을|할까|해야|하면)",
    # 🔧 취향/선호 질의 패턴 추가
    r"내 취향|좋아하는.*있어|좋아한다고.*뭐|취향.*알아|좋아하는.*뭐였지",
    # 가족 관계 이름 조회 패턴 (변수 사용)
    rf"{FAMILY_RELATION_KEYWORDS_PATTERN}.*(이름|이름은|이름이).*\?",
    rf"{FAMILY_RELATION_KEYWORDS_PATTERN}.*(이름|이름은|이름이).*(뭐|무엇|알려|말해)",
    r"(이름|이름은|이름이).*\?",
    r"(이름|이름은|이름이).*(뭐|무엇|알려|말해)",
]

def _score(patterns, text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text))

def classify_hybrid(text: str) -> ClassificationResult:
    """하이브리드 분류: 하드코딩 우선 → LLM fallback"""
    
    # 🔧 취향 패턴 우선 체크 (감정보다 먼저)
    preference_score = _score(PREFERENCE_PATTERNS, text)
    # 취향 관련 질의인지 확인 (질의 패턴 또는 '알아/기억' 키워드)
    is_preference_query = preference_score > 0 and (
        _score(QUERY_PATTERNS, text) > 0 or re.search(r"알아|기억", text) is not None
    )
    
    # 1️⃣ 하드코딩 패턴 우선 체크
    cognitive_score = _score(COGNITIVE_PATTERNS, text)
    emotional_score = _score(EMOTIONAL_PATTERNS, text) if preference_score == 0 else 0  # 취향이 있으면 감정 점수 무시
    physical_score = _score(PHYSICAL_PATTERNS, text)
    query_score = _score(QUERY_PATTERNS, text)
    
    # ✅ 감정 키워드가 있으면 우선적으로 emotional로 분류 (가중치 증가)
    if preference_score == 0:  # 취향이 없을 때만
        try:
            from life_assist_dm.life_assist_dm.support_chains import (
                EMOTION_POSITIVE_WORDS, EMOTION_NEGATIVE_WORDS, 
                EMOTION_TIRED_WORDS, EMOTION_ANXIOUS_WORDS
            )
            _all_emotion_words = EMOTION_POSITIVE_WORDS + EMOTION_NEGATIVE_WORDS + EMOTION_TIRED_WORDS + EMOTION_ANXIOUS_WORDS
            has_emotion_keyword = any(word in text for word in _all_emotion_words)
            if has_emotion_keyword:
                emotional_score += 2  # 감정 점수 가중치 증가
        except ImportError:
            pass
    
    
    family_info_patterns = [
        rf"{FAMILY_RELATION_KEYWORDS_PATTERN}.*(이름|이름은|이름이).*(이야|이에요|입니다|야|다)",
        rf"(내|우리|제)\s*{FAMILY_RELATION_KEYWORDS_PATTERN}.*(이름|이름은|이름이).*([가-힣]{{2,}})",
    ]
    family_info_score = _score(family_info_patterns, text)
    if family_info_score > 0:
        is_family_query = any(re.search(p, text) for p in [
            r".*\?$",  # 물음표로 끝나는 경우
            r"(이름|이름은|이름이).*(뭐|무엇|알려|말해)",
        ])
        if not is_family_query:
            # 정보 저장이므로 cognitive로 분류
            cognitive_score += family_info_score * 2  # 가중치 증가
            query_score = max(0, query_score - family_info_score)  # query 점수 감소
    
    if preference_score > 0:
        if is_preference_query:
            return ClassificationResult(
                category="query",
                confidence=0.9,
                probabilities={"query": query_score, "cognitive": preference_score},
                method="preference_query_patterns"
            )
        else:
            # 취향 저장 발화는 cognitive로
            return ClassificationResult(
                category="cognitive",
                confidence=0.9,
                probabilities={"cognitive": preference_score + cognitive_score},
                method="preference_cognitive_patterns"
            )
    
    scores = {
        "cognitive": cognitive_score,
        "emotional": emotional_score,
        "physical": physical_score,
        "query": query_score
    }
    
    # 최고 점수 카테고리 찾기
    max_score = max(scores.values())
    if max_score > 0:
        top_category = max(scores, key=scores.get)
        confidence = min(0.9, 0.5 + (max_score * 0.1))
        return ClassificationResult(
            category=top_category,
            confidence=confidence,
            probabilities=scores,
            method="hardcoded_patterns"
        )
    
    # 2️⃣ 하드코딩 실패 시 LLM fallback
    try:
        from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm
        
        llm = get_llm()
        prompt = f"""다음 한국어 문장을 분석하여 의도를 분류해주세요: "{text}"

카테고리 설명:
- cognitive: 정보 저장/기록 요청 
  * 예: "약 먹었어", "일정 잡아줘", "기억해줘", "양말은 침대 옆에 있어"
  * **가족 관계 정보 저장**: 가족 관계(동생, 형, 누나, 언니, 오빠, 엄마, 아빠, 어머니, 아버지, 부모, 할머니, 할아버지, 
    손자, 손녀, 손주, 며느리, 사위, 시아버지, 시아머니, 장인, 장모, 시누이, 처남, 처제, 처형, 고모, 이모, 삼촌, 
    조카, 사촌, 형수, 제수, 시할머니, 외할머니 등 **모든 가족 관계**)와 이름을 함께 말하는 경우
  * 예: "동생 이름은 권서율이야", "할머니 이름은 홍길순이에요", "며느리 이름은 김영희야", 
    "사위 이름은 박철수입니다", "손자 이름은 이민수야", "시어머니 이름은 이영희예요" 등
  * **중요**: "~이름은 ~이야/이에요/입니다" 형태는 정보 저장(cognitive)이며, "~이름이 뭐야?" 같은 질문은 query입니다
  * **나이 많은 사용자도 고려**: 손자, 손녀, 며느리, 사위, 시댁, 처가 등 다양한 가족 관계를 인식하세요
  
- physical: 물리적 행동 요청 (예: "물 가져와", "찾아줘", "정리해줘")  
- emotional: 감정/인사 표현 (예: "안녕", "고마워", "피곤해", "기분이 좋아")
- query: 정보 조회/질문 요청 
  * 예: "어디있어?", "몇시야?", "뭐야?", "알려줘", "어떻게 해결하는 게 좋을까?"
  * 예: "동생 이름이 뭐야?", "할머니 이름이 뭐예요?" 등 질문 형태

**중요 구분 규칙**:
1. **가족 관계 + 이름 정보 저장**: "~이름은 ~이야/이에요/입니다" → cognitive (정보 저장)
   - 모든 가족 관계(기본, 확장 포함)를 인식: 동생, 형, 손자, 며느리, 사위, 시아버지, 장인, 고모, 이모, 삼촌, 조카 등
2. **가족 관계 + 이름 조회**: "~이름이 뭐야?/뭐예요?" → query (정보 조회)
3. "어떻게 해결하는 게 좋을까?" → query (해결책/조언 요청)

JSON 형식으로 응답:
{{"category": "cognitive", "confidence": 0.95}}"""

        response = llm.invoke(prompt)
        result_text = response.content.strip()
        
        # JSON 파싱
        json_text = re.sub(r'```json\s*', '', result_text)
        json_text = re.sub(r'```\s*$', '', json_text).strip()
        
        try:
            result = json.loads(json_text)
            category = result.get("category", "query")
            confidence = result.get("confidence", 0.8)
            
            return ClassificationResult(
                category=category,
                confidence=confidence,
                method="llm_fallback"
            )
        except json.JSONDecodeError:
            # JSON 파싱 실패 시 기본값
            return ClassificationResult(
                category="query",
                confidence=0.5,
                method="llm_fallback_error"
            )
            
    except Exception as e:
        print(f"[WARN] LLM 분류 실패: {e}")
        # LLM 실패 시 기본값
        return ClassificationResult(
            category="query",
            confidence=0.3,
            method="llm_error_fallback"
        )

# 기존 호환성을 위한 함수들
def classify_text(text: str) -> str:
    """기존 호환성을 위한 단순 분류"""
    result = classify_hybrid(text)
    return result.category

def is_physical_action(text: str) -> bool:
    """물리적 행동 여부 확인"""
    result = classify_hybrid(text)
    return result.category == "physical"

def is_emotional(text: str) -> bool:
    """감정적 표현 여부 확인"""
    result = classify_hybrid(text)
    return result.category == "emotional"

def is_cognitive(text: str) -> bool:
    """인지적 작업 여부 확인"""
    result = classify_hybrid(text)
    return result.category == "cognitive"

def is_query(text: str) -> bool:
    """질문 여부 확인"""
    result = classify_hybrid(text)
    return result.category == "query"


# ==========================================================
# 🔧 LLM+규칙 융합형 의도 분류기 (선택적 사용)
# ==========================================================
def classify_intent(text: str, llm_intent: Optional[str] = None) -> str:
    """LLM 결과와 규칙 스코어를 융합해 intent를 판별한다.

    Args:
        text: 사용자 입력 텍스트
        llm_intent: 외부 LLM이 제안한 intent("cognitive"|"emotional"|"query"|"physical")
    Returns:
        최종 intent 문자열
    """
    try:
        t = str(text or "")

        # 규칙 기반 점수 (존재 여부 스코어)
        pref_score = _score(PREFERENCE_PATTERNS, t)
        emo_score = _score(EMOTIONAL_PATTERNS, t)
        qry_score = _score(QUERY_PATTERNS, t)

        # 문맥 보정
        if "기분" in t:
            emo_score += 1
        elif "좋아" in t and "기분" not in t:
            pref_score += 1

        # LLM 제안 반영 (가중치 0.3~0.4)
        def llm_w(intent: str) -> float:
            return 1.0 if (llm_intent or "").strip() == intent else 0.0

        intent_scores = {
            "cognitive": pref_score * 0.6 + llm_w("cognitive") * 0.4,
            "emotional": emo_score * 0.6 + llm_w("emotional") * 0.4,
            "query": qry_score * 0.7 + llm_w("query") * 0.3,
            # physical은 규칙 우선 - 현재 규칙 기반 분류만 유지
            "physical": _score(PHYSICAL_PATTERNS, t) * 0.8 + llm_w("physical") * 0.2,
        }

        # 최종 의도 선택
        final_intent = max(intent_scores, key=intent_scores.get)
        print(f"[INTENT-FUSION] text='{t}' | scores={intent_scores} | final={final_intent}")
        return final_intent
    except Exception:
        # 문제가 생겨도 보수적으로 cognitive
        return "cognitive"