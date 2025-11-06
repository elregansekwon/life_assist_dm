# task_classifier.py
from __future__ import annotations
from typing import Literal, Dict, List, Optional, Any
import re
import json
from collections import OrderedDict

Category = Literal["cognitive", "emotional", "physical", "query"]

# 확률 분포 기반 분류 결과
class ClassificationResult:
    def __init__(self, category: Category, confidence: float, probabilities: Dict[str, float], reasoning: str = ""):
        self.category = category
        self.confidence = confidence
        self.probabilities = probabilities
        self.reasoning = reasoning
    
    def to_dict(self) -> Dict[str, any]:
        return {
            "category": self.category,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "reasoning": self.reasoning
        }

# 정확 캐싱 (문자열 일치) - LRU 캐시
MAX_CACHE_SIZE = 500
_classification_cache = OrderedDict()

def _add_to_cache(key: str, value: ClassificationResult) -> None:
    """LRU 캐시에 항목 추가 (크기 제한)"""
    # 기존 키가 있으면 제거 후 맨 뒤로 이동
    if key in _classification_cache:
        _classification_cache.move_to_end(key)
    else:
        # 새 키 추가
        _classification_cache[key] = value
        
        # 크기 제한 확인
        if len(_classification_cache) > MAX_CACHE_SIZE:
            # 가장 오래된 항목 제거 (FIFO)
            _classification_cache.popitem(last=False)
            # print(f"[CACHE] LRU 캐시 크기 제한으로 오래된 항목 제거 (현재 크기: {len(_classification_cache)})")  # 테스트 시 로그 제거

def _get_from_cache(key: str) -> Optional[ClassificationResult]:
    """LRU 캐시에서 항목 조회 (접근 시 맨 뒤로 이동)"""
    if key in _classification_cache:
        _classification_cache.move_to_end(key)
        return _classification_cache[key]
    return None

# 의문사 우선 규칙 (질문 우선 처리)
INTERROGATIVE_PATTERNS = [
    r"어디", r"언제", r"무엇", r"누구", r"몇", r"왜", r"어떻게", r"어떤",
    r"어디서", r"언제서", r"무엇을", r"누구를", r"몇 시", r"왜", r"어떻게", r"어떤",
    r"어디에", r"언제에", r"무엇이", r"누구가", r"몇 개", r"왜", r"어떻게", r"어떤",
    r"어디로", r"언제로", r"무엇으로", r"누구로", r"몇 번", r"왜", r"어떻게", r"어떤"
]

# 간단 키워드 룰
COGNITIVE_PATTERNS = [
    r"(예약|추가|기억|기록|저장|넣어).*해",
    r"(치과|병원|미용실|회의|미팅).*(예약|잡아|넣어)",
    r"일정.*넣어|일정.*추가|일정.*등록",
    r".*(먹었어|했다|했어|봤어|읽었어|들었어|갔어|왔어|만났어|만났다)",
    r".*(기억해|기록해|저장해|넣어줘|추가해줘)",
    r"(오늘|어제|지난주|이번주|다음주).*(먹었|봤|읽었|갔|왔|만났|했다|했어)",
    r"(점심|아침|저녁|식사).*(먹었|했어)",
    r"(영화|책|음악|게임).*(봤|읽었|들었|했어)",
    # 일정 정리는 query로 처리 (조회 의미)
]
EMOTIONAL_PATTERNS = [
    r"안녕|고마워|힘들|피곤|기분|좋아|행복|슬퍼|화나|좋아해|싫어해",
    r"날씨|온도|날씨.*어때|날씨.*어떤|날씨.*어떠",
    r"(오늘|현재).*(날짜|며칠|몇\s*일)",
    r"(지금|현재).*(시간|몇\s*시)",
    r"날짜.*알려|시간.*알려|몇시|몇일|몇시야|몇시에|몇시인",
    r"오늘.*날짜|지금.*시간|현재.*시간|현재.*날짜",
    r"기분.*어때|기분.*어떤|기분.*어떠|어떤.*기분",
    r"피곤해|힘들어|좋아해|싫어해|행복해|슬퍼해|화나",
]
PHYSICAL_PATTERNS = [
    r"(가져와|갖다|찾아|어디\s*있|치워|꺼내|찾아와|가져다)",
    r"켜|꺼|밝기|온도|볼륨|작동|재생|멈춰|시작|정지|청소 시작|돌려|전등|불|조명|보일러|에어컨|TV|커튼|쓰레기통|비워",
    # 정리해는 organize 의미로만 처리 (청소 의미는 support_chains.py에서 처리)
    r"정리해|정돈해|치워",
]
QUERY_PATTERNS = [
    r"(알려줘|보여줘|궁금|무엇|뭐야)",
    r"(오늘|내일|이번주|다음주).*(일정|스케줄|약속).*알려",
]

def _score(patterns, text: str) -> int:
    return sum(1 for p in patterns if re.search(p, text))

def _classify_with_llm_probabilities(text: str) -> Optional[ClassificationResult]:
    """LLM을 사용한 확률 분포 기반 분류"""
    try:
        from .gpt_utils import get_llm
        
        llm = get_llm()
        prompt = f"""문장을 분류하세요: "{text}"

카테고리:
- physical: 물건 가져오기, 찾기 (가져와, 찾아줘)
- cognitive: 정보 저장, 기록 (예약해줘, 약 먹었어, 기억해줘)  
- emotional: 인사, 감정 (안녕, 고마워, 날씨 어때)
- query: 정보 요청 (뭐야, 어때, 보여줘, 있을까)

예시:
- "물 가져다줘" → physical
- "약 먹었어" → cognitive
- "안녕하세요" → emotional
- "점심 먹었는데 약 먹어야될 게 있을까?" → query

JSON으로 응답: {{"cognitive": 0.0, "emotional": 0.0, "physical": 0.0, "query": 1.0, "top_category": "query"}}"""

        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # JSON 파싱 (더 안전한 처리)
        try:
            # 응답 텍스트에서 JSON 부분만 추출
            response_text = str(response_text).strip()
            
            # JSON이 포함된 부분 찾기
            if "{" in response_text and "}" in response_text:
                start = response_text.find("{")
                end = response_text.rfind("}") + 1
                json_text = response_text[start:end]
            else:
                json_text = response_text
            
            result = json.loads(json_text)
            
            # 확률 분포 추출 (안전한 변환) - key 이름 강제 추출
            probabilities = {}
            for category in ["cognitive", "emotional", "physical", "query"]:
                try:
                    # 다양한 가능한 key 이름 시도
                    possible_keys = [category, category.upper(), category.capitalize()]
                    value = 0.0
                    
                    for key in possible_keys:
                        if key in result:
                            value = result[key]
                            break
                    
                    # 값이 없으면 0.0으로 설정
                    if value is None:
                        value = 0.0
                    
                    probabilities[category] = float(value)
                except (ValueError, TypeError):
                    probabilities[category] = 0.0
            
            # 최고 확률 카테고리 찾기
            if probabilities:
                top_category = max(probabilities.items(), key=lambda x: x[1])
                category = top_category[0]
                confidence = top_category[1]
                
                # 디버깅 로그 추가
                # print(f"[LLM] 분류 결과: category={category}, confidence={confidence:.2f}")  # 테스트 시 로그 제거
                
                return ClassificationResult(category, confidence, probabilities, f"LLM 분류: {category} (신뢰도: {confidence:.2f})")
            else:
                # print(f"[LLM] 분류 실패: 확률 분포 없음")  # 테스트 시 로그 제거
                return None
            
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            # print(f"[WARN] LLM 분류 결과 파싱 실패: {e}")  # 테스트 시 로그 제거
            # print(f"[WARN] 원본 응답: {response_text[:100]}...")  # 테스트 시 로그 제거
            # JSON 파싱 실패 시 None 반환하여 규칙 fallback 강제 연결
            return None
            
    except Exception as e:
        # print(f"[WARN] LLM 분류 실패: {e}")  # 테스트 시 로그 제거
        return None

def _is_physical_action_llm(text: str) -> bool:
    """LLM을 사용한 물리적 행동 판단 (기존 호환성 유지)"""
    try:
        from .gpt_utils import get_llm
        
        llm = get_llm()
        prompt = f"""다음 문장이 물리적 행동(물건을 가져오기, 찾기, 꺼내기, 전달하기 등)을 요청하는지 판단해주세요.

                        문장: "{text}"

                        물리적 행동의 예시:
                        - 가져다줘, 가져와, 갖다줘, 가지고 와
                        - 꺼내다줘, 꺼내와, 꺼내
                        - 찾아줘, 찾아와, 찾기
                        - 전달해줘, 배달해줘, 건네줘
                        - 열어줘, 닫아줘
                        - 청소해줘, 정리해줘, 치워줘

                        물리적 행동이 아닌 예시:
                        - 예약해줘, 약속잡아줘, 일정잡아줘 (인지적 요청)
                        - 안녕하세요, 고마워요 (정서적 요청)
                        - 몇시야, 날씨어때 (질문)

                        YES 또는 NO로만 답변해주세요."""

        response = llm.invoke(prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        return "YES" in response_text.upper()
    except Exception as e:
        # print(f"[WARN] LLM 물리적 행동 판단 실패: {e}")  # 테스트 시 로그 제거
        return False

def _has_interrogative(text: str) -> bool:
    """의문사가 포함되어 있는지 확인"""
    return any(re.search(pattern, text) for pattern in INTERROGATIVE_PATTERNS)

def _has_location_query(text: str) -> bool:
    """위치 관련 질문 키워드가 포함되어 있는지 확인"""
    location_keywords = [
        "어디", "위치", "어디에", "어디서", "어디로", 
        "어디에 두었", "어디에 있", "어디에 두었지", "어디에 있지",
        "어디에 놓았", "어디에 놓았지", "어디에 보관", "어디에 보관했",
        "위치가", "위치를", "위치에", "위치에서", "위치로"
    ]
    return any(keyword in text for keyword in location_keywords)

def _has_schedule_query(text: str) -> bool:
    """일정 관련 질문 키워드가 포함되어 있는지 확인"""
    schedule_keywords = [
        "일정", "스케줄", "계획", "예정", "예약", "약속",
        "뭐가 있", "무엇이 있", "무슨 일", "어떤 일",
        "정리해봐", "보여줘", "알려줘", "말해봐"
        # "정리해"는 물리적 행동으로 분류되므로 제거
    ]
    return any(keyword in text for keyword in schedule_keywords)

def classify_hybrid(text: str, entities: Dict[str, List[Dict[str, Any]]] = None) -> ClassificationResult:
    """하이브리드 분류: 정확 캐싱 → LLM 우선 + 규칙 fallback + Domain-specific Prior"""
    t = (text or "").strip()
    
    # 중복 응답 맥락 필터링 (overwrite/추가/유지 관련)
    duplicate_keywords = ["덮어", "바꿔", "업데이트", "추가", "같이", "둘다", "유지", "그대로", "놔둬", "삭제", "취소"]
    if any(k in t for k in duplicate_keywords):
        return ClassificationResult(
            category="cognitive",
            confidence=1.0,
            probabilities={"cognitive": 1.0, "emotional": 0.0, "physical": 0.0, "query": 0.0},
            reasoning="중복 응답 맥락 감지"
        )
    
    # 약 관련 발화 분류 (강화된 질문 감지)
    if entities and "user.약" in entities and entities["user.약"]:
        # 약 관련 질문 키워드가 있으면 query로, 없으면 cognitive로
        question_keywords = ["있을까", "있어", "뭐야", "어때", "확인", "알려", "보여", "해야", "먹어야", "알려줘", "있으면", "있나", "뭔가", "뭐가"]
        if any(keyword in t for keyword in question_keywords):
            return ClassificationResult(
                category="query",
                confidence=1.0,
                probabilities={"cognitive": 0.0, "emotional": 0.0, "physical": 0.0, "query": 1.0},
                reasoning="약 관련 질문 감지"
            )
        else:
            return ClassificationResult(
                category="cognitive",
                confidence=1.0,
                probabilities={"cognitive": 1.0, "emotional": 0.0, "physical": 0.0, "query": 0.0},
                reasoning="약 관련 정보 저장 감지"
            )
    
    # 약 관련 질문 감지 (엔티티 없이도)
    if any(keyword in t for keyword in ["약", "먹어야", "언제", "뭐야", "알려줘", "있으면", "있을까"]):
        return ClassificationResult(
            category="query",
            confidence=0.9,
            probabilities={"cognitive": 0.1, "emotional": 0.0, "physical": 0.0, "query": 0.9},
            reasoning="약 관련 질문 감지"
        )
    
    # 물건 위치 정보 저장 요청은 cognitive로 라우팅 (맥락 구분 강화)
    if entities and "user.물건" in entities and entities["user.물건"]:
        # 저장 요청 키워드 (질문 키워드와 구분)
        storage_keywords = ["기억해", "저장해", "알아둬", "알아두", "기억하", "저장하"]
        # 질문 키워드가 있으면 query로, 저장 키워드가 있으면 cognitive로
        question_keywords = ["있을까", "있어", "뭐야", "어때", "확인", "보여", "알려줘", "있으면", "있나", "뭔가", "뭐가"]
        
        if any(keyword in t for keyword in question_keywords):
            return ClassificationResult(
                category="query",
                confidence=1.0,
                probabilities={"cognitive": 0.0, "emotional": 0.0, "physical": 0.0, "query": 1.0},
                reasoning="물건 위치 질문 감지"
            )
        elif any(keyword in t for keyword in storage_keywords):
            return ClassificationResult(
                category="cognitive",
                confidence=1.0,
                probabilities={"cognitive": 1.0, "emotional": 0.0, "physical": 0.0, "query": 0.0},
                reasoning="물건 위치 저장 요청 감지"
            )
    
    # 입력 정규화 (공백 제거, 소문자 변환)
    norm_t = re.sub(r"\s+", " ", t.strip().lower())
    
    # 1차: 정확 캐싱 확인 (정규화된 문자열 일치)
    cached_result = _get_from_cache(norm_t)
    if cached_result:
        # print(f"[CACHE] 정확 캐싱 사용: '{norm_t}' (원본: '{t}')")  # 테스트 시 로그 제거
        return cached_result
    
    # 2차: 규칙 기반 분류 (우선순위 높임)
    rule_result = _classify_with_rules(t)
    
    if rule_result and rule_result.confidence >= 0.8:
        # 규칙 기반 confidence가 높으면 그대로 사용
        # 결과를 정확 캐싱에 저장 (정규화된 키 사용)
        _add_to_cache(norm_t, rule_result)
        return rule_result
    
    # 3차: LLM 확률 분포 기반 분류
    llm_result = _classify_with_llm_probabilities(t)
    
    if llm_result:
        # LLM 결과가 있으면 규칙과 결합
        combined_result = _combine_llm_and_rule_results(llm_result, rule_result, t)
        # 결과를 정확 캐싱에 저장 (정규화된 키 사용)
        _add_to_cache(norm_t, combined_result)
        return combined_result
    else:
        # LLM 실패 시 규칙 결과만 사용
        # 결과를 정확 캐싱에 저장 (정규화된 키 사용)
        _add_to_cache(norm_t, rule_result)
        return rule_result

def _classify_with_rules(text: str) -> ClassificationResult:
    """규칙 기반 분류 (개선된 로직)"""
    t = (text or "").strip()
    
    # 1️⃣ 물리적 행동 요청 규칙 (최우선) - PHYSICAL_PATTERNS와 중복 제거
    # physical_verbs = r"(가져다줘|갖다줘|찾아줘|정리해줘|꺼내줘|전달해줘|열어줘|닫아줘|청소해줘|꺼줘|켜줘|가져와|갖다|찾아와|정리해|꺼내와|전달해|가져다줄래|갖다줄래|찾아봐|찾아봐줘|가져와줘|갖다와|찾아와줘|정리해줘|꺼내와줘|전달해줘|가져와봐|갖다와봐|찾아와봐|정리해봐|꺼내와봐|전달해봐)"
    # if re.search(physical_verbs, t):
    #     return ClassificationResult(
    #         category="physical",
    #         confidence=0.95,
    #         probabilities={"physical": 0.95, "cognitive": 0.05, "emotional": 0.0, "query": 0.0},
    #         reasoning="물리적 행동 요청 패턴 감지"
    #     )
    
    # 2️⃣ 감정 표현 우선 규칙
    if re.search(r"(기쁘|행복|슬프|피곤|짜증|화나|외로|속상|우울|좋아|나빠|힘들|즐거워|신나|만족|뿌듯)", t):
        return ClassificationResult(
            category="emotional",
            confidence=0.9,
            probabilities={"emotional": 0.9, "cognitive": 0.1, "physical": 0.0, "query": 0.0},
            reasoning="감정 표현 패턴 감지"
        )
    
    # 3️⃣ 위치 선언 패턴 (cognitive) - 물리적 행동이 없을 때만
    if re.search(r"(에 있어\.|에 놓여\.|에 두고\.|에 보관\.|에 보관해\.|에 넣어\.|에 두었어\.|에 놓았어\.)", t):
        return ClassificationResult(
            category="cognitive",
            confidence=0.85,
            probabilities={"cognitive": 0.85, "query": 0.15, "emotional": 0.0, "physical": 0.0},
            reasoning="위치 선언 패턴 감지"
        )
    
    # 4️⃣ 기록/선언/업데이트 패턴 (cognitive)
    if re.search(r"(했어|먹었어|다녀왔어|봤어|있어|넣어줘|추가해줘|기억해줘|저장해줘|알아둬|알아두|알려줘|알려주)", t):
        return ClassificationResult(
            category="cognitive",
            confidence=0.85,
            probabilities={"cognitive": 0.85, "query": 0.15, "emotional": 0.0, "physical": 0.0},
            reasoning="기록/선언/업데이트 패턴 감지"
        )
    
    # 5️⃣ 조회/질문 패턴 (query)
    if re.search(r"(알려줘|있지\?|기억해\?|보여줘|확인해|말해줘|요약|뭐야|어때|몇시|날짜|시간)", t):
        return ClassificationResult(
            category="query",
            confidence=0.85,
            probabilities={"query": 0.85, "cognitive": 0.15, "emotional": 0.0, "physical": 0.0},
            reasoning="조회/질문 패턴 감지"
        )
    
    # 5️⃣ 위치 질문 우선 규칙
    if _has_location_query(t):
        scores = {
            "cognitive": 10,
            "emotional": 0,
            "physical": -1,
            "query": 10,
        }
    # 6️⃣ 일정 질문 우선 규칙 (조회 vs 추가 구분)
    elif _has_schedule_query(t):
        # 일정 조회는 query로, 일정 추가는 cognitive로
        if any(keyword in t for keyword in ["알려", "보여", "뭐야", "어때", "확인"]):
            scores = {
                "cognitive": 0,
                "emotional": 0,
                "physical": -1,
                "query": 15,  # 일정 조회는 query로 강하게 분류
            }
        else:
            scores = {
                "cognitive": 10,
                "emotional": 0,
                "physical": -1,
                "query": 5,
            }
    # 7️⃣ 시간/날씨 질문 우선 규칙 (EMOTIONAL_PATTERNS에서 이미 처리됨)
    elif any(keyword in t for keyword in ["몇시야", "몇시에", "몇시인", "날씨 어때", "날씨 어떤"]):
        scores = {
            "cognitive": 0,
            "emotional": 15,  # 시간/날씨 질문은 emotional로 매우 강하게 분류
            "physical": -1,
            "query": 0,
        }
    # 8️⃣ 의문사 우선 규칙
    elif _has_interrogative(t):
        scores = {
            "cognitive": _score(COGNITIVE_PATTERNS, t) + 2,
            "emotional": _score(EMOTIONAL_PATTERNS, t) + 5,  # 날짜/시간 질문을 emotional로 강하게 분류
            "physical": 0,
            "query": _score(QUERY_PATTERNS, t) + 2,
        }
    else:
        # 9️⃣ 일반적인 패턴 매칭
        scores = {
            "cognitive": _score(COGNITIVE_PATTERNS, t),
            "emotional": _score(EMOTIONAL_PATTERNS, t),
            "physical": _score(PHYSICAL_PATTERNS, t),
            "query": _score(QUERY_PATTERNS, t),
        }
        
        # 예약 관련 패턴 강화
        if any(keyword in t for keyword in ["예약", "약속", "일정", "스케줄", "계획", "예정"]):
            scores["cognitive"] += 5  # 예약 관련은 cognitive로 강하게 분류
        
        # 날씨/시간 질문 강화 (이미 위에서 처리됨)
        # if any(keyword in t for keyword in ["날씨", "몇시", "시간", "날짜", "몇시야", "몇시에", "몇시인"]):
        #     scores["emotional"] += 15  # 날씨/시간 질문은 emotional로 매우 강하게 분류
        
        # LLM을 통한 물리적 행동 추가 판단
        if _is_physical_action_llm(t):
            scores["physical"] += 2
    
    # 우선순위: query > cognitive > emotional > physical
    cat = max(
        sorted(scores.items(), key=lambda kv: {"query":4, "cognitive":3, "emotional":2, "physical":1}[kv[0]]),
        key=lambda kv: kv[1]
    )[0]
    
    # Fallback: 모든 점수가 0이면 cognitive 기본 처리
    if scores[cat] == 0:
        cat = "cognitive"
    
    # 점수를 확률로 변환 (0-1 범위)
    max_score = max(scores.values()) if max(scores.values()) > 0 else 1
    probabilities = {k: v / max_score for k, v in scores.items()}
    confidence = probabilities[cat]
    
    # 디버깅 로그 추가
    # print(f"[RULE] 분류 결과: category={cat}, confidence={confidence:.2f}")  # 테스트 시 로그 제거
    
    return ClassificationResult(cat, confidence, probabilities)

def _combine_llm_and_rule_results(llm_result: ClassificationResult, rule_result: ClassificationResult, text: str) -> ClassificationResult:
    """LLM과 규칙 결과를 결합 (단순화된 로직)"""
    
    # 물리적 명령은 규칙 우선 (명확한 패턴)
    physical_keywords = ["가져다줘", "찾아줘", "꺼내줘", "전달해줘", "열어줘", "닫아줘", "청소해줘", "정리해줘"]
    if any(keyword in text for keyword in physical_keywords) and rule_result.category == "physical":
        return rule_result
    
    # 시간/날씨 질문은 규칙 우선 (명확한 패턴)
    time_weather_keywords = ["몇시야", "몇시에", "몇시인", "날씨 어때", "날씨 어떤", "몇시", "날씨"]
    if any(keyword in text for keyword in time_weather_keywords) and rule_result.category == "emotional":
        return rule_result
    
    # 기본: LLM 결과 우선 (70% LLM, 30% 규칙)
    combined_probabilities = {}
    for category in ["cognitive", "emotional", "physical", "query"]:
        llm_prob = llm_result.probabilities.get(category, 0.0)
        rule_prob = rule_result.probabilities.get(category, 0.0)
        combined_probabilities[category] = 0.7 * llm_prob + 0.3 * rule_prob
    
    # 최고 확률 카테고리 찾기
    top_category = max(combined_probabilities.items(), key=lambda x: x[1])
    category = top_category[0]
    confidence = top_category[1]
    
    return ClassificationResult(category, confidence, combined_probabilities, f"LLM+규칙 결합: {category} (신뢰도: {confidence:.2f})")


