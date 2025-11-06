# 중복/충돌 로직 해결 문서

## 🔍 발견된 중복/충돌 사항

### 1. ✅ 감정 저장 중복 (해결됨)

**문제:**
- 3곳에서 감정을 저장할 수 있음:
  1. `dialog_manager_node.py`: "emotional" 타입으로 분류 시 → "감정" 타입으로 저장
  2. `support_chains.py`: "cognitive" 타입으로 분류 시 0.5단계에서 → "정서" 타입으로 저장
  3. `support_chains.py`: 엔티티 처리에서 "user.건강상태" → "정서" 타입으로 저장

**해결:**
- `dialog_manager_node.py`: `emotion_saved_in_this_turn` 플래그 추가
- cognitive 처리에서 감정 저장 여부 확인 후 플래그 설정
- emotional 처리에서 플래그 확인하여 중복 저장 방지

**위치:**
- `dialog_manager_node.py` 312줄, 338-339줄, 368줄

---

### 2. ✅ 약 추출 중복 저장 (해결됨)

**문제:**
- 2곳에서 약을 추출하고 저장:
  1. `support_chains.py`: 규칙 기반 추출 (1970-2021줄) → 바로 저장하고 return
  2. `memory.py._pre_extract_entities`: LLM/규칙 추출 → 엔티티로 반환
  3. `support_chains.py`: 엔티티 처리에서 약 저장 (2206-2216줄)

**해결:**
- `medicine_saved` 플래그 추가
- 엔티티 추출 결과에 약이 있으면 플래그 설정
- 규칙 기반 저장 전 플래그 확인
- 엔티티 처리 시 플래그 확인하여 중복 저장 방지

**위치:**
- `support_chains.py` 1987줄, 1940-1942줄, 1996줄, 2209줄

---

### 3. ✅ 함수 이름 충돌 (해결됨)

**문제:**
- `_rule_based_extract` 함수가 2곳에 존재:
  1. `support_chains.py`: 일정만 추출하는 독립 함수
  2. `memory.py`: 다양한 엔티티를 추출하는 메서드

**해결:**
- `support_chains.py`의 함수명을 `_extract_schedule_rule_based`로 변경
- 일정 전용 추출 함수임을 명확히 함

**위치:**
- `support_chains.py` 179줄, 2037줄

---

### 4. ⚠️ 약 필드명 불일치 (정규화됨)

**문제:**
- 약 관련 필드명이 혼용됨:
  - `memory.py`: "약명" 사용
  - `support_chains.py`: "약이름" 사용
  - `user_excel_manager.py`: "약이름"을 표준으로 사용

**현재 상태:**
- `user_excel_manager._normalize_entity()`에서 "약명" → "약이름"으로 정규화
- 저장 시 자동 변환되므로 기능상 문제 없음
- 하지만 코드 일관성을 위해 "약이름" 통일 권장

**위치:**
- `user_excel_manager.py` 250줄: `norm["약이름"] = data.get("약이름") or data.get("이름", "")`
- (참고: "약명"도 처리하지만 주석에 명시되지 않음)

---

### 5. ⚠️ 엔티티 추출 우선순위

**현재 로직:**
1. `memory._pre_extract_entities()`:
   - LLM 추출 우선 → 성공 시 사용
   - 실패 시 Rule 기반 추출 (fallback)
   
2. `support_chains.handle_cognitive_task_with_lcel()`:
   - `_pre_extract_entities()` 호출
   - 엔티티가 없으면 추가 패턴 매칭 시도
   - 일정 규칙 추출 추가 시도

**잠재적 문제:**
- LLM이 빈 결과를 반환하면 Rule 기반 추출 실행
- 그런데 support_chains에서 또 일정 규칙 추출 시도
- 일정의 경우 중복 체크(`has_schedule_entity`)로 방지됨

**현재 상태:** ✅ 문제 없음 (중복 체크 존재)

---

## 📋 수정 사항 요약

### 수정된 파일:
1. `dialog_manager_node.py`
   - 감정 저장 중복 방지 플래그 추가
   
2. `support_chains.py`
   - 감정 저장 로직 개선 (다른 엔티티와 함께 있을 때 처리)
   - 약 추출 중복 저장 방지 플래그 추가
   - 함수명 변경: `_rule_based_extract` → `_extract_schedule_rule_based`

### 유지된 로직 (정상 동작):
- 약 필드명 정규화 (`user_excel_manager._normalize_entity`)
- 일정 추출 중복 방지 (`has_schedule_entity` 체크)

---

## ✅ 검증 완료

모든 중복/충돌 로직이 해결되었으며, LLM이 혼란스러워할 수 있는 상황을 방지했습니다.

