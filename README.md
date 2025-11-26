# 🧠 LifeAssist DM  
**Dialog Manager for Life Assist Robot**

**KETI 한국전자기술연구원 2025-2026 인턴십**

ROS 2 기반 **생활 지원 로봇 대화 관리 노드**입니다.  
사용자의 **음성(STT)**을 인식하고, **LangChain 기반 LLM**과 **다층 메모리 모듈**로 처리하여  
**인지 / 정서 / 물리적 지원** 형태의 응답 또는 **로봇 제어 명령**을 생성합니다.

---

## 🚀 주요 구성 (Core Components)

| 모듈 | 설명 |
|------|------|
| 🗣️ **dialog_manager_node.py** | ROS2 `Node`로 동작하며, STT → LLM → TTS 대화 전체 흐름 제어 |
| 🧠 **memory.py** | LangChain 기반 다층 기억 관리 (SQLite / Chroma / Excel) |
| 🔗 **support_chains.py** | Cognitive / Physical Chain 정의 및 응답 처리 |
| 🧩 **task_classifier.py** | 사용자 발화의 인텐트 분류 (인지 / 정서 / 물리적 지원) |
| 📊 **user_excel_manager.py** | 사용자별 Excel 입출력 및 중복 병합 로직 |
| 🚀 **launch/dialog_manager.launch.py** | ROS2 노드 일괄 실행 (dialog_manager / stt / tts) |

---

## ⚙️ 설치 (Installation)

```bash
# 1️⃣ 워크스페이스 생성 및 소스 다운로드
cd ~/ros_ws && mkdir -p dm_ws/src
cd ~/ros_ws/dm_ws/src
git clone https://github.com/keti-ai/life_assist_dm.git

# 2️⃣ Python 의존성 설치
cd life_assist_dm/life_assist_dm
pip install -r requirements.txt
cd ../../..

# 3️⃣ ROS2 빌드
colcon build --symlink-install
````

---

## 💬 실행 (Usage)

```bash
# Launch the dialog manager with STT/TTS nodes
ros2 launch life_assist_dm dialog_manager.launch.py
````
해당 launch 파일은 다음 노드를 함께 실행합니다:

🧠 dialog_manager: 메인 대화 관리 노드

🗣️ stt_node: STT(음성 인식) 노드 (whisper_model='base')

🔊 tts_node: TTS(음성 합성) 노드

---

## 🧩 시스템 동작 개요 (System Flow)

```text
[User Speech]
   ↓
[STT Node] — 음성 인식 결과를 텍스트로 변환
   ↓
[Dialog Manager Node]
   ├─ 인텐트 분류 (인지 / 정서 / 물리적 지원)
   ├─ 메모리 처리 (기억, 일정, 약, 물건 등)
   ├─ SQLite / Chroma / Excel 저장
   └─ 응답 생성 또는 로봇 명령 생성
   ↓
[TTS Node] — 최종 응답을 음성으로 출력
````

---

## 🧠 LLM 기반 체인 요약

### 🔹 **SupportClassifier**

- 입력 문장을 **[인지]**, **[정서]**, **[물리적 지원]** 중 하나로 분류  
- 인지/정서 → 바로 자연어 응답 생성  
- 물리적 지원 → 물리 수행 문장 + 영어로 번역 관리된 명령 로못에게 전달

**예시:**
````
[인지] 오늘 감기약 드셨나요?
[정서] 오늘 날씨가 좋네요.
[물리적 지원] 물 갖다 드릴까요?   { "action": "deliver", "target": "water", "location": "refrigerator", "original": "Please deliver water from refrigerator" }
````

---

### 🔹 **SentenceCorrector**

- STT 인식 결과의 **띄어쓰기 / 문법 보정** 수행  
- 보다 자연스러운 문장으로 LLM 입력

---

## ⚙️ 주요 설정 (Dialog Manager Node)

```python
MemoryConfig(
    sqlite_path="~/.life_assist_dm/history.sqlite",
    chroma_dir="~/.life_assist_dm/chroma",
    use_window_k=5,
    summary_enabled=True,
    entity_enabled=True,
    auto_export_enabled=True,
    export_dir="conversation_extract",
)
````
- 대화 종료 시 Excel로 자동 내보내기
- 종료 키워드 ("종료", "그만", "안녕") 감지 시 세션 저장 및 마감

---

## 🧾 예시 발화 (Examples)

| 사용자 발화 | 처리 결과 |
|--------------|------------|
| “내일 오전 9시에 치과 예약해줘.” | 일정 엔티티로 인식 → Excel 저장 후 응답 출력 |
| “자기 전에 감기약 한 알 먹어야 돼.” | 복약 정보로 인식 → Excel 기록 |
| “핸드크림은 식탁 위에 있어.” | 물건 위치 정보로 저장 |
| “이번 주 금요일 저녁 7시에 친구랑 약속 있어.” | 일정 정보 추출 및 시간 인식 |

---

## 🧰 기술 스택 (Tech Stack)

| 항목 | 내용 |
|------|------|
| **Framework** | ROS 2 Rolling (Python 3.10) |
| **Language Model** | OpenAI GPT-4o-mini |
| **Framework** | LangChain (PromptTemplate + OutputParser) |
| **Storage** | SQLite + Chroma + Excel (Pandas) |
| **Service Interfaces** | `STTListen.srv`, `TTSSpeak.srv` |

---

## 📄 라이선스 (License)

**MIT License © 2025 KETI AI / 권서연**

<br><br>
본 프로젝트는 **생활 지원 로봇 대화 관리 모듈 연구**를 목적으로 제작되었습니다.  
실험용 예제이며, 실제 서비스 배포 환경에서는 **STT / TTS / LLM API 설정**이 필요합니다.

<br><br>
