# life_assist_dm

# 🧠 LifeAssist DM
**Dialog Manager for Life Assist Robot**  
ROS 2 기반의 생활 지원 로봇 대화 관리 노드입니다.  
사용자의 음성(STT)을 인식하고, 대화 내용을 LangChain 기반 LLM과 메모리 모듈로 처리하여  
인지 / 정서 / 물리적 지원 형태의 응답 또는 로봇 제어 명령을 생성합니다.

---

## 📦 주요 구성 (Core Components)

| 모듈 | 역할 |
|------|------|
| **dialog_manager_node.py** | ROS2 `Node`로 동작하며, STT → LLM → TTS 전체 대화 흐름 제어 |
| **memory.py** | LangChain Memory 기반 다층 기억 관리 (SQLite / Chroma / Excel) |
| **support_chains.py** | Cognitive/Physical Chain 정의 및 응답 처리 |
| **task_classifier.py** | 사용자 발화의 인텐트 분류 (인지 / 정서 / 물리적 지원) |
| **user_excel_manager.py** | 사용자별 Excel 파일 입출력 및 중복 병합 처리 |
| **launch/dialog_manager.launch.py** | ROS2 노드 일괄 실행 (dialog_manager / stt / tts) |

---

## ⚙️ 설치 (Installation)

```bash
# 워크스페이스 생성 및 소스 다운로드
cd ~/ros_ws && mkdir -p dm_ws/src
cd ~/ros_ws/dm_ws/src
git clone https://github.com/keti-ai/life_assist_dm.git

# Python 의존성 설치
cd life_assist_dm/life_assist_dm
pip install -r requirements.txt
cd ../../..

# ROS2 빌드
colcon build --symlink-install
🚀 실행 (Usage)
bash
코드 복사
# Launch the dialog manager with STT/TTS nodes
ros2 launch life_assist_dm dialog_manager.launch.py
해당 launch 파일은 다음 노드를 함께 실행합니다:

dialog_manager : 메인 대화 관리 노드

stt_node : STT(음성 인식) 노드 (whisper_model='base')

tts_node : TTS(음성 합성) 노드

🧠 동작 개요 (System Flow)
text
코드 복사
[User Speech]
   ↓
[STT Node] — 음성 인식 결과를 문장으로 변환
   ↓
[Dialog Manager Node]
   ├─ Service Type 분류 (인지 / 정서 / 물리적 지원)
   ├─ Memory 처리 (기억, 일정, 약, 물건 등)
   ├─ Excel / SQLite / Chroma 저장
   └─ 응답 생성 또는 로봇 명령 생성
   ↓
[TTS Node] — 응답을 음성으로 출력
🔍 LLM 프롬프트 요약
🧩 SupportClassifier
입력된 문장을 보고 [인지], [정서], [물리적 지원] 중 하나로 분류

인지/정서 → 바로 응답 문장 생성

물리적 지원 → 수행 여부 묻는 문장과 영어 번역을 /로 구분해 반환

예시:

css
코드 복사
[인지] 오늘 감기약 드셨나요?  
[정서] 오늘 날씨가 좋네요.  
[물리적 지원] 물 갖다 드릴까요? / Would you like me to bring you some water?
✏️ SentenceCorrector
STT로 생성된 문장의 띄어쓰기 및 문법을 보정하여 자연스럽게 수정

🧩 주요 설정 (dialog_manager_node.py)
LifeAssistMemory 초기화 시 구성:

python
코드 복사
MemoryConfig(
    sqlite_path="~/.life_assist_dm/history.sqlite",
    chroma_dir="~/.life_assist_dm/chroma",
    use_window_k=5,
    summary_enabled=True,
    entity_enabled=True,
    auto_export_enabled=True,
    export_dir="conversation_extract",
)
세션 종료 시 Excel 대화기록 및 요약 자동 저장

대화 종료 키워드 감지 (“종료”, “그만”, “안녕” 등)

🧾 예시 발화
사용자 발화	처리 결과
“내일 오전 9시에 치과 예약해줘.”	일정 엔티티로 인식 → Excel 저장 후 응답 출력
“자기 전에 감기약 한 알 먹어야 돼.”	복약 정보로 인식 → Excel 기록
“핸드크림은 식탁 위에 있어.”	물건 위치 정보로 저장
“이번 주 금요일 저녁 7시에 친구랑 약속 있어.”	일정으로 기록 및 시간 추출

🧰 기술 스택
Framework: ROS 2 Rolling (Python 3.10)

Language Model: OpenAI GPT-4o-mini

LangChain 기반 체인 구성 (PromptTemplate + OutputParser)

Storage: SQLite + Chroma + Excel (Pandas)

Service Interfaces: STTListen.srv, TTSSpeak.srv

📄 라이선스
MIT License © 2025 KETI AI / 권서연

본 프로젝트는 생활 지원 로봇 대화 관리 모듈 연구를 목적으로 제작되었습니다.
실험용 예제이며, 실제 서비스 배포 환경에서는 STT/TTS/LLM API 설정이 필요합니다.
