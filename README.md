# 🧠 LifeAssist DM  
**Dialog Manager for Life Assist Robot**

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
- 물리적 지원 → 수행 여부 확인 문장 + 영어 번역(`/`로 구분)

**예시:**
[인지] 오늘 감기약 드셨나요?
[정서] 오늘 날씨가 좋네요.
[물리적 지원] 물 갖다 드릴까요? / Would you like me to bring you some water?

yaml
코드 복사

---

### 🔹 **SentenceCorrector**

- STT 인식 결과의 **띄어쓰기 / 문법 보정** 수행  
- 보다 자연스러운 문장으로 LLM 입력
