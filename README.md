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

## 💬 실행 (Usage)

```bash
# Launch the dialog manager with STT/TTS nodes
ros2 launch life_assist_dm dialog_manager.launch.py
````
해당 launch 파일은 다음 노드를 함께 실행합니다:

🧠 dialog_manager: 메인 대화 관리 노드

🗣️ stt_node: STT(음성 인식) 노드 (whisper_model='base')

🔊 tts_node: TTS(음성 합성) 노드
