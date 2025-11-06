import time
import re
import sys
import os
from threading import Thread
from datetime import datetime
import logging

# UTF-8 인코딩 강제 설정 (rqt_service_caller ASCII 오류 방지)
if sys.stdout.encoding != 'utf-8':
    # stdout/stderr을 UTF-8로 재설정
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 환경 변수 설정 (Python 런타임 레벨)
os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import rclpy
from rclpy.node import Node

from life_assist_dm.life_assist_dm.dialog_manager.config import DialogManagerHeader
from life_assist_dm.life_assist_dm.llm.gpt_utils import LifeAssistant
from life_assist_dm_msgs.srv import Conversation

from life_assist_dm.life_assist_dm.task_classifier import classify_hybrid
from life_assist_dm.life_assist_dm.memory import LifeAssistMemory
# ✅ PhysicalSupportChain은 현재 사용되지 않음 (handle_physical_task를 직접 호출)

# rqt 메모리 과부하 방지: httpx, httpcore 로그 억제
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def _filter_safety_apology(response_text: str) -> str:
    """
    LLM 안전 필터로 인해 자동 추가된 불필요한 사과문을 제거합니다.
    
    OpenAI LLM이 "실제로 물건을 가져올 수 없다"는 안전 필터 때문에
    "죄송하지만, 제가 실제로는 가져다드릴 수 없습니다" 같은 문장을 자동으로 추가합니다.
    하지만 ROS2 로봇 시스템에서는 실제로 robot_command를 전달하므로 이런 문장이 불필요합니다.
    
    Args:
        response_text: 원본 응답 텍스트
        
    Returns:
        사과문이 제거된 깔끔한 응답 텍스트
    """
    if not response_text:
        return response_text
    
    cleaned_text = response_text
    
    # ✅ 우선순위 1: "죄송하지만"이 포함된 경우, 그 이전 부분만 유지
    if "죄송하지만" in cleaned_text:
        cleaned_text = cleaned_text.split("죄송하지만")[0].strip()
    
    # ✅ 우선순위 2: 다른 사과문 패턴들 제거
    bad_phrases = [
        "직접 가셔야",
        "직접 가져",
        "드릴 수는 없습니다",
        "드릴 수 없습니다",
        "제가 실제로",
        "실제로는.*?수 없습니다",
        "가져다 드릴 수는 없습니다",
        "가져다드릴 수는 없습니다",
        "물건을 가져다 드릴 수는 없습니다",
        "직접 가져오셔야",
        "직접 가져오셔야 할 것 같습니다",
    ]
    
    for phrase in bad_phrases:
        # 정규식 패턴이면 re.sub 사용, 아니면 문자열 split 사용
        if ".*?" in phrase:
            cleaned_text = re.sub(phrase, "", cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        else:
            if phrase in cleaned_text:
                # 해당 문구 이전까지만 유지
                idx = cleaned_text.find(phrase)
                if idx > 0:
                    cleaned_text = cleaned_text[:idx].strip()
    
    # 연속된 공백 정리
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # 마침표나 쉼표로 끝나도록 정리
    if cleaned_text and not cleaned_text.endswith(('.', '!', '?', '요', '다', '니다')):
        # 자연스러운 마무리를 위해 "요" 추가하지 않음 (원문 유지)
        pass
    
    return cleaned_text if cleaned_text else response_text

<<<<<<< HEAD
#추가
from life_assist_dm.llm.memory import LifeAssistMemory, MemoryConfig
import os
#

from life_assist_dm_msgs.srv import STTListen, TTSSpeak
=======
>>>>>>> 9f3045d (2025-11-06 수정 사항 반영)



class DialogManager(Node):
    def __init__(self):
        super().__init__('dialog_manager')
        header = DialogManagerHeader(self)
        self.cfg = header.cfg

<<<<<<< HEAD
        #수정
        #self.life_assistant = LifeAssistant(model_name=self.cfg.dm.gpt_model)
        self.session_id = os.getenv("DM_SESSION_ID", "user1")
        self.memory = LifeAssistMemory(
            MemoryConfig(
                sqlite_path="~/.life_assist_dm/history.sqlite",
                chroma_dir="~/.life_assist_dm/chroma",
                use_window_k=5,
                summary_enabled=True,
                entity_enabled=True,
                auto_export_enabled=True,
                export_dir="conversation_extract",
            )
        )
        #
        self.stt_listen_client = self.create_client(STTListen, 'stt_listen')
        self.tts_speak_client  = self.create_client(TTSSpeak, 'tts_speak')
=======
        self.life_assistant = LifeAssistant(model_name=self.cfg.dm.gpt_model)
        self.memory = LifeAssistMemory(self.cfg)
        # ✅ PhysicalSupportChain은 현재 사용되지 않음 (handle_physical_task를 직접 호출)
        # self.support_chain = PhysicalSupportChain()

        # 사용자 이름 확인 상태 추적
        self.user_name_status = {}  # {session_id: "unknown" | "asking" | "confirmed"}
        
        # 대화 타임아웃 추적 (3분)
        self.last_conversation_time = {}  # {session_id: timestamp}
        self.session_timeout = 180  # 3분 = 180초
>>>>>>> 9f3045d (2025-11-06 수정 사항 반영)

        self.conversation_service = self.create_service(Conversation,   
                                                        'conversation',
                                                        self.handle_conversation)   

    def _summarize_emotion_context(self, user_text: str) -> str:
        """감정 표현의 원인/상황을 간결하게 요약
        
        예시:
        - "내 남자친구가 연락을 안받아서 너무 속상해" → "남자친구의 연락 문제"
        - "시험에서 떨어져서 너무 슬퍼" → "시험 실패"
        - "오늘 회사에서 상사한테 혼나서 기분이 안좋아" → "직장 문제"
        """
        try:
            from life_assist_dm.life_assist_dm.support_chains import _summarize_emotion_context_for_save
            from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm
            llm = get_llm()
            return _summarize_emotion_context_for_save(user_text, llm)
        except Exception as e:
            self.get_logger().warning(f"감정 상황 요약 실패: {e}")
            # 실패 시 원문 일부 사용
            return user_text[:30]

    def handle_conversation(self, request, response):
        # 서비스 요청 처리 (UTF-8 인코딩은 모듈 상단에서 이미 설정됨)
        user_text = request.ask
        # 로그는 UTF-8로 안전하게 출력
        try:
            self.get_logger().info(f"USER -> ROBOT: {user_text}")
        except Exception as e:
            # 로깅 실패해도 처리 계속
            self.get_logger().warning(f"로그 출력 실패: {e}")
        self.cfg.user.command = user_text

<<<<<<< HEAD
            if self.cfg.user.call:
                while rclpy.ok():
                    self.listen_and_do_loop()

    def listen_and_do_loop(self):
        # STT에 명령 듣기 요청
        # self.get_logger().info(f"Wait for command...")
        self.send_stt_listen(listen_type='command')

        # STT 음성인식 출력
        self.get_logger().info(f"USER -> ROBOT: {self.cfg.user.command}")
        to_gpt = self.cfg.user.command

        # 대화 종료 감지
        if self._is_conversation_end(to_gpt):
            self.get_logger().info("대화 종료 감지 - 엑셀 파일 추출 중...")
            self.memory.save_final_summary(self.session_id)
            self.cfg.user.call = False  # 대화 종료
            return

        # loop가 한번 돈 후 dm.srv_type이 정해진 경우
        if self.cfg.dm.srv_type:
            to_gpt = f"[{self.cfg.dm.srv_type}] {to_gpt}"
        # GPT 입력
        self.get_logger().info(f"GPT INPUT: {to_gpt}")

        # GPT에 명령 분석, 대답 생성
        response = self.classify_service_type(to_gpt)
        self.get_logger().info(f"ROBOT -> USER: {response}")

        # GPT 대답에서 서비스 종류와 응답 구분
        srv_type, answer = self.split_srv_and_command(response)
        self.get_logger().info(f"STT: {self.cfg.user.command} -> {srv_type}")

        # 서비스 종류 정하기
        if not self.cfg.dm.srv_type:
            self.cfg.dm.srv_type = srv_type

        # 서비스 종류에 따라 수행
        if srv_type == '인지' or srv_type == '정서':
            self.send_tts_text(answer)
        elif srv_type == '물리적지원':
            ask, rb_command = self.split_pysical_command(answer)
            self.send_tts_text(ask)

    def split_srv_and_command(self, text):
        match = re.match(r'\[(.*?)\]\s*(.*)', text)
        if match:
            srv_type = match.group(1).replace(" ", "")
            command = match.group(2).removeprefix('서비스: ')
            return srv_type, command
        else:
            return False, False

    def split_pysical_command(self, text):
        parts = [part.strip() for part in text.split('/')]
        answer = parts[0]
        rb_command = parts
        return answer, rb_command

    def _is_conversation_end(self, user_input):
        """대화 종료 키워드 감지"""
        end_keywords = [
            "종료", "끝", "그만", "안녕", "잘가", "바이", "bye", "exit", "quit",
            "대화 끝", "대화 종료", "그만해", "그만하자", "끝내자", "끝내",
            "고마워", "감사해", "수고했어", "수고했어요"
        ]
        
        user_input_lower = user_input.lower().strip()
        return any(keyword in user_input_lower for keyword in end_keywords)

    def classify_service_type(self, command):
        #수정
        # srv_type = self.life_assistant(command)
        # return srv_type
        return self.memory.generate(self.session_id, command)
        #

    def send_stt_listen(self, listen_type):
        req = STTListen.Request()
        req.type = listen_type
        future = self.stt_listen_client.call_async(req)

        # 비동기 처리 → 응답 기다림 (옵션)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            if listen_type == 'call':
                # self.get_logger().info(f"{future.result().success}")
                if future.result().success:
                    self.cfg.user.call = True
=======
        try:
            session_id = "default_session"
            current_time = time.time()
            
            # -1️⃣ 세션 타임아웃 체크 (사용자 이름 확인 후)
            # 마지막 대화로부터 3분 이상 지났는지 확인 (단, 사용자 이름이 있는 경우에만)
            if session_id in self.last_conversation_time and session_id in self.memory.user_names:
                time_elapsed = current_time - self.last_conversation_time[session_id]
                if time_elapsed > self.session_timeout:
                    self.get_logger().info(f"[SESSION TIMEOUT] {time_elapsed:.1f}초 경과 - 세션 종료")
                    
                    # 대화 요약 저장
                    try:
                        user_name = self.memory.user_names.get(session_id)
                        if user_name and user_name != "사용자":
                            now = datetime.now()
                            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                            
                            memory_vars = self.memory.conversation_memory.load_memory_variables({})
                            history = memory_vars.get("history", "")
                            
                            if history:
                                summary = f"{timestamp}: 세션 타임아웃(3분) - {history[:200] if len(history) > 200 else history}"
                                self.memory.excel_manager.save_conversation_summary(user_name, summary)
                                self.get_logger().info(f"대화 요약 저장 완료: {user_name}")
                            
                            # 타임아웃 시 버퍼 플러시
                            try:
                                self.memory.flush_memory_to_excel(session_id)
                                self.get_logger().info(f"[FLUSH] 세션 타임아웃 - 데이터 엑셀로 저장 완료: {user_name}")
                            except Exception as e:
                                self.get_logger().warning(f"[FLUSH WARNING] 세션 타임아웃 flush 실패: {e}")
                    except Exception as e:
                        self.get_logger().warning(f"대화 요약 저장 실패: {e}")
                    
                    # 세션 초기화
                    if session_id in self.memory.user_names:
                        del self.memory.user_names[session_id]
                    if session_id in self.user_name_status:
                        del self.user_name_status[session_id]
                    if session_id in self.last_conversation_time:
                        del self.last_conversation_time[session_id]
                    
                    response.success = True
                    response.answer = "세션이 3분 동안 대화가 없어 자동 종료되었습니다. 새로운 세션을 시작하려면 메시지를 보내주세요."
                    response.act_type = "emotional"
                    return response
            
            # 마지막 대화 시간 업데이트 (사용자 이름이 확인된 경우에만)
            if session_id in self.memory.user_names:
                self.last_conversation_time[session_id] = current_time
            
            # 1️⃣ pending_question 체크 - 이전 질문에 대한 답변이 있는지 확인
            # 새 명령인지 확인 (물건+위치 패턴이 있으면 새 명령, 또는 명시적 새 정보 제공)
            is_new_command = any(keyword in user_text for keyword in [
                "에 있어", "에 있어.", "에서",  # 명확한 위치 표현 (마침표 포함)
                "가져", "갖다", "가져와",  # 가져오기 명령 (공백 제거)
                "찾아", "정리", "꺼내",  # 기타 명령
                "이름", "약", "일정", "약속",  # 새 정보 제공
            ])
            
            if hasattr(self.memory, 'pending_question') and session_id in self.memory.pending_question and not is_new_command:
                # 새 명령이 아니면 pending_question 처리
                self.get_logger().info(f"[PENDING] 처리 시작: {self.memory.pending_question[session_id]}")
                from life_assist_dm.life_assist_dm.support_chains import handle_pending_answer
                answer = handle_pending_answer(user_text, self.memory, session_id)
                if isinstance(answer, dict):
                    response.success = answer.get('success', True)
                    response.answer = answer.get('message', str(answer))
                    response.act_type = "physical"  # pending_question은 주로 physical
>>>>>>> 9f3045d (2025-11-06 수정 사항 반영)
                else:
                    response.success = True
                    response.answer = str(answer)
                    response.act_type = "physical"
                self.get_logger().info(f"[PENDING] 처리 완료: {answer}")
                return response
            
            # 1️⃣ 분류 단계 (task_classifier) - 사용자 이름이 확인된 경우에만
            user_name = self.memory.user_names.get(session_id)
            self.get_logger().info(f"[NAME CHECK] 사용자 이름 상태: {user_name}")
            
            if not user_name:
                # 사용자 이름이 없으면 분류하지 않고 바로 이름 요청
                self.get_logger().info(f"[NAME REQUEST] 사용자 이름 없음 - 분류 건너뛰기")
                if self.user_name_status.get(session_id) != "asking":
                    self.user_name_status[session_id] = "asking"
                    response.success = True
                    response.answer = "안녕하세요! 대화 시작 전에 우선 지금 말씀 중인 사용자 분 이름을 말해주세요!"
                    response.act_type = "emotional"
                    self.get_logger().info(f"[NAME REQUEST] 이름 물어보기: {user_text}")
                    return response
                
                # 이미 이름을 물어본 상태이고 사용자가 답변한 경우
                if self.user_name_status.get(session_id) == "asking":
                    # 개인정보가 포함된 문장은 이름 처리하지 않고 cognitive로 넘김
                    if any(keyword in user_text for keyword in ["나이", "살", "학교", "다녀", "직업", "취미"]):
                        self.get_logger().info(f"[NAME SKIP] 개인정보 포함 문장 - cognitive로 처리: {user_text}")
                        # 이름 처리 단계를 건너뛰고 cognitive 처리로 넘어감
                        # 사용자 이름을 기본값으로 설정
                        self.memory.user_names[session_id] = "사용자"
                        self.user_name_status[session_id] = "confirmed"
                        pass
                    else:
                        # LLM을 사용한 이름 추출 (더 정확하게)
                        try:
                            from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm
                            llm = get_llm()
                            
                            prompt = f"""다음 사용자 응답에서 이름만 추출하세요. 다른 정보는 무시하세요.
                            사용자 응답: "{user_text}"

                            이름 추출 규칙:
                            - 사람 이름만 추출 (예: 홍길동, 김철수, 이영희)
                            - 불필요한 접미사(입니다, 이에요, 이야 등) 제거
                            - 불필요한 접두사(제가, 저는 등) 제거
                            - 숫자나 특수문자는 제외
                            - 이름이 없으면 "unknown" 반환

                            추출된 이름만 출력하세요:"""
                            
                            response_llm = llm.invoke(prompt)
                            name = response_llm.content.strip() if hasattr(response_llm, 'content') else str(response_llm).strip()
                            
                            # "unknown"이나 빈 값 체크
                            if not name or name.lower() in ["unknown", "알수없음", "모름"] or len(name) == 0:
                                # LLM 실패 시 간단한 처리로 fallback
                                name = user_text.strip()
                                # 불필요한 접미사 제거
                                name = name.replace("입니다", "").replace("이에요", "").replace("이야", "").replace("이에요", "").replace("입니다요", "").strip()
                                
                                # 여전히 유효하지 않으면 기본값
                                if not name or len(name) < 1 or len(name) > 20:  # 너무 긴 이름도 거부
                                    name = "사용자"
                            
                            # 이름 유효성 검증: 한글, 영문, 숫자만 허용
                            import re
                            if not re.match(r'^[가-힣A-Za-z0-9\s]+$', name):
                                self.get_logger().warning(f"유효하지 않은 이름 형식: {name}")
                                name = "사용자"
                            
                            # 사용자 이름 저장
                            self.memory.user_names[session_id] = name
                            self.user_name_status[session_id] = "confirmed"
                            
                            # 사용자 엑셀 파일 초기화 (없으면 생성)
                            # ✅ memory.excel_manager 사용 (새 인스턴스 생성 금지)
                            excel_manager = self.memory.excel_manager
                            
                            self.get_logger().info(f"[EXCEL] 사용자 엑셀 파일 확인 중: {name}")
                            
                            if not excel_manager.user_exists(name):
                                self.get_logger().info(f"[EXCEL] 새 사용자 - 엑셀 파일 생성: {name}")
                                excel_manager.initialize_user_excel(name)
                                self.get_logger().info(f"[EXCEL] 엑셀 파일 생성 완료: {excel_manager.get_user_excel_path(name)}")
                            else:
                                self.get_logger().info(f"[EXCEL] 기존 사용자 - 엑셀 데이터 로딩: {name}")
                                # 기존 사용자인 경우 엑셀에서 데이터 로딩
                                try:
                                    self.memory.load_user_data_from_excel(name, session_id)
                                except Exception as e:
                                    self.get_logger().warning(f"엑셀 데이터 로딩 실패: {e}")
                            
                            response.success = True
                            response.answer = f"네! {name}님, 반가워요. 이제 다시 원하시는 사항을 말씀해주세요."
                            response.act_type = "emotional"
                            return response
                        except Exception as e:
                            # LLM 실패 시 간단한 처리
                            name = user_text.strip()
                            name = name.replace("입니다", "").replace("이에요", "").replace("이야", "").strip()
                            
                            if name and len(name) > 0:
                                self.memory.user_names[session_id] = name
                                self.user_name_status[session_id] = "confirmed"
                                
                                # ✅ memory.excel_manager 사용 (새 인스턴스 생성 금지)
                                excel_manager = self.memory.excel_manager
                                if not excel_manager.user_exists(name):
                                    excel_manager.initialize_user_excel(name)
                                else:
                                    # 기존 사용자인 경우 엑셀에서 데이터 로딩
                                    try:
                                        self.memory.load_user_data_from_excel(name, session_id)
                                    except Exception as e:
                                        self.get_logger().warning(f"엑셀 데이터 로딩 실패: {e}")
                                
                                response.success = True
                                response.answer = f"네! {name}님, 반가워요. 이제 다시 원하시는 사항을 말씀해주세요."
                                response.act_type = "emotional"
                                return response
            
            # 데이터 새로고침 명령: 엑셀→캐시 강제 리로드
            if any(k in user_text for k in ["새로고침", "리프레시", "다시 불러", "업데이트 해"]):
                user_name = self.memory.user_names.get(session_id)
                if user_name:
                    try:
                        self.memory.load_user_data_from_excel(user_name, session_id)
                        response.success = True
                        response.answer = "엑셀에서 최신 정보를 다시 불러왔어요."
                        response.act_type = "query"
                        return response
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        self.get_logger().warning(f"새로고침 실패: {tb}")
                        response.success = False
                        # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
                        response.answer = "새로고침 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
                        response.act_type = "unknown"
                        return response

            # 질의 강제 라우팅: 물음표/기억/알고/알아 포함 시 query로 우회
            if user_text.strip().endswith("?") or any(k in user_text for k in ["기억", "알고", "알아"]):
                from life_assist_dm.life_assist_dm.support_chains import handle_query_with_lcel
                answer = handle_query_with_lcel(user_text, self.memory, session_id)
                response.success = True
                response.answer = str(answer)
                response.act_type = "query"
                # 대화 기록을 엑셀에 간단 저장
                try:
                    user_name_log = self.memory.user_names.get(session_id)
                    if user_name_log and user_name_log != "사용자":
                        # ✅ memory.excel_manager 사용 (새 인스턴스 생성 금지)
                        summary_text = f"Q: {user_text} | A: {response.answer}"
                        self.memory.excel_manager.save_conversation_summary(user_name_log, summary_text)
                except Exception as e:
                    self.get_logger().warning(f"대화 기록 저장 실패: {e}")
                return response

            # 0.5️⃣ 의도 분류 (classify_hybrid 사용 - 중복 LLM 호출 방지)
            # ✅ classify_hybrid()는 이미 하드코딩 패턴 + LLM fallback을 포함하므로 중복 호출 제거
            result = classify_hybrid(user_text)
            act_types = result.categories if hasattr(result, 'categories') else [result.category]

            self.get_logger().info(f"[CLASSIFY] {act_types}")

            #  사용자 이름 확인 강화: 분류 결과와 상관없이 사용자 이름이 없으면 무조건 이름 요청
            if not user_name:
                self.get_logger().info(f"[NAME OVERRIDE] 사용자 이름 없음 - 분류 결과 무시하고 이름 요청")
                response.success = True
                response.answer = "안녕하세요! 대화 시작 전에 우선 지금 말씀 중인 사용자 분 이름을 말해주세요!"
                response.act_type = "emotional"
                return response

            # 2️⃣ 실제 동작 라우팅 (복합 intent 순차 처리)
            answer_parts = []
            processed_physical = False  # physical 처리 완료 여부 추적
            emotion_saved_in_this_turn = False  # 이번 턴에서 감정 저장 여부 추적 (중복 저장 방지)

            for act_type in act_types:
                # --- Cognitive (인지적 처리 + 저장)
                if act_type == "cognitive":
                    # 질문형 가드: 기억/알고/알아/물음표가 있으면 조회로 우회
                    q_guard = (user_text.strip().endswith("?") or any(k in user_text for k in ["기억", "알고", "알아"]))
                    if q_guard:
                        from life_assist_dm.life_assist_dm.support_chains import handle_query_with_lcel
                        answer = handle_query_with_lcel(user_text, self.memory, session_id)
                        answer_parts.append(str(answer))
                        continue
                    # 사용자 정보 직접 저장 분기 제거 → LLM/체인으로 일원화
                    
                    from life_assist_dm.life_assist_dm.support_chains import handle_cognitive_task_with_lcel
                    answer = handle_cognitive_task_with_lcel(user_text, self.memory, session_id)
                    # cognitive는 항상 메시지 반환 (dict 여부와 상관없이 처리)
                    if isinstance(answer, dict):
                        answer_parts.append(answer.get('message', str(answer)))
                        # dict가 반환되면 이미 physical이 체인 처리된 것
                        if any(keyword in user_text for keyword in ["가져", "갖다", "와", "찾아", "정리", "꺼내"]):
                            processed_physical = True
                    else:
                        answer_text = str(answer) if answer else ""
                        # 응답이 비어있으면 기본 메시지 제공
                        if not answer_text or answer_text.strip() == "":
                            answer_text = "말씀하신 내용을 기록해두었어요."
                        answer_parts.append(answer_text)
                        # 디버그 로그 추가
                        self.get_logger().info(f"[COGNITIVE] 응답 메시지: {answer_text}")
                    
                    # cognitive 처리에서 감정이 저장되었는지 확인 (응답 메시지로 판단)
                    if "기록해둘게요" in str(answer) or "이해해요" in str(answer):
                        emotion_saved_in_this_turn = True

                # --- Physical (물리적 지원: 위치 검색 + 행동 실행)
                elif act_type == "physical" and not processed_physical:
                    # cognitive에서 이미 처리된 경우 건너뛰기
                    from life_assist_dm.life_assist_dm.support_chains import handle_physical_task
                    try:
                        physical_result = handle_physical_task(user_text, self.memory, session_id)
                        self.get_logger().info(f"[PHYSICAL RESULT] {physical_result}")
                        # ✅ physical_result의 message를 직접 사용 (필터링은 최종 합성 단계에서 수행)
                        if isinstance(physical_result, dict):
                            answer_parts.append(physical_result.get('message', str(physical_result)))
                        else:
                            answer_parts.append(str(physical_result))
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        self.get_logger().error(f"[PHYSICAL ERROR] {tb}")
                        # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
                        answer_parts.append("물리적 작업 처리 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요.")

                # --- Emotional (감정적 응대)
                elif act_type == "emotional":
                    from life_assist_dm.life_assist_dm.support_chains import build_emotional_reply
                    # 사용자 이름이 확인된 상태인지 체크
                    user_name_confirmed = bool(self.memory.user_names.get(session_id))
                    answer = build_emotional_reply(user_text, self.memory.llm, user_name_confirmed)
                    answer_parts.append(str(answer))
                    # 🔧 감정 기록을 Excel에 저장 (간단 키워드 라벨링)
                    # 주의: 이 경로는 "emotional" 타입으로 분류된 경우만 실행됨
                    # "cognitive" 타입으로 분류된 경우는 support_chains.handle_cognitive_task_with_lcel에서 저장됨
                    # 중복 저장 방지: cognitive에서 이미 감정을 저장한 경우 건너뛰기
                    if not emotion_saved_in_this_turn:
                        try:
                            user_name_log = self.memory.user_names.get(session_id)
                            if user_name_log and user_name_log != "사용자":
                                # ✅ 공통 함수 사용하여 감정 단어와 라벨 추출
                                from life_assist_dm.life_assist_dm.support_chains import _extract_emotion_word_and_label
                                emotion_word, emo_label = _extract_emotion_word_and_label(user_text)
                                
                                # ✅ 실제 감정 단어가 있으면 그것을 저장, 없으면 라벨 저장 (없으면 "중립")
                                emotion_to_save = emotion_word if emotion_word else (emo_label if emo_label else "중립")
                                
                                # ✅ 감정의 원인/상황을 간결하게 요약
                                info_summary = self._summarize_emotion_context(user_text)
                                
                                # ✅ memory.excel_manager 사용 (새 인스턴스 생성 금지)
                                self.memory.excel_manager.save_entity_data(user_name_log, "감정", {
                                    "감정": emotion_to_save,
                                    "정보": info_summary
                                })
                                emotion_saved_in_this_turn = True
                        except Exception as e:
                            self.get_logger().warning(f"감정 기록 저장 실패: {e}")
                    else:
                        self.get_logger().debug("[SKIP] 감정 기록 중복 저장 방지 (cognitive에서 이미 저장됨)")

                # --- Query (정보 조회)
                elif act_type == "query":
                    from life_assist_dm.life_assist_dm.support_chains import handle_query_with_lcel
                    answer = handle_query_with_lcel(user_text, self.memory, session_id)
                    answer_parts.append(str(answer))

                # --- 예외 처리
                else:
                    # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
                    answer_parts.append("죄송해요, 지금은 그 요청을 처리할 수 없어요.")

            # 결과 합치기
            response.success = True
            # answer_parts가 비어있으면 기본 메시지 제공
            if not answer_parts:
                answer_parts.append("처리 완료했어요.")
            # 응답 문자열 안전화 (따옴표 등 특수문자 처리)
            safe_answer = " ".join(answer_parts)
            # ✅ LLM 안전 필터로 인한 불필요한 사과문 제거 (최종 응답 합성 후)
            safe_answer = _filter_safety_apology(safe_answer)
            safe_answer = safe_answer.replace('"', '＂').replace("'", "＇")
            response.answer = safe_answer
            response.act_type = ",".join(act_types)
            # 디버그 로그: 최종 응답 확인
            self.get_logger().info(f"[RESPONSE] 최종 응답: {response.answer}")

            # 대화 기록을 엑셀에 간단 저장 (세션 중 요약 축적)
            try:
                user_name_log = self.memory.user_names.get(session_id)
                if user_name_log and user_name_log != "사용자":
                    summary_text = f"Q: {user_text} | A: {response.answer}"
                    self.memory.excel_manager.save_conversation_summary(user_name_log, summary_text)
            except Exception as e:
                self.get_logger().warning(f"대화 기록 저장 실패: {e}")
            
            # ============================================
            # ✅ 세션 캐시 → Excel 파일로 flush (데이터 무결성 보장)
            # ============================================
            # 세션 종료 전 모든 버퍼링된 엔티티를 엑셀에 안전하게 동기화
            # - FileLock 기반으로 동시 접근 방지
            # - ROS2 노드 강제 종료 시에도 데이터 유실 방지
            # - Excel/캐시 간 불일치 방지
            # flush 중복 방지 가드 추가
            try:
                # 버퍼에 변경사항이 있는 경우에만 flush
                if hasattr(self.memory.excel_manager, "_buffered_changes"):
                    buffered_changes = self.memory.excel_manager._buffered_changes
                    # 해당 세션의 사용자 이름 확인
                    user_name = self.memory.user_names.get(session_id)
                    if user_name and user_name != "사용자":
                        # 해당 사용자의 버퍼가 비어있지 않은 경우만 flush
                        has_changes = any(
                            uname == user_name for uname, _ in buffered_changes.keys()
                        )
                        if has_changes:
                            self.memory.flush_memory_to_excel(session_id)
                            self.get_logger().debug(f"[FLUSH] 세션({session_id}) 데이터 엑셀로 저장 완료")
                        else:
                            self.get_logger().debug(f"[SKIP] flush 생략 (변경 없음): {session_id}")
            except Exception as e:
                self.get_logger().warning(f"[FLUSH WARNING] 엑셀 flush 실패: {e}")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            # 에러 로그는 UTF-8로 안전하게 출력
            self.get_logger().error(f"[ERROR] {tb}")
            response.success = False
            # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
            response.answer = "죄송해요, 처리 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
            response.act_type = "unknown"


        return response


def main(args=None):
    # UTF-8 환경 변수 강제 설정 (rqt_service_caller ASCII 오류 방지)
    # rclpy.init 전에 설정해야 rqt가 시작될 때 적용됨
    os.environ.setdefault('LC_ALL', 'C.UTF-8')
    os.environ.setdefault('LANG', 'C.UTF-8')
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    os.environ.setdefault('PYTHONUTF8', '1')
    
    rclpy.init(args=args)
    node = DialogManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("DialogManager 종료됨")
        # 시스템 종료 시 대화 요약 저장
        try:
            session_id = "default_session"
            user_name = node.memory.user_names.get(session_id)
            if user_name and user_name != "사용자":
                # 현재 세션의 대화 요약 생성
                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                
                # LCEL 메모리에서 대화 히스토리 가져오기
                memory_vars = node.memory.conversation_memory.load_memory_variables({})
                history = memory_vars.get("history", "")
                
                if history:
                    # 대화 요약 생성
                    summary = f"{timestamp}: 세션 종료 - {history[:200] if len(history) > 200 else history}"
                    node.memory.excel_manager.save_conversation_summary(user_name, summary)
                    node.get_logger().info(f"대화 요약 저장 완료: {user_name}")
                
                # 버퍼를 엑셀로 최종 플러시
                node.memory.flush_memory_to_excel(session_id)
                node.get_logger().info(f"엑셀 버퍼 플러시 완료: {user_name}")
        except Exception as e:
            node.get_logger().warning(f"엑셀 flush 실패: {e}")
    finally:
        node.destroy_node()
        # 중복 shutdown 방지
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
