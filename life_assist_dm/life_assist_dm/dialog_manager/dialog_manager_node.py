import time
import re
import sys
import os
from threading import Thread
from datetime import datetime
import logging

if sys.stdout.encoding != 'utf-8':

    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

os.environ['LC_ALL'] = 'C.UTF-8'
os.environ['LANG'] = 'C.UTF-8'
os.environ['PYTHONIOENCODING'] = 'utf-8'

import rclpy
from rclpy.node import Node

from life_assist_dm.dialog_manager.config import DialogManagerHeader
from life_assist_dm.llm.gpt_utils import LifeAssistant
from life_assist_dm_msgs.srv import Conversation

from life_assist_dm.task_classifier import classify_hybrid
from life_assist_dm.memory import LifeAssistMemory

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def _filter_safety_apology(response_text: str) -> str:
    if not response_text:
        return response_text
    
    cleaned_text = response_text
    
    if "죄송하지만" in cleaned_text:
        cleaned_text = cleaned_text.split("죄송하지만")[0].strip()
    
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

        if ".*?" in phrase:
            cleaned_text = re.sub(phrase, "", cleaned_text, flags=re.IGNORECASE | re.DOTALL)
        else:
            if phrase in cleaned_text:

                idx = cleaned_text.find(phrase)
                if idx > 0:
                    cleaned_text = cleaned_text[:idx].strip()
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    if cleaned_text and not cleaned_text.endswith(('.', '!', '?', '요', '다', '니다')):

        pass
    
    return cleaned_text if cleaned_text else response_text

class DialogManager(Node):
    def __init__(self):
        super().__init__('dialog_manager')
        header = DialogManagerHeader(self)


        self.cfg = header.cfg
        self.get_logger().info(f"cfg: {self.cfg}")

        # 사용자 이름 확인 상태 추적
        self.user_name_status = {}  # {session_id: "unknown" | "asking" | "confirmed"}
        
        # 대화 타임아웃 추적 (3분)
        self.last_conversation_time = {}  # {session_id: timestamp}
        self.session_timeout = 180  # 3분 = 180초

        # 서비스를 먼저 등록 (초기화 실패해도 서비스는 등록되도록)
        self.conversation_service = self.create_service(Conversation,   
                                                        'conversation',
                                                        self.handle_conversation)
        
        # 초기화 (서비스 등록 후)
        try:
            self.life_assistant = LifeAssistant(model_name=self.cfg.dm.gpt_model)
            self.get_logger().info("LifeAssistant 초기화 완료")
        except Exception as e:
            self.get_logger().error(f"LifeAssistant 초기화 실패: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.life_assistant = None
        
        try:
            self.memory = LifeAssistMemory(self.cfg)
            self.get_logger().info("LifeAssistMemory 초기화 완료")
        except Exception as e:
            self.get_logger().error(f"LifeAssistMemory 초기화 실패: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            self.memory = None
        
        # ✅ PhysicalSupportChain은 현재 사용되지 않음 (handle_physical_task를 직접 호출)
        # self.support_chain = PhysicalSupportChain()   

    def _summarize_emotion_context(self, user_text: str) -> str:
        try:
            from life_assist_dm.support_chains import _summarize_emotion_context_for_save
            from life_assist_dm.llm.gpt_utils import get_llm
            llm = get_llm()
            return _summarize_emotion_context_for_save(user_text, llm)
        except Exception as e:
            self.get_logger().warning(f"감정 상황 요약 실패: {e}")

            return user_text[:30]

    def handle_conversation(self, request, response):

        user_text = request.ask

        # 초기화 실패 체크
        if self.life_assistant is None or self.memory is None:
            self.get_logger().error("LifeAssistant 또는 Memory가 초기화되지 않았습니다.")
            response.success = False
            response.answer = "시스템 초기화 중 오류가 발생했습니다. 로그를 확인해주세요."
            response.act_type = "unknown"
            return response

        try:
            self.get_logger().info(f"USER -> ROBOT: {user_text}")
        except Exception as e:

            self.get_logger().warning(f"로그 출력 실패: {e}")
        self.cfg.user.command = user_text

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
                from life_assist_dm.support_chains import handle_pending_answer
                answer = handle_pending_answer(user_text, self.memory, session_id)
                if isinstance(answer, dict):
                    response.success = answer.get('success', True)
                    response.answer = answer.get('message', str(answer))
                    response.act_type = "physical"  # pending_question은 주로 physical
                else:
                    response.success = True
                    response.answer = str(answer)
                    response.act_type = "physical"
                self.get_logger().info(f"[PENDING] 처리 완료: {answer}")
                return response
            
            user_name = self.memory.user_names.get(session_id)
            self.get_logger().info(f"[NAME CHECK] 사용자 이름 상태: {user_name}")
            
            if not user_name:

                self.get_logger().info(f"[NAME REQUEST] 사용자 이름 없음 - 분류 건너뛰기")
                if self.user_name_status.get(session_id) != "asking":
                    self.user_name_status[session_id] = "asking"
                    response.success = True
                    response.answer = "안녕하세요! 대화 시작 전에 우선 지금 말씀 중인 사용자 분 이름을 말해주세요!"
                    response.act_type = "emotional"
                    self.get_logger().info(f"[NAME REQUEST] 이름 물어보기: {user_text}")
                    return response
                
                if self.user_name_status.get(session_id) == "asking":

                    if any(keyword in user_text for keyword in ["나이", "살", "학교", "다녀", "직업", "취미"]):
                        self.get_logger().info(f"[NAME SKIP] 개인정보 포함 문장 - cognitive로 처리: {user_text}")

                        self.memory.user_names[session_id] = "사용자"
                        self.user_name_status[session_id] = "confirmed"
                        pass
                    else:

                        try:
                            from life_assist_dm.llm.gpt_utils import get_llm
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
                            
                            if not name or name.lower() in ["unknown", "알수없음", "모름"] or len(name) == 0:

                                name = user_text.strip()

                                name = name.replace("입니다", "").replace("이에요", "").replace("이야", "").replace("이에요", "").replace("입니다요", "").strip()
                                
                                if not name or len(name) < 1 or len(name) > 20:
                                    name = "사용자"
                            
                            import re
                            if not re.match(r'^[가-힣A-Za-z0-9\s]+$', name):
                                self.get_logger().warning(f"유효하지 않은 이름 형식: {name}")
                                name = "사용자"
                            
                            self.memory.user_names[session_id] = name
                            self.user_name_status[session_id] = "confirmed"
                            
                            excel_manager = self.memory.excel_manager
                            
                            self.get_logger().info(f"[EXCEL] 사용자 엑셀 파일 확인 중: {name}")
                            
                            if not excel_manager.user_exists(name):
                                self.get_logger().info(f"[EXCEL] 새 사용자 - 엑셀 파일 생성: {name}")
                                excel_manager.initialize_user_excel(name)
                                self.get_logger().info(f"[EXCEL] 엑셀 파일 생성 완료: {excel_manager.get_user_excel_path(name)}")
                            else:
                                self.get_logger().info(f"[EXCEL] 기존 사용자 - 엑셀 데이터 로딩: {name}")

                                try:
                                    self.memory.load_user_data_from_excel(name, session_id)
                                except Exception as e:
                                    self.get_logger().warning(f"엑셀 데이터 로딩 실패: {e}")
                            
                            response.success = True
                            response.answer = f"네! {name}님, 반가워요. 이제 다시 원하시는 사항을 말씀해주세요."
                            response.act_type = "emotional"
                            return response
                        except Exception as e:

                            name = user_text.strip()
                            name = name.replace("입니다", "").replace("이에요", "").replace("이야", "").strip()
                            
                            if name and len(name) > 0:
                                self.memory.user_names[session_id] = name
                                self.user_name_status[session_id] = "confirmed"
                                
                                excel_manager = self.memory.excel_manager
                                if not excel_manager.user_exists(name):
                                    excel_manager.initialize_user_excel(name)
                                else:

                                    try:
                                        self.memory.load_user_data_from_excel(name, session_id)
                                    except Exception as e:
                                        self.get_logger().warning(f"엑셀 데이터 로딩 실패: {e}")
                                
                                response.success = True
                                response.answer = f"네! {name}님, 반가워요. 이제 다시 원하시는 사항을 말씀해주세요."
                                response.act_type = "emotional"
                                return response
            
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

                        response.answer = "새로고침 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
                        response.act_type = "unknown"
                        return response

            if user_text.strip().endswith("?") or any(k in user_text for k in ["기억", "알고", "알아"]):
                from life_assist_dm.support_chains import handle_query_with_lcel
                answer = handle_query_with_lcel(user_text, self.memory, session_id)
                response.success = True
                response.answer = str(answer)
                response.act_type = "query"

                try:
                    user_name_log = self.memory.user_names.get(session_id)
                    if user_name_log and user_name_log != "사용자":

                        summary_text = f"Q: {user_text} | A: {response.answer}"
                        self.memory.excel_manager.save_conversation_summary(user_name_log, summary_text)
                except Exception as e:
                    self.get_logger().warning(f"대화 기록 저장 실패: {e}")
                return response

            result = classify_hybrid(user_text)
            act_types = result.categories if hasattr(result, 'categories') else [result.category]

            self.get_logger().info(f"[CLASSIFY] {act_types}")

            if not user_name:
                self.get_logger().info(f"[NAME OVERRIDE] 사용자 이름 없음 - 분류 결과 무시하고 이름 요청")
                response.success = True
                response.answer = "안녕하세요! 대화 시작 전에 우선 지금 말씀 중인 사용자 분 이름을 말해주세요!"
                response.act_type = "emotional"
                return response

            answer_parts = []
            processed_physical = False
            emotion_saved_in_this_turn = False
            robot_command_str = ""

            for act_type in act_types:

                if act_type == "cognitive":

                    q_guard = (user_text.strip().endswith("?") or any(k in user_text for k in ["기억", "알고", "알아"]))
                    if q_guard:
                        from life_assist_dm.support_chains import handle_query_with_lcel
                        answer = handle_query_with_lcel(user_text, self.memory, session_id)
                        answer_parts.append(str(answer))
                        continue

                    from life_assist_dm.support_chains import handle_cognitive_task_with_lcel
                    answer = handle_cognitive_task_with_lcel(user_text, self.memory, session_id)

                    if isinstance(answer, dict):
                        answer_parts.append(answer.get('message', str(answer)))
                        # cognitive → physical 위임 결과에서 robot_command 추출
                        rc = answer.get('robot_command')
                        if rc:
                            import json as _json
                            robot_command_str = _json.dumps(rc, ensure_ascii=False) if isinstance(rc, dict) else str(rc)
                            self.get_logger().info(f"[ROBOT COMMAND] {robot_command_str}")
                        if any(keyword in user_text for keyword in ["가져", "갖다", "와", "찾아", "정리", "꺼내"]):
                            processed_physical = True
                    else:
                        answer_text = str(answer) if answer else ""

                        if not answer_text or answer_text.strip() == "":
                            answer_text = "말씀하신 내용을 기록해두었어요."
                        answer_parts.append(answer_text)

                        self.get_logger().info(f"[COGNITIVE] 응답 메시지: {answer_text}")
                    
                    if "기록해둘게요" in str(answer) or "이해해요" in str(answer):
                        emotion_saved_in_this_turn = True

                elif act_type == "physical" and not processed_physical:

                    from life_assist_dm.support_chains import handle_physical_task
                    try:
                        physical_result = handle_physical_task(user_text, self.memory, session_id)
                        self.get_logger().info(f"[PHYSICAL RESULT] {physical_result}")

                        if isinstance(physical_result, dict):
                            answer_parts.append(physical_result.get('message', str(physical_result)))
                            # robot_command 추출해서 저장
                            rc = physical_result.get('robot_command')
                            if rc:
                                import json as _json
                                robot_command_str = _json.dumps(rc, ensure_ascii=False)
                                self.get_logger().info(f"[ROBOT COMMAND] {robot_command_str}")
                        else:
                            answer_parts.append(str(physical_result))
                            robot_command_str = ""
                    except Exception as e:
                        import traceback
                        tb = traceback.format_exc()
                        self.get_logger().error(f"[PHYSICAL ERROR] {tb}")

                        answer_parts.append("물리적 작업 처리 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요.")

                elif act_type == "emotional":
                    from life_assist_dm.support_chains import build_emotional_reply

                    user_name_confirmed = bool(self.memory.user_names.get(session_id))
                    answer = build_emotional_reply(user_text, self.memory.llm, user_name_confirmed)
                    answer_parts.append(str(answer))

                    if not emotion_saved_in_this_turn:
                        try:
                            user_name_log = self.memory.user_names.get(session_id)
                            if user_name_log and user_name_log != "사용자":

                                from life_assist_dm.support_chains import _extract_emotion_word_and_label
                                emotion_word, emo_label = _extract_emotion_word_and_label(user_text)
                                
                                emotion_to_save = emotion_word if emotion_word else (emo_label if emo_label else "중립")
                                
                                info_summary = self._summarize_emotion_context(user_text)
                                
                                self.memory.excel_manager.save_entity_data(user_name_log, "감정", {
                                    "감정": emotion_to_save,
                                    "정보": info_summary
                                })
                                emotion_saved_in_this_turn = True
                        except Exception as e:
                            self.get_logger().warning(f"감정 기록 저장 실패: {e}")
                    else:
                        self.get_logger().debug("[SKIP] 감정 기록 중복 저장 방지 (cognitive에서 이미 저장됨)")

                elif act_type == "query":
                    from life_assist_dm.support_chains import handle_query_with_lcel
                    answer = handle_query_with_lcel(user_text, self.memory, session_id)
                    answer_parts.append(str(answer))

                else:

                    answer_parts.append("죄송해요, 지금은 그 요청을 처리할 수 없어요.")

            response.success = True

            if not answer_parts:
                answer_parts.append("처리 완료했어요.")

            safe_answer = " ".join(answer_parts)

            safe_answer = _filter_safety_apology(safe_answer)
            safe_answer = safe_answer.replace('"', '＂').replace("'", "＇")
            response.answer = safe_answer
            response.robot_command = robot_command_str
            # robot_command가 있으면 act_type을 physical로 덮어쓰기
            if robot_command_str:
                response.act_type = "physical"
            else:
                response.act_type = ",".join(act_types)

            self.get_logger().info(f"[RESPONSE] 최종 응답: {response.answer}")

            try:
                user_name_log = self.memory.user_names.get(session_id)
                if user_name_log and user_name_log != "사용자":
                    summary_text = f"Q: {user_text} | A: {response.answer}"
                    self.memory.excel_manager.save_conversation_summary(user_name_log, summary_text)
            except Exception as e:
                self.get_logger().warning(f"대화 기록 저장 실패: {e}")
            
            try:

                if hasattr(self.memory.excel_manager, "_buffered_changes"):
                    buffered_changes = self.memory.excel_manager._buffered_changes

                    user_name = self.memory.user_names.get(session_id)
                    if user_name and user_name != "사용자":

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

            self.get_logger().error(f"[ERROR] {tb}")
            response.success = False

            response.answer = "죄송해요, 처리 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요."
            response.act_type = "unknown"

        return response

def main(args=None):

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

        try:
            session_id = "default_session"
            user_name = node.memory.user_names.get(session_id)
            if user_name and user_name != "사용자":

                now = datetime.now()
                timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
                
                memory_vars = node.memory.conversation_memory.load_memory_variables({})
                history = memory_vars.get("history", "")
                
                if history:

                    summary = f"{timestamp}: 세션 종료 - {history[:200] if len(history) > 200 else history}"
                    node.memory.excel_manager.save_conversation_summary(user_name, summary)
                    node.get_logger().info(f"대화 요약 저장 완료: {user_name}")
                
                node.memory.flush_memory_to_excel(session_id)
                node.get_logger().info(f"엑셀 버퍼 플러시 완료: {user_name}")
        except Exception as e:
            node.get_logger().warning(f"엑셀 flush 실패: {e}")
    finally:
        node.destroy_node()

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
