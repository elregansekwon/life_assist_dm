from __future__ import annotations

import warnings
try:
    from langchain_core._api import LangChainDeprecationWarning
    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except Exception:
    pass
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import os
import sqlite3
import json
import hashlib
from datetime import datetime, timedelta
import re
import dateparser
import numpy as np
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from langchain.schema import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine, text


CONFIDENCE_THRESHOLD = 0.6


logger = logging.getLogger(__name__)

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables.config import RunnableConfig
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.schema import StrOutputParser, Document
from langchain_core.output_parsers import JsonOutputParser

from .task_classifier import classify_hybrid, ClassificationResult
from .support_chains import build_emotional_reply, handle_physical_task, handle_pending_answer
from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm, get_embedding
import pandas as pd


@dataclass
class MemoryConfig:
    sqlite_path: str = "~/.life_assist_dm/history.sqlite"
    chroma_dir: str = "~/.life_assist_dm/chroma"
    summary_enabled: bool = True
    retriever_search_k: int = 5
    buffer_window: int = 3
    auto_export_enabled: bool = True
    export_dir: str = "conversation_extract"



NORMALIZE_KEYS = {
    "엄마": "어머니",
    "아빠": "아버지",
    "핸드폰": "휴대폰",
    "핸폰": "휴대폰",
}


STOPWORDS = {
    "한", "번", "쯤", "시쯤", "식후에", "식전에", "반쯤", "시",
    "만", "도", "만큼", "정도", "쯤", "가량", "약", "대략"
}


NAME_BLACKLIST = {
    "오늘", "어제", "내일", "그제", "아침", "점심", "저녁", "밤", "낮",
    "그냥", "그래", "응", "네", "아니", "맞아", "좋아", "싫어",
    "먹었", "먹어", "마셔", "마셨", "갔", "왔", "있", "없",
    "했", "했어", "할", "할게", "할래", "하고", "해서",
    "그", "이", "저", "내", "네", "우리", "너", "나", "저희",
    "뭐라고", "누구게", "뭐게", "뭔가", "뭔지", "뭐지", "뭐야",
    "누구야", "누구지", "누구인지", "누구인가", "누구인가요"
}


MEDICINE_KEYWORDS = [
    "알약", "처방", "복용", "복용법", "복용시간", "식후", "식전", "공복",
    "비타민", "영양제", "오메가3", "오메가 3", "철분제", "프로틴", "보충제",
    "유산균", "프로바이오틱스", "마그네슘", "칼슘", "아연", "엽산"
]


COMMON_MED_PATTERNS = [
    r"(비타민\s*[A-Z]?\d*)", r"(오메가\s*3)", r"(오메가3)", r"(철분제)", r"(프로틴)",
    r"(보충제)", r"(영양제)", r"(유산균)", r"(유산균제)", r"(프로바이오틱스)",
    r"(마그네슘)", r"(칼슘)", r"(아연)", r"(엽산)", r"([가-힣A-Za-z]+)\s*(?:을|를)?\s*먹"
]


METHOD_PATTERNS = [
    r"식후\s*(\d+)\s*분", r"식후\s*(\d+)분", r"식전", r"공복", r"식사\s*후", r"식사\s*전"
]


TIME_OF_DAY_KEYWORDS = {
    "아침": ["아침", "조식", "morning", "breakfast", "기상", "일어나자마자", "일어나자 마자", "기상 후", "기상 시"],
    "점심": ["점심", "중식", "lunch"],
    "저녁": ["저녁", "석식", "dinner", "evening"]
}


MEDICATION_METHOD_KEYWORDS = {
    "공복": ["공복", "공복에"],
    "식전": ["식전", "식전에"],
    "식후": ["식후", "식후에"]
}


KOREAN_NUMBERS_STR = {
    "한": "1", "두": "2", "세": "3", "네": "4", "다섯": "5",
    "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9", "열": "10"
}


KOREAN_NUMBERS_INT = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10
}


class LifeAssistMemory:

    def __init__(self, cfg: MemoryConfig, session_id: str = "default", debug: bool = False):
        self.cfg = cfg
        self.session_id = session_id
        

        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        

        self.time_context = {}
        

        self.emotional_state = {}
        

        

        self.user_names = {}
        

        from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm, get_embedding
        self.llm = get_llm()
        self.debug = debug


        self.sqlite_path = str(Path(os.path.expanduser(cfg.sqlite_path)))
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)


        self.vectorstore = None
        self.vector_store = None
        self.retriever = None
        

        from .user_excel_manager import UserExcelManager
        self.excel_manager = UserExcelManager()
        

        self.classification_cache = {}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.cache_embeddings = None
        self.cache_texts = []
        self.similarity_threshold = 0.95
        

        self.date_cache = {}
        self.max_date_cache_size = 1000


        self.entity_chain = self._build_entity_chain()


        from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
        from langchain_community.chat_message_histories import SQLChatMessageHistory
        

        engine = create_engine(f"sqlite:///{self.sqlite_path}")
        self._ensure_message_table_exists(engine)
        

        self.conversation_memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            chat_memory=SQLChatMessageHistory(
                session_id="default_session",
                connection=engine
            )
        )
        

        from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm, get_embedding
        llm = get_llm()
        self.summary_memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1000,
            return_messages=True,
            memory_key="summary_history"
        )


        self.pending_context: Dict[str, str] = {}
        self.asked_pending: Dict[str, str] = {}
        self.pending_question: Dict[str, dict] = {}
        self.current_question: Dict[str, str] = {}
        self._init_sqlite()


    def _init_sqlite(self):

        conn = sqlite3.connect(self.sqlite_path)
        c = conn.cursor()
        

        c.execute(
            "CREATE TABLE IF NOT EXISTS conversation_summary ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, "
            "summary TEXT, "
            "entity_type TEXT, "
            "content TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        

        c.execute(
            "CREATE TABLE IF NOT EXISTS conversation_summaries ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, "
            "summary_text TEXT NOT NULL, "
            "token_count INTEGER, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        

        c.execute(
            "CREATE TABLE IF NOT EXISTS conversation_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, "
            "message_type TEXT NOT NULL, "
            "message_content TEXT NOT NULL, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        

        c.execute(
            "CREATE TRIGGER IF NOT EXISTS trg_update_summary "
            "AFTER UPDATE ON conversation_summary "
            "BEGIN "
            "UPDATE conversation_summary SET updated_at = CURRENT_TIMESTAMP "
            "WHERE id = NEW.id; "
            "END;"
        )
        

        c.execute(
            "CREATE TABLE IF NOT EXISTS message_store ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, "
            "role TEXT, "
            "content TEXT, "
            "message TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        

        try:
            c.execute("ALTER TABLE message_store ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:

            pass
        

        c.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON conversation_summary(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_message_session_id ON message_store(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_summaries_session_id ON conversation_summaries(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_history_session_id ON conversation_history(session_id)")
        

        try:
            c.execute("ALTER TABLE message_store ADD COLUMN message TEXT")
        except sqlite3.OperationalError:

            pass
        

        try:
            c.execute("ALTER TABLE message_store ADD COLUMN role TEXT")
        except sqlite3.OperationalError:

            pass
        

        try:
            c.execute("ALTER TABLE message_store ADD COLUMN name TEXT")
        except sqlite3.OperationalError:
            pass
        
        try:
            c.execute("ALTER TABLE message_store ADD COLUMN tool_calls_json TEXT")
        except sqlite3.OperationalError:
            pass
        
        try:
            c.execute("ALTER TABLE message_store ADD COLUMN additional_kwargs TEXT")
        except sqlite3.OperationalError:
            pass
        
        conn.commit()
        conn.close()




    def _get_user_entities_from_excel(self) -> List[str]:
        entities = []
        try:
            session_id = "default_session"
            if hasattr(self, "excel_cache"):
                sess = self.excel_cache.get(session_id, {})

                user = sess.get("사용자", [])
                if user and user[0].get("이름"):
                    entities.append(f"사용자 이름: {user[0]['이름']}")

                items = sess.get("물건", [])
                for it in items:
                    if it.get("이름") and it.get("위치"):
                        entities.append(f"{it['이름']}: {it['위치']}")
        except Exception as e:
            print(f"[ERROR] 엔티티 조회 실패: {e}")
        return entities

    

    def get_location(self, target: str, return_dict: bool = False) -> Optional[Union[str, dict]]:
        try:
            session_id = "default_session"

            if hasattr(self, 'excel_cache'):
                items = self.excel_cache.get(session_id, {}).get("물건", [])
                for it in reversed(items):
                    if it.get("이름") == target:
                        if return_dict:
                            place = it.get("장소", "")
                            sub_location = it.get("세부위치", "")
                            if place or sub_location:
                                return {"장소": place, "세부위치": sub_location}
                        elif it.get("위치"):
                            return it.get("위치")

            user_name = self.user_names.get(session_id)
            if not user_name:
                return None
            df = self.excel_manager.load_sheet_data(user_name, "물건위치")
            if df is not None and not df.empty:
                for _, row in df.iloc[::-1].iterrows():
                    name_v = row.get("물건이름", None) or row.get("이름", "")
                    if str(name_v) == target:
                        if return_dict:
                            place = str(row.get("장소", "") or "").strip()
                            sub_location = str(row.get("세부위치", "") or "").strip()

                            if place.lower() in ['nan', 'none', '']:
                                place = ''
                            if sub_location.lower() in ['nan', 'none', '']:
                                sub_location = ''
                            if place or sub_location:
                                return {"장소": place, "세부위치": sub_location}
                        else:
                            loc_v = row.get("위치", "")
                            if str(loc_v).strip() != "":
                                return str(loc_v).strip()

                            place = str(row.get("장소", "") or "").strip()
                            sub_location = str(row.get("세부위치", "") or "").strip()

                            if place.lower() in ['nan', 'none', '']:
                                place = ''
                            if sub_location.lower() in ['nan', 'none', '']:
                                sub_location = ''
                            if place and sub_location:
                                return f"{place} {sub_location}"
                            elif place:
                                return place
                            elif sub_location:
                                return sub_location
            return None
        except Exception as e:
            print(f"[ERROR] get_location 실패: {e}")
            return None

    def save_location(self, item_name: str, location: str, overwrite: bool = True) -> None:
        try:
            session_id = "default_session"
            user_name = self.user_names.get(session_id)
            if not user_name:
                return
            self.excel_manager.save_entity_data(user_name, "물건", {"물건이름": item_name, "위치": location})

            if not hasattr(self, 'excel_cache'):
                self.excel_cache = {}
            session_cache = self.excel_cache.setdefault(session_id, {})
            items = session_cache.setdefault("물건", [])
            if overwrite:

                items = [it for it in items if it.get("이름") != item_name]
                session_cache["물건"] = items
            items.append({"이름": item_name, "위치": location})
        except Exception as e:
            print(f"[ERROR] save_location 실패: {e}")

    def _build_context_for_llm(self, user_input: str, session_id: str) -> str:
        try:
            context_parts = []
            

            try:
                mem_vars = self.conversation_memory.load_memory_variables({})
                history = mem_vars.get("history", "")
                if history:
                    context_parts.append(f"[현재 세션 대화 히스토리]\n{history}")
            except Exception as e:
                print(f"[WARN] LCEL Chain 히스토리 로딩 실패: {e}")
            

            try:
                if hasattr(self, 'excel_cache'):
                    sess = session_id or "default_session"
                    user_data = self.excel_cache.get(sess, {})
                    items = user_data.get("물건", [])
                    if items:
                        lines = [f"- {it.get('이름')}: {it.get('위치')}" for it in items[-5:]]
                        context_parts.append("[저장된 물건 위치]\n" + "\n".join(lines))
            except Exception as e:
                print(f"[WARN] 엑셀 캐시 컨텍스트 구성 실패: {e}")
            

            if context_parts:
                return "\n\n".join(context_parts) + "\n\n"
            else:
                return ""
                
        except Exception as e:
            print(f"[ERROR] 맥락 구성 실패: {e}")
            return ""

    def end_session(self, session_id: str) -> str:
        try:
            print(f"[DEBUG] 세션 종료: {session_id}")
            

            messages = self.conversation_memory.chat_memory.messages
            
            if len(messages) < 2:
                print(f"[DEBUG] 요약할 대화가 부족함: {len(messages)}개 메시지")
                return ""
            

            conversation_text = ""
            for msg in messages:
                if hasattr(msg, 'content'):
                    role = "사용자" if msg.__class__.__name__ == "HumanMessage" else "AI"
                    conversation_text += f"{role}: {msg.content}\n"
            

            from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm, get_embedding
            llm = get_llm()
            
            summary_prompt = f"""
다음 대화 내용을 간단히 요약해주세요. 주요 정보(이름, 위치, 일정 등)와 중요한 대화 내용만 포함해주세요.

대화 내용:
{conversation_text}

요약:
"""
            
            summary = llm.invoke(summary_prompt).content.strip()
            
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO conversation_summary (session_id, summary, created_at, updated_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, summary, datetime.now().isoformat(), datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
            
            return summary
            
        except Exception as e:
            print(f"[ERROR] 세션 종료 처리 실패: {e}")
            return ""

    def handle_duplicate_answer(self, user_input: str, pending_data: dict) -> dict:
        text = (user_input or "").strip().lower()
        

        positive = ["응", "어", "그래", "맞아", "바꿔", "업데이트", "덮어", "새로", "다시", "저장해", "좋아", "네", "예", "추가", "함께", "같이", "둘다", "또", "새로운", "더", "그리고", "또한"]

        negative = ["아니", "아냐", "아닌", "그냥", "놔둬", "유지", "그대로", "안돼", "싫어", "아니요", "아니야", "취소"]
        

        if any(k in text for k in negative):
            return {
                "success": True,
                "duplicate": False,
                "message": "알겠어요. 저장하지 않을게요."
            }
        

        if any(k in text for k in positive):
            entity_type = pending_data.get("entity_type")
            new_data = pending_data.get("new_data", {})
            session_id = pending_data.get("session_id")
            
            try:
                user_name = self.user_names.get(session_id or "default")
                if user_name and user_name != "사용자":
                    self.excel_manager.save_entity_data(user_name, entity_type, new_data)
                    
                    return {
                        "success": True,
                        "duplicate": False,
                        "message": f"네, '{pending_data.get('new', '정보')}'를 저장했어요."
                    }
                else:
                    return {
                        "success": False,
                        "duplicate": False,
                        "message": "사용자 정보가 없어서 저장할 수 없어요."
                    }
            except Exception as e:
                print(f"[ERROR] 엑셀 저장 실패: {e}")
                return {
                    "success": False,
                    "duplicate": False,
                    "message": "저장 중 오류가 발생했어요."
                }
        

        return {
            "success": False,
            "duplicate": True,
            "message": "저장할까요, 아니면 취소할까요? (예/아니오)"
        }

    def _check_duplicate_entity(self, entity_type: str, new_data: dict, session_id: str = None) -> dict:
        try:
            user_name = self.user_names.get((session_id or "default_session"))
            if not user_name:
                return {"is_duplicate": False}


            if entity_type == "사용자" and new_data.get("이름"):
                from .dialog_manager.config.config_loader import get_excel_sheets
                sheets = get_excel_sheets()
                df_kv = self.excel_manager.load_sheet_data(user_name, sheets.get("user_info_kv", "사용자정보KV"))
                if df_kv is not None and not df_kv.empty:
                    for _, row in df_kv.iloc[::-1].iterrows():
                        if str(row.get("키", "")).strip() == "이름":
                            existing = str(row.get("값", "")).strip()
                            if existing:
                                if existing == str(new_data.get("이름")):
                                    return {"is_duplicate": True, "message": f"이미 '{existing}'으로 저장되어 있습니다."}
                                else:
                                    return {"is_duplicate": True, "message": f"이미 '{existing}'으로 저장되어 있어요. '{new_data.get('이름')}'로 바꿀까요?"}
                            break
                return {"is_duplicate": False}


            if entity_type == "물건":
                df = self.excel_manager.load_sheet_data(user_name, "물건위치")
                if df is not None and not df.empty:
                    new_name = new_data.get("이름") or new_data.get("물건이름")
                    new_loc = str(new_data.get("위치", "")).strip()
                    for _, row in df.iterrows():
                        name_v = str(row.get("물건이름", "")).strip() or str(row.get("이름", "")).strip()
                        loc_v = str(row.get("위치", "")).strip()
                        if name_v and name_v == new_name:
                            if loc_v == new_loc:
                                return {"is_duplicate": True, "message": f"이미 '{name_v}'이(가) '{loc_v}'에 저장되어 있습니다."}
                            else:
                                return {"is_duplicate": True, "message": f"'{name_v}'이(가) 이미 '{loc_v}'에 저장되어 있어요. '{new_loc}'로 바꿀까요?"}
            return {"is_duplicate": False}
        except Exception as e:
            print(f"[ERROR] 엑셀 기반 중복 확인 실패: {e}")
            return {"is_duplicate": False}

    def _delete_existing_entity(self, entity_type: str, existing_value: str):
        try:
            user_name = self.user_names.get("default_session")
            if not user_name:
                return
            if entity_type == "물건":
                df = self.excel_manager.load_sheet_data(user_name, "물건위치")
                if df is not None and not df.empty:

                    filtered_records = []
                    for _, row in df.iterrows():
                        if str(row.get("물건이름", "")) != str(existing_value):
                            filtered_records.append(row.to_dict())

                    if filtered_records:
                        for record in filtered_records:
                            self.excel_manager.save_entity_data(user_name, "물건", record)

                        self.excel_manager.request_flush(user_name)
        except Exception as e:
            print(f"[WARN] 엑셀 엔티티 삭제 실패: {e}")

    def _check_missing_fields(self, entity_type: str, data: dict) -> dict:
        required_fields = {
            "user.약": ["약명"],  
            "user.식사": ["끼니"],
            "user.일정": ["날짜", "시간", "제목"],
            "user.사용자": ["이름"]
        }

        missing = []
        for field in required_fields.get(entity_type, []):
            if not data.get(field):
                missing.append(field)

        if missing:
            return {
                "has_missing": True,
                "message": f"{entity_type} 엔티티 저장에 필요한 정보({', '.join(missing)})가 빠졌습니다. 알려주실 수 있나요?",
                "missing_fields": missing
            }

        return {"has_missing": False}

    def _extract_medicine_entities(self, text: str) -> list:
        medicines = []
        
        
        med_match = re.search(r"([가-힣A-Za-z]+약)", text)
        if med_match:
            med_name = med_match.group(1)
            dose_match = re.search(rf"{re.escape(med_name)}.*?(\d+)\s*(알|정|캡슐|포|mg|ml|병)(?:씩)?(?!\s*(?:분|시간|시))", text)
            
            if dose_match:

                match_pos = dose_match.start()
                number_pos = text.find(dose_match.group(1), match_pos)
                before_number = text[:number_pos]
                


                if re.search(r"(식후|식전|공복|식사\s*(?:전|후))\s*\d+\s*(?:분|시간|시)", before_number + dose_match.group(0)):

                    medicines.append({
                        "이름": med_name,
                        "용량": "",
                        "단위": ""
                    })
                else:

                    medicines.append({
                        "이름": med_name,
                        "용량": dose_match.group(1).strip(),
                        "단위": dose_match.group(2)
                    })
            else:


                korean_dose_match = re.search(rf"{re.escape(med_name)}.*?({'|'.join(KOREAN_NUMBERS_STR.keys())})\s*(알|정|캡슐|포|병)?", text)
                if korean_dose_match:
                    korean_num = korean_dose_match.group(1)
                    unit = korean_dose_match.group(2) if korean_dose_match.group(2) else "알"
                    medicines.append({
                        "이름": med_name,
                        "용량": KOREAN_NUMBERS_STR.get(korean_num, "1"),
                        "단위": unit
                    })
                else:
                    medicines.append({
                        "이름": med_name,
                        "용량": "",
                        "단위": ""
                    })
        else:

            med_match = re.search(r"([가-힣A-Za-z]+)\s+약", text)
            if med_match:
                med_name = med_match.group(1).strip()
                medicines.append({
                    "이름": med_name,
                    "용량": "",
                    "단위": ""
                })
            else:


                pattern = r"([가-힣A-Za-z]+)\s*(\d+)\s*(알|정|캡슐|포|mg|ml|병)?"
                matches = re.findall(pattern, text)

                for match in matches:
                    name, dose, unit = match

                    if name not in ["하루", "식후", "식전", "아침", "점심", "저녁"]:
                        medicines.append({
                            "이름": name.strip(),
                            "용량": dose.strip(),
                            "단위": unit if unit else "알"
                        })
                
                if not medicines:
                    for pattern in COMMON_MED_PATTERNS:
                        med_match = re.search(pattern, text, re.IGNORECASE)
                        if med_match:
                            med_name = med_match.group(1).strip()
                            med_name = med_name.replace("을", "").replace("를", "").strip()
                            medicines.append({
                                "이름": med_name,
                                "용량": "",
                                "단위": ""
                            })
                            break

        return medicines

    def _extract_meal_entity(self, text: str) -> dict:

        from datetime import datetime, timedelta
        
        meal = {"날짜": None, "끼니": None, "메뉴": []}

        
        try:
            has_med_keyword = any(keyword in text for keyword in MEDICINE_KEYWORDS)

            if not has_med_keyword and "약" in text and "약속" not in text:

                if re.search(r"[가-힣A-Za-z]+약|약\s*[먹드]", text):
                    has_med_keyword = True

            if not has_med_keyword and len(text) < 200:
                if re.search(r"[가-힣A-Za-z]+약(?!속)", text):
                    has_med_keyword = True
            if has_med_keyword:
                return meal
        except Exception as e:
            print(f"[WARN] 약 키워드 체크 중 오류: {e}")


        for time_key, keywords in TIME_OF_DAY_KEYWORDS.items():
            if any(k in text for k in keywords):
                meal["끼니"] = time_key
                break


        if "어제" in text:
            meal["날짜"] = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "오늘" in text:
            meal["날짜"] = datetime.now().strftime("%Y-%m-%d")
        elif "내일" in text:
            meal["날짜"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "모레" in text:
            meal["날짜"] = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")



        food_pattern = r"(?:\s|^)([가-힣A-Za-z]+)\s*(먹었|먹음|먹다|먹을)"
        matches = re.findall(food_pattern, text)
        if matches:
            meal["메뉴"] = [m[0] for m in matches]

        return meal

    def extract_with_fallback(self, text: str, entity_type: str):
        if entity_type == "약":
            meds = self._extract_medicine_entities(text)
            if meds:
                return meds
        elif entity_type == "식사":
            meal = self._extract_meal_entity(text)
            if meal.get("끼니") or meal.get("메뉴"):
                return meal


        try:

            if hasattr(self, 'llm_chain') and self.llm_chain:
                result = self.llm_chain.invoke({"input": text, "entity_type": entity_type})
                return result
            else:
                print(f"[WARN] LLM 체인이 없어서 fallback 불가: {entity_type}")
                return None
        except Exception as e:
            print(f"[WARN] LLM fallback 실패: {e}")
            return None

    def _get_recent_conversation_summary(self, session_id: str) -> str:
        try:
            conn = sqlite3.connect(self.sqlite_path)
            c = conn.cursor()
            

            c.execute("""
                SELECT summary FROM conversation_summary 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT 3
            """, (session_id,))
            
            summaries = c.fetchall()
            conn.close()
            
            if summaries:

                recent_summaries = [summary[0] for summary in reversed(summaries)]
                return "\n".join(recent_summaries)
            else:
                return ""
                
        except Exception as e:
            print(f"[ERROR] 대화 요약 불러오기 실패: {e}")
            return ""

    def _convert_conversation_history_to_string(self, conversation_history) -> str:
        if isinstance(conversation_history, str):

            if len(conversation_history) > 2000:
                return conversation_history[-2000:] + "..."
            return conversation_history
        elif isinstance(conversation_history, list):

            history_text = ""
            recent_messages = conversation_history[-10:]
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    history_text += f"{msg.content}\n"
                else:
                    history_text += f"{str(msg)}\n"
            

            if len(history_text) > 2000:
                return history_text[-2000:] + "..."
            return history_text.strip()
        else:
            return str(conversation_history) if conversation_history else ""

    def _get_current_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()





    def save_entity_to_vectorstore(self, entity_type: str, data: dict, session_id: str = None) -> dict:
        import json
        from datetime import datetime
        
        try:

            print(f"[DEBUG] 중복 확인 시작: entity_type={entity_type}, data={data}")
            duplicate_info = self._check_duplicate_entity(entity_type, data, session_id)
            print(f"[DEBUG] 중복 확인 결과: {duplicate_info}")

            if duplicate_info.get("is_duplicate"):

                return {
                    "success": False,
                    "duplicate": True,
                    "existing": duplicate_info.get("existing"),
                    "new": duplicate_info.get("new"),
                    "message": duplicate_info.get("message"),
                    "pending_data": {
                        "entity_type": entity_type,
                        "new_data": data,
                        "existing": duplicate_info.get("existing"),
                        "new": duplicate_info.get("new"),
                        "session_id": session_id
                    }
                }



            if entity_type == "물건":
                doc = {
                    "type": "물건",
                    "이름": data.get("이름"),
                    "위치": data.get("위치"),
                    "출처": "사용자 발화",
                    "session_id": session_id or "default",
                    "timestamp": datetime.now().isoformat()
                }
            elif entity_type == "감정" or entity_type == "정서":
                doc = {
                    "type": "정서",
                    "감정": data.get("상태") or data.get("감정") or data.get("증상"),
                    "강도": data.get("강도") or data.get("정도", "보통"),
                    "날짜": data.get("날짜", datetime.now().strftime("%Y-%m-%d")),
                    "출처": "사용자 발화",
                    "session_id": session_id or "default",
                    "timestamp": datetime.now().isoformat()
                }
            elif entity_type == "사용자":
                doc = {
                    "type": "사용자",
                    "이름": data.get("이름"),
                    "확인됨": data.get("확인됨", True),
                    "출처": "사용자 발화",
                    "session_id": session_id or "default",
                    "timestamp": datetime.now().isoformat()
                }
            elif entity_type == "일정":
                doc = {
                    "type": "일정",
                    "제목": data.get("제목"),
                    "날짜": data.get("날짜"),
                    "시간": data.get("시간"),
                    "출처": "사용자 발화",
                    "session_id": session_id or "default",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                doc = {
                    "type": entity_type,
                    **data,
                    "출처": "사용자 발화",
                    "session_id": session_id or "default",
                    "timestamp": datetime.now().isoformat()
                }
            

            try:
                user_name = self.user_names.get(session_id or "default")
                

                if user_name and user_name != "사용자" and len(user_name.strip()) > 0:

                    import re
                    if not re.match(r'^[가-힣A-Za-z0-9\s]+$', user_name) or len(user_name) > 20:
                        print(f"[WARNING] 유효하지 않은 사용자 이름: {user_name}")
                        user_name = None
                
                if user_name and user_name != "사용자":

                    self.excel_manager.save_entity_data(user_name, entity_type, data)
                    print(f"[DEBUG] 엔티티 버퍼링 완료: {user_name} - {entity_type}")
                else:
                    print(f"[WARNING] 사용자 이름 없음 또는 유효하지 않음 - 엔티티 저장 건너뜀")
            except Exception as excel_error:
                print(f"[WARNING] 엑셀 저장 실패: {excel_error}")
                
                return {
                    "success": True,
                    "duplicate": False,
                    "message": f"{entity_type} 엔티티가 저장되었습니다."
                }
            except Exception as e:
                print(f"[ERROR] save_entity_to_vectorstore 실패: {e}")
                return {
                    "success": False,
                    "duplicate": False,
                    "message": "엔티티 저장 중 오류가 발생했어요."
                }
        except Exception as e:
            print(f"[ERROR] save_entity_to_vectorstore 전체 실패: {e}")
            return {
                "success": False,
                "duplicate": False,
                "message": "엔티티 저장 중 오류가 발생했어요."
            }


    def _history(self, session_id: str) -> SQLChatMessageHistory:

        engine = create_engine(f"sqlite:///{self.sqlite_path}")
        
        self._ensure_message_table_exists(engine)
        
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=engine
        )
    
    def _ensure_message_table_exists(self, engine):
        with engine.connect() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS message_store (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT,
                    content TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            conn.commit()


    def _get_recent_summaries(self, session_id: str, limit: int = 3) -> List[str]:
        conn = sqlite3.connect(self.sqlite_path)
        c = conn.cursor()
        c.execute(
            "SELECT summary FROM conversation_summary "
            "WHERE session_id=? ORDER BY updated_at DESC LIMIT ?",
            (session_id, limit),
        )
        rows = c.fetchall()
        conn.close()
        return [row[0] for row in rows] if rows else []


    def _safe_parse_json(self, text: Any) -> Dict[str, Any]:
        if isinstance(text, dict):
            return text
        if text is None:
            return {}
        s = str(text).strip()
        try:
            return json.loads(s)
        except Exception:
            pass
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(s[start:end + 1])
            except Exception:
                return {}
        return {}


    def _normalize_datetime(self, text: str) -> Dict[str, Optional[str]]:
        if not text:
            return {"날짜": None, "시간": None, "식전후": None}

        t = re.sub(r"(쯤|경|약)", "", str(text).strip())
        rel_date = None

        if "그제" in t:
            rel_date = (datetime.now() - timedelta(days=2))
        elif "어제" in t:
            rel_date = (datetime.now() - timedelta(days=1))
        elif "내일" in t:
            rel_date = (datetime.now() + timedelta(days=1))
        elif "모레" in t:
            rel_date = (datetime.now() + timedelta(days=2))
        elif "오늘" in t:
            rel_date = datetime.now()


        m = re.search(r"(오전|오후|저녁|밤)?\s*(\d{1,2})\s*시(?:\s*(반|(\d{1,2})\s*분))?", t)
        if m:
            part = m.group(1)
            hour = int(m.group(2))
            mm = 30 if (m.group(3) == "반") else (int(m.group(4)) if m.group(4) else 0)

            if part in ("오후", "저녁") and hour < 12:
                hour += 12
            elif part == "밤" and hour < 12:
                hour += 12
            elif part == "오전" and hour == 12:
                hour = 0

            return {
                "날짜": (rel_date or datetime.now()).strftime("%Y-%m-%d"),
                "시간": f"{hour:02d}:{mm:02d}:00",
                "식전후": None
            }

        return {
            "날짜": rel_date.strftime("%Y-%m-%d") if rel_date else None,
            "시간": None,
            "식전후": None
        }


    def _build_entity_chain(self) -> Runnable:
        parser = JsonOutputParser()
        fmt = parser.get_format_instructions()
        prompt = ChatPromptTemplate.from_template(
            (
                "당신은 엔티티 추출기입니다.\n"
                "아래 '사용자 발화'에서 언급된 사실만을 단일 JSON 객체로 추출하세요.\n\n"
                "**엔티티 추출기 역할:**\n"
                "1. 가능한 한 엔티티 JSON으로 완성해라.\n"
                "2. 만약 필수 정보가 빠졌거나 애매하면, JSON 대신 '질문' 키를 만들어 사용자에게 물어볼 질문을 생성해라.\n\n"
                "중요 규칙:\n"
                "1) 발화에 직접 등장하지 않은 날짜/시간/장소/이름/수치는 절대 추측하지 말고 생략하거나 null로 둡니다.\n"
                "2) 엔티티 키는 아래 허용 엔티티 중에서만 사용합니다.\n"
                "3) JSON 외 텍스트는 금지합니다. 아무것도 없으면 빈 객체 {{}} 만 출력합니다.\n"
                "4) 필수 정보가 부족한 경우: {{\"질문\": \"구체적인 질문 내용\"}} 형태로 출력하세요.\n"
                "5) 약 복용 정보(복용 횟수, 시간, 방법)는 실제로 언급된 경우에만 추출하고, 언급되지 않으면 절대 추측하지 마세요!\n"
                "6) '혈압약이랑 비염약을 받아왔어' → 복용 정보가 없으므로 복용 필드를 생략하세요!\n\n"
                "허용 엔티티 유형: [\"사용자\",\"약\",\"식사\",\"가족\",\"물건\",\"일정\",\"기념일\",\"취미\",\"취향\",\"건강상태\"]\n"
                "**새로운 관계나 엔티티 타입이 필요하면 자유롭게 생성하세요!**\n"
                "- 예: \"남자친구\", \"여자친구\", \"동료\", \"선생님\", \"친구\", \"이웃\" 등\n"
                "- 새로운 관계 엔티티는 관계와 이름 필드를 포함하여 저장\n"
                "- 기존 가족 엔티티는 관계와 이름 필드를 포함하여 저장\n\n"
                "**엔티티 타입 생성 규칙 (중요!):**\n"
                "- \"남자친구\", \"여자친구\" → user.남자친구, user.여자친구\n"
                "- \"이모\", \"고모\", \"삼촌\", \"외삼촌\" → user.가족\n"
                "- \"동료\", \"선생님\", \"친구\" → user.동료, user.선생님, user.친구\n"
                "- **절대 user.user.xxx 형식을 사용하지 말 것**\n"
                "- **올바른 형식: user.남자친구, user.가족, user.동료**\n"
                "- **잘못된 형식: user.user.남자친구, user.user.가족, user.user.동료**\n\n"
                "**예시:**\n"
                "- \"내 남자친구 이름은 안건호야\" → user.남자친구 엔티티로 생성\n"
                "- \"우리 이모 이름은 송혜교야\" → user.가족 엔티티로 생성\n"
                "- \"우리 동료 김철수야\" → user.동료 엔티티로 생성\n\n"
                "중요: 약과 식사를 구분하세요!\n"
                "- 약: 혈압약, 비염약, 감기약, 아스피린, 타이레놀, 유산균, 유산균제, 프로바이오틱스 등 의약품 복용\n"
                "- 식사: 햄버거, 김치찌개, 불고기, 라면, 밥, 빵, 과일 등 음식 섭취\n"
                "- '약 먹었어', '혈압약 먹었어', '비염약 먹었어' 등은 약 복용이지 식사가 아닙니다!\n"
                "- 약명(약으로 끝나는 단어)이 포함된 문장은 무조건 약 엔티티로만 추출하세요!\n"
                "- 약을 식사로 착각하지 마세요! 약은 약 엔티티로만 처리하세요!\n"
                "- '나는 혈압약을 먹어' → user.약 엔티티 (절대 user.식사 아님!)\n"
                "- '나는 햄버거를 먹었어' → user.식사 엔티티\n\n"
                "**약명 추출 규칙 (매우 중요!):**\n"
                "- 약명은 발화에서 명시적으로 언급된 약의 이름을 정확히 추출하세요!\n"
                "- '유산균을 먹어야해' → 약명: '유산균' (절대 '일어나자마자', '아침' 등이 약명이 아님!)\n"
                "- '혈압약을 먹었어' → 약명: '혈압약'\n"
                "- '비염약을 복용했어' → 약명: '비염약'\n"
                "- '불면증약을 먹어야해' → 약명: '불면증약'\n"
                "- **절대 금지**: 시간 표현('아침', '일어나자마자', '기상'), 복용 방법('공복', '식후'), 복용 횟수('하루 3번') 등을 약명으로 추출하지 마세요!\n"
                "- **절대 금지**: '일어나자마자'는 약명이 아니라 복용 시간 표현입니다! 실제 약명을 찾아주세요!\n\n"
                "**약 복용 시간 추출 규칙 (중요!):**\n"
                "- '아침에', '아침으로', '기상 시', '일어나자마자', '일어나자 마자', '기상 후' → 시간대: '아침' 또는 복용: [{{\"원문\": \"아침\"}}]\n"
                "- '점심에', '점심으로' → 시간대: '점심' 또는 복용: [{{\"원문\": \"점심\"}}]\n"
                "- '저녁에', '저녁으로' → 시간대: '저녁' 또는 복용: [{{\"원문\": \"저녁\"}}]\n"
                "- '공복에', '식전에' → 복용방법: '공복' 또는 '식전'\n"
                "- '식후에', '식후 30분' → 복용방법: '식후 30분' (정확한 시간이 있으면 포함)\n"
                "- '하루 3번' → 복용: [{{\"원문\": \"하루 3번\"}}] (시간대는 생략 가능)\n"
                "- **예시**: '유산균을 아침에 일어나자마자 1알 공복에 먹어야해' → 약명: '유산균', 용량: '1', 단위: '알', 시간대: '아침', 복용방법: '공복'\n\n"
                "식사 끼니 추출 규칙 (중요!):\n"
                "- '아침에', '아침으로' → 끼니: '아침'\n"
                "- '점심에', '점심으로' → 끼니: '점심'\n"
                "- '저녁에', '저녁으로' → 끼니: '저녁'\n"
                "- 시간으로 끼니 추론: 6-11시 → 아침, 11-15시 → 점심, 15-22시 → 저녁\n"
                "- 명시적으로 끼니가 언급되지 않으면 끼니 필드를 null로 두세요!\n"
                "- '아침에 밥 먹었어' → 끼니: '아침' (절대 '저녁'이 아님!)\n\n"
                "**사용자 이름 및 별명 추출 규칙 (중요!):**\n"
                "- '내 이름은 홍길동이야' → {{\"user.사용자\": [{{\"이름\": \"홍길동\"}}]}}\n"
                "- '내 별명은 사유리야' → {{\"user.사용자\": [{{\"별명\": \"사유리\"}}]}}\n"
                "- '내 별명은 사유리라고 해' → {{\"user.사용자\": [{{\"별명\": \"사유리\"}}]}}\n"
                "- '편하게 서연이라고 불러' → {{\"user.사용자\": [{{\"별명\": \"서연\"}}]}}\n"
                "- '나는 홍길동이고, 편하게 길동이라고 불러도 돼' → {{\"user.사용자\": [{{\"이름\": \"홍길동\", \"별명\": \"길동\"}}]}}\n"
                "- 별명은 '별명', '별칭', 'alias' 중 하나의 키로 저장 가능\n"
                "- '내 별명은', '별명은', '별칭은', '편하게 ~라고 불러' 등의 패턴을 모두 인식해야 함\n\n"
                "필드 예시:\n"
                "- 사용자: {{\"이름\": \"홍길동\"}} 또는 {{\"별명\": \"사유리\"}} 또는 {{\"이름\": \"홍길동\", \"별명\": \"길동\"}}\n"
                "- 일정: {{\"제목\": \"병원 예약\", \"날짜\": \"내일\", \"시간\": \"오후 3시\", \"장소\": null}}\n"
                "- 약: {{\"약명\": \"혈압약\", \"복용\": [{{\"원문\": \"하루 두 번\"}}], \"복용 기간\": \"일주일치\"}} (복용 정보는 실제 언급된 경우만!)\n"
                "- 약 (복용 정보 없음): {{\"약명\": \"혈압약\"}} (복용 정보가 언급되지 않으면 생략!)\n"
                "- 식사: {{\"끼니\": \"점심\", \"메뉴\": [\"햄버거\"], \"날짜\": \"오늘\", \"시간\": \"12:30\"}}\n"
                "- 기념일: {{\"관계\": \"사용자\", \"제목\": \"생일\", \"날짜\": \"4월 7일\"}}\n"
                "- 건강상태: {{\"증상\": \"두통\", \"정도\": \"심함\", \"기간\": \"3일\", \"질병\": \"당뇨\"}}\n"
                "- 물건: {{\"이름\": \"열쇠\", \"위치\": \"거실 책상 위에\", \"장소\": \"거실 책상\", \"세부위치\": \"위에\"}}\n"
                "- 물건: {{\"이름\": \"안약\", \"위치\": \"내방 안에\", \"장소\": \"내방\", \"세부위치\": \"안에\"}}\n"
                "- 물건: {{\"이름\": \"지갑\", \"위치\": \"침실 옆에\", \"장소\": \"침실\", \"세부위치\": \"옆에\"}}\n"
                "- 물건 (위치만): {{\"이름\": \"펜\", \"위치\": \"책상\", \"장소\": \"책상\", \"세부위치\": \"\"}}\n\n"
                "**물건 위치 추출 규칙 (중요!):**\n"
                "- '안약은 내방 안에 있어' → {{\"user.물건\": [{{\"이름\": \"안약\", \"위치\": \"내방 안에\", \"장소\": \"내방\", \"세부위치\": \"안에\"}}]}}\n"
                "- '열쇠는 거실 책상 위에 있어' → {{\"user.물건\": [{{\"이름\": \"열쇠\", \"위치\": \"거실 책상 위에\", \"장소\": \"거실 책상\", \"세부위치\": \"위에\"}}]}}\n"
                "- '지갑은 침실에 있어' → {{\"user.물건\": [{{\"이름\": \"지갑\", \"위치\": \"침실\", \"장소\": \"침실\", \"세부위치\": \"\"}}]}}\n"
                "- 위치 표현에서 \"위에\", \"안에\", \"옆에\", \"앞에\", \"뒤에\", \"아래에\" 같은 방향 표현은 반드시 \"세부위치\" 필드로 추출하세요!\n"
                "- \"장소\" 필드는 방향 표현을 제외한 나머지 부분입니다 (예: \"내방 안에\" → 장소=\"내방\", 세부위치=\"안에\")\n\n"
                "**일정 추출 강화 (중요!):**\n"
                "- '내일 오후 3시에 병원 예약이 있어' → {{\"user.일정\": [{{\"제목\": \"병원 예약\", \"날짜\": \"내일\", \"시간\": \"오후 3시\"}}]}}\n"
                "- '다음 주 금요일에 회의가 있어' → {{\"user.일정\": [{{\"제목\": \"회의\", \"날짜\": \"다음 주 금요일\"}}]}}\n"
                "- '12월 25일에 크리스마스 파티가 있어' → {{\"user.일정\": [{{\"제목\": \"크리스마스 파티\", \"날짜\": \"12월 25일\"}}]}}\n"
                "- '10월 5일에 추석 여행 일정있어' → {{\"user.일정\": [{{\"제목\": \"추석 여행 일정\", \"날짜\": \"10월 5일\"}}]}}\n"
                "- '오늘 저녁 7시에 친구 만나기로 했어' → {{\"user.일정\": [{{\"제목\": \"친구 만나기\", \"날짜\": \"오늘\", \"시간\": \"저녁 7시\"}}]}}\n"
                "- **절대 중요**: 날짜 필드는 '10월 5일', '12월 25일', '내일', '다음 주 금요일' 등의 시간 표현만 들어가야 합니다!\n"
                "- **절대 중요**: 제목 필드는 '추석 여행', '크리스마스 파티', '회의', '병원 예약' 등의 일정 내용만 들어가야 합니다!\n"
                "- **절대 금지**: 날짜와 제목을 바꿔서 추출하지 마세요! '10월 5일'은 날짜이고, '추석 여행'은 제목입니다!\n"
                "- 날짜/시간이 명시되지 않아도 제목만 있으면 일정으로 추출\n\n"
                "{format_instructions}\n\n"
                "[사용자 발화]\n"
                "{utterance}"
            )
        ).partial(format_instructions=fmt)
        return prompt | self.llm | parser

    def _extract_item_location_rule(self, user_input: str) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        t = user_input.strip()
        

        sentences = [s.strip() for s in t.split(',') if s.strip()]
        
        for sentence in sentences:

            location_patterns = [

                r"(?:내|네|이|그)\s*(.+?)\s*(?:은|는)\s*(.+?)\s*(?:안에|위에|밖에|옆에|앞에|뒤에|아래에|에)\s*(?:있어|있고|있어요|있습니다|둬|놔|두|놓|보관)",

                r"(.+?)\s*(?:은|는)\s*(.+?에|위에|안에|밖에|옆에|앞에|뒤에|아래에)\s*(?:있|둬|놔|두|놓|보관)",

                r"(.+?)\s*(?:에|위에|안에|밖에|옆에|앞에|뒤에|아래에)\s*(.+?)\s*(?:이|가)?\s*(?:있어|있고|있어요|있습니다)",

                r"(.+?)\s*(.+?에|위에|안에|밖에|옆에|앞에|뒤에|아래에)\s*(?:에|에서)\s*(?:있|둬|놔|두|놓|보관)",

                r"(.+?)\s*(?:을|를)\s*(.+?에|위에|안에|밖에|옆에|앞에|뒤에|아래에)\s*(?:두었|놓았|보관했)",

                r"(.+?)\s*(?:은|는)\s*(.+?에|위에|안에|밖에|옆에|앞에|뒤에|아래에)\s*(?:있고|있어)",

                r"(.+?)\s*(?:은|는)\s*(.+?에)\s*(?:있고|있어)",
            ]
            
            for pattern in location_patterns:
                m = re.search(pattern, sentence)
                if m:

                    groups = m.groups()
                    if len(groups) >= 2:

                        g1, g2 = groups[0].strip(), groups[1].strip()
                        

                        location_keywords = ["에", "위에", "안에", "밖에", "옆에", "앞에", "뒤에", "아래에", "주방", "거실", "침실", "찬장", "서랍", "책상", "방"]
                        
                        if any(kw in g2 for kw in location_keywords):

                            item = re.sub(r"^(내|네|이|그)\s*", "", g1)
                            location = g2
                        elif any(kw in g1 for kw in location_keywords):

                            location = g1
                            item = re.sub(r"^(내|네|이|그)\s*", "", g2)
                        else:

                            item = re.sub(r"^(내|네|이|그)\s*", "", g1)
                            location = g2
                        

                        if (item and location and 
                            len(item) >= 1 and len(location) >= 2 and
                            item not in ["것", "거", "이것", "그것", "저것"]):
                            


                            direction_keywords = ["위에", "옆에", "앞에", "뒤에", "아래에", "안에", "밖에", "에"]
                            
                            place = None
                            sub_location = ""
                            


                            direction_found = None
                            for direction in ["위에", "옆에", "앞에", "뒤에", "아래에", "안에", "밖에"]:
                                if location.endswith(direction):
                                    direction_found = direction
                                    location_without_direction = location[:-len(direction)].strip()
                                    break
                            

                            if not direction_found and location.endswith("안"):
                                direction_found = "안"
                                location_without_direction = location[:-1].strip()
                            

                            if direction_found:



                                place = location_without_direction
                                sub_location = direction_found
                            elif location.endswith("에"):

                                place = location[:-1].strip()
                                sub_location = "에"
                            else:


                                place = location
                                sub_location = ""
                            
                            out.setdefault("user.물건", []).append({
                                "이름": item,
                                "위치": location,
                                "장소": place,
                                "세부위치": sub_location,
                                "추출방법": "rule-based"
                            })
                            break
        
        return out

    def _extract_item_command_rule(self, user_input: str) -> Dict[str, List[Dict[str, Any]]]:
        out: Dict[str, List[Dict[str, Any]]] = {}
        t = user_input.strip()
        

        command_patterns = [

            r"(.+?에서|에)\s*(.+?)\s*(?:꺼내와|가져와|꺼내다|가져다|꺼내줘|가져다줘)",

            r"(.+?)\s*(?:꺼내와|가져와|꺼내다|가져다|꺼내줘|가져다줘)",

            r"(.+?)\s*(?:찾아줘|찾아다|찾아)",

            r"(.+?)\s*(?:어디|위치).*?(?:있어|있나)",
        ]
        
        for pattern in command_patterns:
            matches = re.findall(pattern, t)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        location, item = match

                        if location and item:
                            item = item.strip()
                            location = location.strip()
                            if (len(item) >= 1 and len(location) >= 2 and
                                item not in ["것", "거", "이것", "그것", "저것"]):
                                out.setdefault("user.물건", []).append({
                                    "이름": item,
                                    "위치": location,
                                    "추출방법": "command-rule"
                                })
                    else:

                        item = match[0] if match else ""
                        if item and len(item) >= 1 and item not in ["것", "거", "이것", "그것", "저것"]:
                            out.setdefault("user.물건", []).append({
                                "이름": item,
                                "위치": None,
                                "추출방법": "command-rule"
                            })
                else:

                    item = match.strip()
                    if item and len(item) >= 1 and item not in ["것", "거", "이것", "그것", "저것"]:
                        out.setdefault("user.물건", []).append({
                            "이름": item,
                            "위치": None,
                            "추출방법": "command-rule"
                        })
        
        return out


    def _rule_based_extract(self, text: str, session_id: str = None) -> Dict[str, Any]:

        out: Dict[str, Any] = {}
        groups = []
        try:
            t = text.strip() if text else ""
            if not t:
                return out


            if re.search(r"\?$", t):
                return out



            has_medicine_keyword = False
            try:
                has_medicine_keyword = any(keyword in t for keyword in MEDICINE_KEYWORDS)

                if not has_medicine_keyword and "약" in t and "약속" not in t:

                    if re.search(r"[가-힣A-Za-z]+약|약\s*[먹드]", t):
                        has_medicine_keyword = True

                if not has_medicine_keyword and len(t) < 200:
                    if re.search(r"[가-힣A-Za-z]{1,20}약(?!속)", t):
                        has_medicine_keyword = True
            except Exception as e:
                print(f"[WARN] 약 키워드 체크 중 오류: {e}")
            

            has_medicine_pattern = False
            try:

                if "약속" not in t:

                    has_medicine_pattern = bool(re.search(r"약.*?먹|약.*?복용|복용", t)) or (has_medicine_keyword and "먹" in t)
                    

                    if not has_medicine_pattern:

                        if re.search(r"\d+\s*알.*?먹|\d+\s*알.*?(공복|식전|식후)|[한두세네다섯]\s*알.*?먹", t):
                            has_medicine_pattern = True

                        elif re.search(r"(공복|식전|식후|복용)", t) and re.search(r"\d+\s*알|[한두세네다섯]\s*알", t):
                            has_medicine_pattern = True
                else:

                    has_medicine_pattern = False
            except Exception as e:
                print(f"[WARN] 약 패턴 체크 중 오류: {e}")
            

            has_food_keyword = False
            try:
                food_keywords = r"(밥|식사|음식|요리|메뉴|김치|찌개|국|탕|면|라면|치킨|피자|햄버거|떡볶이|삼겹살|갈비)"
                has_food_keyword = bool(re.search(food_keywords, t))
            except Exception as e:
                print(f"[WARN] 식사 키워드 체크 중 오류: {e}")
            
            if has_medicine_pattern and not has_food_keyword:

                try:
                    print(f"[DEBUG] 약 복용 패턴 매칭: {t}")
                    medicines = self._extract_medicine_entities(t)
                    print(f"[DEBUG] _extract_medicine_entities 결과: {medicines}")
                    

                    time_keywords = {
                        "아침": "아침", "점심": "점심", "저녁": "저녁", 
                        "밤": "밤", "새벽": "새벽", "오전": "오전", "오후": "오후",
                        "자기 전": "밤", "자기전": "밤", "취침 전": "밤", "취침전": "밤",
                        "잠들기 전": "밤", "잠들기전": "밤"
                    }
                    
                    time_of_day_list = []
                    for keyword, time in time_keywords.items():
                        if keyword in t:
                            time_of_day_list.append(time)
                    


                    if not time_of_day_list:

                        frequency_match = re.search(r"하루\s*(?:에\s*)?(\d+|[한두세네다섯])\s*번", t)
                        if frequency_match:
                            freq_str = frequency_match.group(1)

                            if freq_str.isdigit():
                                frequency = int(freq_str)
                            else:
                                frequency = KOREAN_NUMBERS_INT.get(freq_str, 0)
                            


                            if frequency == 1:

                                time_of_day_list = []
                            elif frequency == 2:
                                time_of_day_list = ["아침", "저녁"]
                            elif frequency == 3:
                                time_of_day_list = ["아침", "점심", "저녁"]
                            elif frequency >= 4:

                                time_of_day_list = ["아침", "점심", "저녁", "밤"]
                    

                    if time_of_day_list:
                        time_of_day = "/".join(time_of_day_list)
                    else:
                        time_of_day = None
                    

                    date_str = None
                    if "오늘" in t or "지금" in t:
                        date_str = datetime.now().strftime("%Y-%m-%d")
                    elif "내일" in t:
                        date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                    elif "모레" in t:
                        date_str = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
                    else:

                        date_str = datetime.now().strftime("%Y-%m-%d")
                    
                    if medicines:


                        복용방법_값 = ""
                        for pattern in METHOD_PATTERNS:
                            method_match = re.search(pattern, t)
                            if method_match:
                                if "식후" in pattern and method_match.group(1):
                                    복용방법_값 = f"식후 {method_match.group(1)}분"
                                elif "식후" in pattern:
                                    복용방법_값 = "식후"
                                elif "식전" in pattern:
                                    복용방법_값 = "식전"
                                elif "공복" in pattern:
                                    복용방법_값 = "공복"
                                else:
                                    복용방법_값 = method_match.group(0)
                                break
                        

                        복용기간_값 = ""



                        period_with_keyword_patterns = [
                            r"복용\s*기간\s*(?:은|는|이|가)?\s*일주일(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                            r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*일(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                            r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*주(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                            r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*개월(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                            r"복용\s*기간.*?일주일",
                            r"복용\s*기간.*?(\d+)\s*일",
                            r"복용\s*기간.*?(\d+)\s*주",
                            r"복용\s*기간.*?(\d+)\s*개월",
                        ]
                        for pattern in period_with_keyword_patterns:
                            period_match = re.search(pattern, t)
                            if period_match:
                                if "일주일" in pattern:
                                    복용기간_값 = "7일"
                                    break
                                elif period_match.lastindex and period_match.group(1):
                                    unit = "일"
                                    if "주" in pattern:
                                        unit = "주"
                                    elif "개월" in pattern:
                                        unit = "개월"
                                    복용기간_값 = f"{period_match.group(1)}{unit}"
                                    break
                        

                        if not 복용기간_값:
                            period_patterns = [
                                r"일주일\s*동안",
                                r"(\d+)\s*일\s*동안",
                                r"(\d+)\s*주일\s*동안",
                                r"(\d+)\s*주\s*동안",
                                r"(\d+)\s*달\s*동안",
                                r"(\d+)\s*개월\s*동안",
                                r"한\s*달\s*동안",
                                r"일주일치",
                                r"(\d+)\s*주일치",
                                r"(\d+)\s*주\s*치",
                                r"(\d+)\s*일치",
                                r"(\d+)\s*개월치",
                            ]
                            for pattern in period_patterns:
                                period_match = re.search(pattern, t)
                                if period_match:
                                    if "일주일" in pattern:
                                        복용기간_값 = "7일"
                                        break
                                    elif "한 달" in pattern or "한달" in pattern:
                                        복용기간_값 = "30일"
                                        break
                                    elif period_match.lastindex and period_match.group(1):
                                        unit = "일"
                                        if "주" in pattern or "주일" in pattern:
                                            unit = "주"
                                        elif "달" in pattern or "개월" in pattern:
                                            unit = "개월"
                                        복용기간_값 = f"{period_match.group(1)}{unit}"
                                        break
                        
                        for medicine in medicines:


                            actual_time = time_of_day or ""

                            medication_entity = {
                                "약명": medicine.get("이름", ""),
                                "용량": medicine.get("용량", ""),
                                "단위": medicine.get("단위", ""),
                                "시간대": actual_time,
                                "복용": "예정" if "먹을" in t else "완료",
                                "날짜": date_str,
                                "복용방법": 복용방법_값,
                                "복용기간": 복용기간_값
                            }
                            
                            out.setdefault("user.약", []).append(medication_entity)
                            print(f"[DEBUG] 약 복용 엔티티 추출: {medication_entity}")
                        return out
                    else:


                        if '복용방법_값' not in locals():
                            복용방법_값 = ""
                            for pattern in METHOD_PATTERNS:
                                method_match = re.search(pattern, t)
                                if method_match:
                                    if "식후" in pattern and method_match.group(1):
                                        복용방법_값 = f"식후 {method_match.group(1)}분"
                                    elif "식후" in pattern:
                                        복용방법_값 = "식후"
                                    elif "식전" in pattern:
                                        복용방법_값 = "식전"
                                    elif "공복" in pattern:
                                        복용방법_값 = "공복"
                                    else:
                                        복용방법_값 = method_match.group(0)
                                    break
                        
                        if '복용기간_값' not in locals() or not 복용기간_값:
                            복용기간_값 = ""

                            period_with_keyword_patterns = [
                                r"복용\s*기간\s*(?:은|는|이|가)?\s*일주일(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                                r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*일(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                                r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*주(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                                r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*개월(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                                r"복용\s*기간.*?일주일",
                                r"복용\s*기간.*?(\d+)\s*일",
                                r"복용\s*기간.*?(\d+)\s*주",
                                r"복용\s*기간.*?(\d+)\s*개월",
                            ]
                            for pattern in period_with_keyword_patterns:
                                period_match = re.search(pattern, t)
                                if period_match:
                                    if "일주일" in pattern:
                                        복용기간_값 = "7일"
                                        break
                                    elif period_match.lastindex and period_match.group(1):
                                        unit = "일"
                                        if "주" in pattern:
                                            unit = "주"
                                        elif "개월" in pattern:
                                            unit = "개월"
                                        복용기간_값 = f"{period_match.group(1)}{unit}"
                                        break
                        
                        actual_time = time_of_day or ""
                        medication_entity = {
                            "시간대": actual_time,
                            "복용": "예정" if "먹을" in t else "완료",
                            "날짜": date_str,
                            "복용방법": 복용방법_값,
                            "복용기간": 복용기간_값
                        }
                        
                        out.setdefault("user.약", []).append(medication_entity)
                        print(f"[DEBUG] 약 복용 엔티티 추출 (fallback, 약 이름 없음): {medication_entity}")
                        return out
                except Exception as e:
                    print(f"[ERROR] 약 복용 엔티티 추출 중 오류: {e}")
                    import traceback
                    traceback.print_exc()



            schedule_keywords = ["약속", "모임", "회식", "미팅", "미팅이", "일정", "예약", "만남", "데이트", "모임이", "약속이"]
            has_schedule_keyword = any(k in t for k in schedule_keywords)
            



            if has_schedule_keyword:
                print(f"[DEBUG] [RULE] 일정 키워드 감지 → 물건 추출 완전 제외: {t}")
            else:


                item_location_keywords = ["있어", "위치", "서랍", "찬장", "방", "주방", "책상", "물건", "보관", "놓았", "두었", "안에", "위에"]
                has_item_location_keywords = any(k in t for k in item_location_keywords)
                

                if has_item_location_keywords and "user.물건" not in out:
                    try:
                        item_extracted = self._extract_item_location_rule(t)
                        if item_extracted and item_extracted.get("user.물건"):
                            out.setdefault("user.물건", []).extend(item_extracted["user.물건"])
                            print(f"[DEBUG] _rule_based_extract에서 물건 위치 추출: {item_extracted['user.물건']}")
                    except Exception as e:
                        print(f"[DEBUG] 물건 위치 추출 중 오류: {e}")
                        pass


            try:
                print(f"[DEBUG] 사용자 개인정보 추출 시작: '{t}'")
            except Exception:
                pass
            

            age_patterns = [
                r"내\s*나이(?:는|가)?\s*(\d+)(?:살|세)",
                r"나는\s*(\d+)(?:살|세)",
                r"저는\s*(\d+)(?:살|세)",
                r"(\d+)(?:살|세)(?:야|이야|입니다|이에요)"
            ]
            
            age = None
            for pattern in age_patterns:
                m = re.search(pattern, t)
                if m:
                    age = f"{m.group(1)}살"
                    break
            

            school_patterns = [
                r"나는\s*([가-힣\s]+(?:중학교|고등학교|대학교|초등학교|학교))에?\s*다녀",
                r"저는\s*([가-힣\s]+(?:중학교|고등학교|대학교|초등학교|학교))에?\s*다녀",
                r"([가-힣\s]+(?:중학교|고등학교|대학교|초등학교|학교))에?\s*다녀",
                r"([가-힣\s]+(?:중학교|고등학교|대학교|초등학교|학교))에?\s*다니고"
            ]
            
            school = None
            for pattern in school_patterns:
                m = re.search(pattern, t)
                if m:
                    school = m.group(1).strip()
                    break
        

            job_patterns = [
                r"나는\s*([가-힣\s]+)(?:이야|이에요|입니다)",
                r"저는\s*([가-힣\s]+)(?:이에요|입니다)",
                r"직업(?:은|이)?\s*([가-힣\s]+)(?:야|이야|이에요|입니다)"
            ]
            
            job = None
            for pattern in job_patterns:
                m = re.search(pattern, t)
                if m:
                    job_candidate = m.group(1).strip()

                    if any(keyword in job_candidate for keyword in ["학생", "회사원", "선생님", "의사", "간호사", "엔지니어", "개발자", "디자이너"]):
                        job = job_candidate
                        break
            

            if age or school or job:
                user_entity = {}
                if age:
                    user_entity["나이"] = age
                if school:
                    user_entity["학교"] = school
                if job:
                    user_entity["직업"] = job
                
                out.setdefault("user.사용자", []).append(user_entity)
                print(f"[DEBUG] 사용자 개인정보 추출: {user_entity}")
        

            if self._is_name_question(t):
                print(f"[DEBUG] 이름 질문 패턴으로 인해 스킵")
                return out
            

            llm_result = self._extract_name_llm(t)
            if llm_result and llm_result.get("name"):
                user_entity = {"이름": llm_result["name"], "확인됨": True}
                if llm_result.get("alias"):
                    user_entity["별칭"] = llm_result["alias"]
                out.setdefault("user.사용자", []).append(user_entity)
            else:

                clean_text = re.sub(r'[\.,!?]+$', '', t.strip())
                

                family_patterns = [
                    r"우리\s*(동생|엄마|아빠|형|누나|언니|오빠|할머니|할아버지)",
                    r"(동생|엄마|아빠|형|누나|언니|오빠|할머니|할아버지)\s*이름",
                    r"가족\s*이름"
                ]
                
                is_family_context = any(re.search(pattern, clean_text) for pattern in family_patterns)
                

                name_patterns = [
                    r"내\s*이름(?:은|이)?\s*([가-힣A-Za-z\s]{2,10})(?:이야|이에요|입니다|예요|야|다|어|아)?",
                    r"나는\s*([가-힣A-Za-z\s]{2,10})(?:야|이다|입니다|이에요|예요)",
                    r"저는\s*([가-힣A-Za-z\s]{2,10})(?:입니다|이에요|예요|야)",
                    r"난\s*([가-힣A-Za-z\s]{2,10})(?:야|이다|이야)"
                ]
                

                if not is_family_context:

                    for pattern in name_patterns:
                        m = re.search(pattern, clean_text)
                        if m:
                            name = self._normalize_name(m.group(1))
                            if self._is_valid_name(name):
                                out.setdefault("user.사용자", []).append({"이름": name, "확인됨": True})
                                break
                

                if is_family_context:
                    for pattern in name_patterns:
                        m = re.search(pattern, clean_text)
                        if m:
                            name = self._normalize_name(m.group(1))
                            if (name 
                                and name not in {"누구게", "몰라"} 
                                and name not in NAME_BLACKLIST 
                                and len(name) >= 2 
                                and not re.search(r"[0-9]", name) 
                                and self._is_valid_name(name)):
                                out.setdefault("user.사용자", []).append({"이름": name, "확인됨": True})
                            break



            family_patterns = [
                r"(남편|아내|엄마|어머니|아빠|아버지|아들|딸|형|누나|동생|언니|오빠|할머니|할아버지|손자|손녀|손주|며느리|사위|부모)"
            ]
            for pattern in family_patterns:
                m = re.search(pattern, t)
                if m:
                    rel = NORMALIZE_KEYS.get(m.group(1), m.group(1))

                    if m.group(1) in t:
                        family_info = {"관계": rel}
                    


                    name_patterns = [
                        f"{m.group(1)}\\s*이름(?:은|이)?\\s*([가-힣A-Za-z]{{2,}})(?:이야|이에요|입니다|야|다|어|아|이고|이고요)?",
                        f"{m.group(1)}\\s*([가-힣A-Za-z]{{2,}})(?:이야|이에요|입니다|야|다|어|아|이고|이고요)(?!\\s*(?:이름|은|이|이다|입니다))"
                    ]
                    
                    name_found = False
                    for name_pattern in name_patterns:
                        name_match = re.search(name_pattern, t)
                        if name_match:
                            name = self._normalize_name(name_match.group(1))

                            if (name 
                                and name not in NAME_BLACKLIST 
                                and name != rel 
                                and name not in ["이름은", "이름이", "이름", "은", "이", "이다", "입니다"]):
                                family_info["이름"] = name
                                name_found = True
                            break
                    

                    if not name_found:
                        direct_name_pattern = f"{m.group(1)}\\s*([가-힣A-Za-z]{{2,}})(?:이라고|라고|라)$"
                        direct_match = re.search(direct_name_pattern, t)
                        if direct_match:
                            name = self._normalize_name(direct_match.group(1))
                            if (name 
                                and name not in NAME_BLACKLIST 
                                and name != rel 
                                and name not in ["이름은", "이름이", "이름", "은", "이", "이다", "입니다"]):
                                family_info["이름"] = name
                                name_found = True
                            break
                    

                    if name_found:

                        existing_family = out.get("user.가족", [])
                        is_duplicate = any(
                            f.get("관계") == family_info.get("관계") and 
                            f.get("이름") == family_info.get("이름")
                            for f in existing_family
                        )
                        
                        if not is_duplicate:
                            out.setdefault("user.가족", []).append(family_info)
                            logger.debug(f"가족 정보 추가: {family_info}")
                        else:
                            logger.debug(f"가족 정보 중복 방지: {family_info}")
                    break


            if (re.search(r"\b약\b", t) or 
                re.search(r"[가-힣A-Za-z]+약", t) or 
                any(drug in t for drug in ["아스피린", "타이레놀", "이부프로펜", "아세트아미노펜"])):
                drugs = self._extract_drugs_with_info(t)
                if drugs:
                    out.setdefault("user.약", []).extend(drugs)


            print(f"[DEBUG] 일정 추출 시작: '{t}'")
            schedule_patterns = [
                r"(내일|오늘|어제)\s*(\d{1,2}시)\s*(?:에|에는)?\s*([가-힣A-Za-z\s]+?)\s*(?:있어|있어요|있습니다)",
                r"(내일|오늘|어제)\s*(오후|오전)?\s*(\d{1,2}시)\s*(?:에|에는)?\s*([가-힣A-Za-z\s]+?)\s*(?:있어|있어요|있습니다)",
                r"(내일|오늘|어제)\s*(오후|오전)?\s*(\d{1,2}시)\s*(?:에|에는)?\s*([가-힣A-Za-z\s]+?)\s*(?:가야|해야|해야해|가야해|해야해요|가야해요)",
                r"([가-힣A-Za-z\s]+?)\s*(?:가야|해야|해야해|가야해|해야해요|가야해요)\s*(?:내일|오늘|어제)?\s*(?:오후|오전)?\s*(\d{1,2}시)?",
                r"(?:병원|회의|약속|미팅|데이트|일정|스케줄|예약)\s*(?:가야|해야|해야해|가야해|해야해요|가야해요)",
                r"(다음\s*주|이번\s*주|다음주|이번주)\s*([가-힣]+요일)\s*(?:에|에는)?\s*([가-힣A-Za-z\s]+?)\s*(?:가야|해야|있어|있어요)",
                r"(이번\s*주|다음\s*주|이번주|다음주)\s*([가-힣]+요일)\s*(저녁|오후|오전|아침)?\s*(\d{1,2}시)?\s*(?:에|에는)?\s*([가-힣A-Za-z\s]+?)\s*(약속|모임|만남)\s*(?:있어|있어요)",
                r"([가-힣A-Za-z\s]+?)\s*(저녁|오후|오전|아침)?\s*(\d{1,2}시)?\s*(?:에|에는)?\s*([가-힣A-Za-z\s]+?)\s*(약속|모임|만남)\s*(?:있어|있어요)",
                r"(\d{1,2}\s*월\s*\d{1,2}\s*일)\s*(?:에|에는)?\s*([가-힣A-Za-z\s]+?)\s*(?:가야|해야|있어|있어요)",
                r"(오늘|내일|어제)\s*(저녁|오후|오전)?\s*(\d{1,2}시)\s*(?:에|에는)?\s*([가-힣A-Za-z\s]+?)\s*(?:로\s*했어|로\s*했어요|만나기로\s*했어|만나기로\s*했어요)",
                r"(내일|오늘|어제)\s*([가-힣A-Za-z\s]+?),\s*(아침|오후|오전|저녁)?\s*(\d{1,2}시)\s*([가-힣A-Za-z\s]+?)(?:야|이야|예요|에요)",
                r"(내일|오늘|어제)\s*([가-힣A-Za-z\s]+?)\s*(?:있어|있어요|있습니다)",
                r"([가-힣A-Za-z\s]+?),\s*(내일|오늘|어제)\s*(아침|오후|오전|저녁)?\s*(\d{1,2}시)"
            ]
            
            cancel_patterns = [
                r"([가-힣A-Za-z\s]+?)\s*(?:취소|취소했어|취소했어요|취소했습니다|취소함)",
                r"(?:취소|취소했어|취소했어요|취소했습니다|취소함)\s*([가-힣A-Za-z\s]+?)",
                r"([가-힣A-Za-z\s]+?)\s*(?:안\s*해|안\s*해요|안\s*합니다|안\s*함)"
            ]
            

            for pattern in cancel_patterns:
                m = re.search(pattern, t)
                if m:
                    title = m.group(1).strip()
                    print(f"[DEBUG] 일정 취소 감지: '{title}'")

                    self._cancel_schedule(session_id, title)
                    return {"user.일정취소": [{"제목": title, "상태": "취소됨"}]}
            
            for i, pattern in enumerate(schedule_patterns):
                m = re.search(pattern, t)
                if m:
                    print(f"[DEBUG] 패턴 {i+1} 매치됨: {pattern}")
                    print(f"[DEBUG] 매치 그룹: {m.groups()}")
                    groups = m.groups()
                    title_part = ""
                    date_part = "오늘"
                    time_part = ""
                    ampm_part = ""
                    
                    
                    is_appointment_pattern = (
                        (i == 4) or
                        (("이번\\s*주" in pattern or "다음\\s*주" in pattern or "이번주" in pattern or "다음주" in pattern) and
                         ("약속|모임|만남" in pattern or "약속" in pattern or "모임" in pattern or "만남" in pattern))
                    )
                    
                    print(f"[DEBUG] 패턴 {i+1} 검사: is_appointment_pattern={is_appointment_pattern}, len(groups)={len(groups)}")
                    
                    if is_appointment_pattern and len(groups) >= 6:

                        print(f"[DEBUG] 약속 패턴 감지됨 (인덱스 {i}) - 일정으로 처리")
                        week_part = groups[0] if len(groups) > 0 and groups[0] else ""
                        day_part = groups[1] if len(groups) > 1 and groups[1] else ""
                        ampm_part = groups[2] if len(groups) > 2 and groups[2] else ""
                        time_hour_part = groups[3] if len(groups) > 3 and groups[3] else ""
                        person_part = groups[4] if len(groups) > 4 and groups[4] else ""
                        keyword_part = groups[5] if len(groups) > 5 and groups[5] else ""
                        

                        if week_part and day_part:
                            date_part = f"{week_part.strip()} {day_part.strip()}".strip()
                        elif week_part:
                            date_part = week_part.strip()
                        elif day_part:
                            date_part = day_part.strip()
                        else:
                            date_part = ""
                        

                        if ampm_part and time_hour_part:
                            time_part = f"{ampm_part.strip()} {time_hour_part.strip()}".strip()
                        elif ampm_part:
                            time_part = ampm_part.strip()
                        elif time_hour_part:
                            time_part = time_hour_part.strip()
                        else:
                            time_part = ""
                        

                        if person_part and keyword_part:
                            title_part = f"{person_part.strip()} {keyword_part.strip()}".strip()
                        elif person_part:
                            title_part = person_part.strip()
                        elif keyword_part:
                            title_part = keyword_part.strip()
                        else:
                            title_part = "약속"
                        

                        if title_part or date_part:
                            schedule_info = {
                                "제목": title_part,
                                "날짜": date_part,
                                "시간": time_part
                            }
                            out.setdefault("user.일정", []).append(schedule_info)
                            print(f"[DEBUG] 일정 엔티티 추출 (약속 패턴): {schedule_info}")

                            if "user.물건" in out:
                                print(f"[DEBUG] 일정 엔티티 저장됨 → 물건 엔티티 제거: {out.get('user.물건')}")
                                out.pop("user.물건", None)

                            return out
                    


                    if i == 3 and len(groups) >= 3 and ("이번" in pattern or "다음" in pattern) and "주" in pattern and "요일" in pattern:

                        week_part = groups[0] if len(groups) > 0 and groups[0] else ""
                        day_part = groups[1] if len(groups) > 1 and groups[1] else ""
                        title_part = groups[2] if len(groups) > 2 and groups[2] else ""
                        

                        if week_part and day_part:
                            date_part = f"{week_part.strip()} {day_part.strip()}".strip()
                        elif week_part:
                            date_part = week_part.strip()
                        elif day_part:
                            date_part = day_part.strip()
                        else:
                            date_part = ""
                        

                        if title_part:
                            title_clean = re.sub(r"^(는|은|이|가|을|를)\s*", "", title_part.strip())
                            title_clean = re.sub(r"(가|이|을|를)\s*$", "", title_clean.strip())
                        else:
                            title_clean = ""
                        

                        if title_clean or date_part:
                            schedule_info = {
                                "제목": title_clean,
                                "날짜": date_part,
                                "시간": ""
                            }
                            out.setdefault("user.일정", []).append(schedule_info)
                            print(f"[DEBUG] 일정 엔티티 추출 (패턴 4): {schedule_info}")

                            if "user.물건" in out:
                                print(f"[DEBUG] 일정 엔티티 저장됨 → 물건 엔티티 제거: {out.get('user.물건')}")
                                out.pop("user.물건", None)

                            return out
                    

                    if i == 0 and len(groups) >= 3 and "있어" in pattern:
                        date_part = groups[0] if groups[0] else "오늘"
                        time_part = groups[1] if groups[1] else ""
                        title_part = groups[2] if groups[2] else ""
                        ampm_part = ""

                        if title_part:
                            title_part = re.sub(r"(가|이|을|를|은|는)\s*$", "", title_part.strip())

                        if title_part or date_part:
                            schedule_info = {
                                "제목": title_part,
                                "날짜": date_part,
                                "시간": time_part
                            }
                            out.setdefault("user.일정", []).append(schedule_info)
                            print(f"[DEBUG] 일정 엔티티 추출 (패턴 1): {schedule_info}")
                            if "user.물건" in out:
                                print(f"[DEBUG] 일정 엔티티 저장됨 → 물건 엔티티 제거: {out.get('user.물건')}")
                                out.pop("user.물건", None)
                            return out

                    elif i == 1 and len(groups) >= 4 and "있어" in pattern:
                        date_part = groups[0] if groups[0] else "오늘"
                        ampm_part = groups[1] if groups[1] else ""
                        time_part = groups[2] if groups[2] else ""
                        title_part = groups[3] if groups[3] else ""

                        if ampm_part:
                            time_part = f"{ampm_part} {time_part}"

                        if title_part:
                            title_part = re.sub(r"(가|이|을|를|은|는)\s*$", "", title_part.strip())

                        if title_part or date_part:
                            schedule_info = {
                                "제목": title_part,
                                "날짜": date_part,
                                "시간": time_part
                            }
                            out.setdefault("user.일정", []).append(schedule_info)
                            print(f"[DEBUG] 일정 엔티티 추출 (패턴 2): {schedule_info}")
                            if "user.물건" in out:
                                print(f"[DEBUG] 일정 엔티티 저장됨 → 물건 엔티티 제거: {out.get('user.물건')}")
                                out.pop("user.물건", None)
                            return out

                    elif pattern.startswith(r"(내일|오늘|어제)") and len(groups) >= 4:
                        date_part = groups[0] if groups[0] else "오늘"
                        ampm_part = groups[1] if groups[1] else ""
                        time_part = groups[2] if groups[2] else ""
                        title_part = groups[3] if groups[3] else ""
                        

                        if title_part and "," in title_part:
                            parts = title_part.split(",")
                            if len(parts) >= 2:
                                title_part = parts[0].strip()
                                time_info = parts[1].strip()

                                if time_info:
                                    time_part = f"{time_info} {time_part}" if time_part else time_info
                        

                        if ampm_part:
                            time_part = f"{ampm_part} {time_part}"
                    

                    elif "약속|모임|만남" in pattern and "있어|있어요" in pattern and "이번.*주|다음.*주" not in pattern:

                        if len(groups) >= 5:

                            date_match = re.search(r"(이번\s*주|다음\s*주|이번주|다음주|오늘|내일|어제)", t)
                            if date_match:
                                date_part = date_match.group(1)
                            time_part = f"{groups[1]} {groups[2]}" if groups[1] and groups[2] else (groups[1] if groups[1] else (groups[2] if groups[2] else ""))
                            title_part = f"{groups[3]} {groups[4]}" if groups[3] and groups[4] else (groups[3] if groups[3] else (groups[4] if groups[4] else ""))
                        elif len(groups) >= 3:
                            date_match = re.search(r"(이번\s*주|다음\s*주|이번주|다음주|오늘|내일|어제)", t)
                            if date_match:
                                date_part = date_match.group(1)
                            time_part = groups[1] if len(groups) > 1 and groups[1] else ""
                            title_part = groups[2] if len(groups) > 2 and groups[2] else ""
                        else:

                            date_match = re.search(r"(이번\s*주|다음\s*주|이번주|다음주|오늘|내일|어제)", t)
                            if date_match:
                                date_part = date_match.group(1)
                            else:
                                date_part = "오늘"
                            time_part = ""
                            title_part = groups[0] if len(groups) > 0 else ""
                    

                    elif "다음" in pattern and "주" in pattern and "약속|모임|만남" not in pattern:
                        date_part = f"{groups[0]} {groups[1]}" if len(groups) > 1 else groups[0] if len(groups) > 0 else ""
                        title_part = groups[2] if len(groups) > 2 else ""
                        time_part = ""

                elif "가야" in pattern and "해야" in pattern:
                    title_part = groups[0] if len(groups) > 0 else ""
                    date_part = groups[1] if len(groups) > 1 and groups[1] else "오늘"
                    time_part = groups[2] if len(groups) > 2 and groups[2] else ""

                elif "병원|회의|약속|미팅|데이트|일정|스케줄|예약" in pattern:
                    title_part = groups[0] if len(groups) > 0 else ""
                    time_part = ""

                    date_match = re.search(r"(오늘|내일|어제|다음주|이번주)", t)
                    if date_match:
                        date_part = date_match.group(1)

                elif "다음.*주|이번.*주" in pattern:
                    date_part = f"{groups[0]} {groups[1]}" if len(groups) > 1 else groups[0] if len(groups) > 0 else ""
                    title_part = groups[2] if len(groups) > 2 else ""
                    time_part = ""

                elif r"\d{1,2}\s*월\s*\d{1,2}\s*일" in pattern:
                    date_part = groups[0] if len(groups) > 0 else ""
                    title_part = groups[1] if len(groups) > 1 else ""
                    time_part = ""

                elif "로\s*했어|로\s*했어요|만나기로\s*했어|만나기로\s*했어요" in pattern:
                    date_part = groups[0] if len(groups) > 0 else ""
                    time_part = f"{groups[1]} {groups[2]}" if len(groups) > 2 and groups[1] and groups[2] else ""
                    title_part = groups[3] if len(groups) > 3 else ""

                elif r"내일|오늘|어제.*,\s*아침|오후|오전|저녁" in pattern:
                    date_part = groups[0] if len(groups) > 0 else ""
                    title_part = groups[1] if len(groups) > 1 else ""
                    time_part = f"{groups[2]} {groups[3]}" if len(groups) > 3 and groups[2] and groups[3] else ""

                    if title_part and time_part and title_part.isdigit() and not time_part.isdigit():

                        temp = title_part
                        title_part = time_part
                        time_part = temp

                    if title_part and "여행" in title_part and "아침" in title_part:

                        if "아침" in title_part:
                            title_part = title_part.replace(" 아침", "").replace("아침", "")
                            time_part = f"아침 {time_part}" if time_part else "아침"

                elif r"내일|오늘|어제.*있어|있어요|있습니다" in pattern and len(groups) == 2:
                    date_part = groups[0] if len(groups) > 0 else ""
                    title_part = groups[1] if len(groups) > 1 else ""
                    time_part = ""

                elif r",\s*내일|오늘|어제.*아침|오후|오전|저녁" in pattern:
                    title_part = groups[0] if len(groups) > 0 else ""
                    date_part = groups[1] if len(groups) > 1 else ""
                    time_part = f"{groups[2]} {groups[3]}" if len(groups) > 3 and groups[2] and groups[3] else ""

                elif "야|이야|예요|에요" in pattern:
                    date_part = groups[0] if len(groups) > 0 else ""
                    title_part = groups[1] if len(groups) > 1 else ""
                    time_part = f"{groups[2]} {groups[3]}" if len(groups) > 3 and groups[2] and groups[3] else ""

                elif len(groups) == 2 and "있어" in t:
                    date_part = groups[0] if len(groups) > 0 else ""
                    title_part = groups[1] if len(groups) > 1 else ""
                    time_part = ""
                

                    if title_part or date_part:

                        if title_part:
                            title_clean = re.sub(r"^(는|은|이|가|을|를)\s*", "", title_part.strip())
                            title_clean = re.sub(r"(가|이|을|를)\s*$", "", title_clean.strip())
                        else:

                            title_clean = date_part if date_part else "일정"
                        

                        if not date_part:
                            date_match = re.search(r"(이번\s*주|다음\s*주|이번주|다음주|오늘|내일|어제)", t)
                            if date_match:
                                date_part = date_match.group(1)
                            else:
                                date_part = "오늘"
                        
                        schedule_info = {
                            "제목": title_clean,
                            "날짜": date_part,
                            "시간": time_part if time_part else ""
                        }
                        out.setdefault("user.일정", []).append(schedule_info)
                        print(f"[DEBUG] 일정 엔티티 추출: {schedule_info}")

                        if "user.물건" in out:
                            print(f"[DEBUG] 일정 엔티티 저장됨 → 물건 엔티티 제거: {out.get('user.물건')}")
                            out.pop("user.물건", None)

                        return out


            birthday_patterns = [
                r"생일.*?(\d{1,2}\s*월\s*\d{1,2}\s*일)",
                r"(\d{1,2}\s*월\s*\d{1,2}\s*일).*?생일",
                r"내\s*생일.*?(\d{1,2}\s*월\s*\d{1,2}\s*일)",
                r"(\d{1,2}\s*월\s*\d{1,2}\s*일).*?기억",
            ]
            for p in birthday_patterns:
                m = re.search(p, t)
                if m:
                    out.setdefault("user.기념일", []).append({
                        "관계": "사용자", "제목": "생일", "날짜": m.group(1)
                    })
                    break
            

            m = re.search(r"(제사|기일|결혼기념일).*?(\d{1,2}\s*월\s*\d{1,2}\s*일)", t)
            if m:
                out.setdefault("user.기념일", []).append({
                    "관계": "",
                    "제목": m.group(1),
                    "날짜": m.group(2)
                })


            m = re.search(r"취미(?:는|가)?\s*([가-힣A-Za-z0-9 ]+)", t)
            if m:
                hobby = re.sub(r"(이야|야|입니다|예요|에요)$", "", m.group(1)).strip()
                if hobby:
                    out.setdefault("user.취미", []).append({"이름": hobby})


            preference_patterns = [
                r"(?:나는|난|전|저는)?\s*([가-힣A-Za-z0-9 ]+?)\s*(?:좋아해|좋아합니다|좋아함|좋아|선호해|선호합니다)",
                r"([가-힣A-Za-z0-9 ]+?)\s*(?:가|이)\s*(?:취향이야|취향이에요|취향입니다|취향이에요)",
                r"([가-힣A-Za-z0-9 ]+?)\s*(?:를|을)\s*(?:좋아해|좋아합니다|좋아함|좋아|선호해|선호합니다)",
                r"([가-힣A-Za-z0-9 ]+?)\s*(?:를|을)\s*(?:선호해|선호합니다|선호함|선호)"
            ]
            for pattern in preference_patterns:
                m = re.search(pattern, t)
                if m:
                    val = m.group(1).strip()

                    val = re.sub(r'(를|을|가|이)$', '', val).strip()

                    val = re.sub(r'^(나는|난|전|저는)\s*', '', val).strip()
                    if val and val not in STOPWORDS:
                        out.setdefault("user.취향", []).append({"종류": "", "값": val})
                    break


            medicine_patterns = [
                r"([가-힣]+약)\s*(?:먹|복용|드셨|드셨어|드셨어요|드셨습니다)",
                r"(?:먹|복용|드셨|드셨어|드셨어요|드셨습니다)\s*([가-힣]+약)",
                r"([가-힣]+약)\s*(?:과|와)\s*([가-힣]+약)\s*(?:먹|복용|드셨|드셨어|드셨어요|드셨습니다)",
                r"(?:하루|매일|일일)\s*(\d+)\s*(?:번|회|차)\s*(?:씩|마다)\s*(?:먹|복용|드셨|드셨어|드셨어요|드셨습니다)",
                r"(\d+)\s*(?:일|주|개월|년)\s*(?:동안|간)\s*(?:먹|복용|드셨|드셨어|드셨어요|드셨습니다)"
            ]
            
            for pattern in medicine_patterns:
                m = re.search(pattern, t)
                if m:
                    if "과" in pattern or "와" in pattern:

                        med1 = m.group(1)
                        med2 = m.group(2)
                        out.setdefault("user.약", []).append({
                            "약명": med1,
                            "복용량": None,
                            "복용주기": None,
                            "복용기간": None
                        })
                        out.setdefault("user.약", []).append({
                            "약명": med2,
                            "복용량": None,
                            "복용주기": None,
                            "복용기간": None
                        })
                    elif "번" in pattern or "회" in pattern or "차" in pattern:

                        frequency = m.group(1)

                        if "user.약" in out and out["user.약"]:
                            for med in out["user.약"]:
                                med["복용주기"] = f"하루 {frequency}번"
                    elif "일" in pattern or "주" in pattern or "개월" in pattern or "년" in pattern:

                        period = m.group(1) + ("일" if "일" in pattern else "주" if "주" in pattern else "개월" if "개월" in pattern else "년")

                        if "user.약" in out and out["user.약"]:
                            for med in out["user.약"]:
                                med["복용기간"] = period
                    else:

                        med_name = m.group(1)
                        out.setdefault("user.약", []).append({
                            "약명": med_name,
                            "복용량": None,
                            "복용주기": None,
                            "복용기간": None
                        })
                    break


            m = re.search(r"(두통|머리\s*아픔|기침|재채기|콧물|피곤|어지럼|열|발열|복통|몸살)", t)
            if m:
                out.setdefault("user.건강상태", []).append({
                    "증상": m.group(1),
                    "정도": None,
                    "기간": None,
                    "기타": None
                })


            drug_patterns = [
                r"([가-힣A-Za-z]+약)(?:을|를)?\s*(?:먹었어|먹었어요|먹었습니다|먹음|드셨어|드셨어요|드셨습니다|드심|복용했어|복용했어요|복용했습니다|복용함)",
                r"(?:먹었어|먹었어요|먹었습니다|먹음|드셨어|드셨어요|드셨습니다|드심|복용했어|복용했어요|복용했습니다|복용함)\s*([가-힣A-Za-z]+약)",
                r"([가-힣A-Za-z]+약)(?:을|를)?\s*(?:먹어|먹어요|먹습니다|먹어야|먹어야해|먹어야해요|먹어야합니다|먹어야함|복용해|복용해요|복용합니다|복용해야|복용해야해|복용해야해요|복용해야합니다|복용해야함)",
                r"(?:먹어|먹어요|먹습니다|먹어야|먹어야해|먹어야해요|먹어야합니다|먹어야함|복용해|복용해요|복용합니다|복용해야|복용해야해|복용해야해요|복용해야합니다|복용해야함)\s*([가-힣A-Za-z]+약)"
            ]
            

            for pattern in drug_patterns:
                m = re.search(pattern, t)
                if m:
                    drug_name = m.group(1).strip()
                    print(f"[DEBUG] 약물 패턴 감지: '{drug_name}'")
                    return {"user.약물": [{"약명": drug_name, "복용일": "오늘"}]}
        



            has_medicine_keyword = any(keyword in t for keyword in MEDICINE_KEYWORDS)

            if not has_medicine_keyword and "약" in t and "약속" not in t:

                if re.search(r"[가-힣A-Za-z]+약|약\s*[먹드]", t):
                    has_medicine_keyword = True

            if not has_medicine_keyword:
                if re.search(r"[가-힣A-Za-z]+약(?!속)", t):
                    has_medicine_keyword = True
        
            if "먹" in t and not has_medicine_keyword:

                skip_patterns = [
                    r"(아무것도|아무것)\s*안\s*먹",
                    r"안\s*먹었어",
                    r"굶었어",
                    r"먹지\s*않았어",
                    r"식사\s*안\s*했어"
                ]
                
                for skip_pattern in skip_patterns:
                    if re.search(skip_pattern, t):

                        break
                else:

                    meal_data = self._extract_meal_entity(t)
                    if meal_data.get("끼니") or meal_data.get("메뉴"):


                        meal_date = meal_data.get("날짜") or "오늘"
                        meal_entity = {
                            "끼니": meal_data.get("끼니"),
                            "메뉴": meal_data.get("메뉴", []),
                            "날짜": meal_date,
                            "시간": None
                        }
                        

                        extracted_time = self._extract_time_from_text(t)
                        if extracted_time:
                            meal_entity["시간"] = extracted_time
                        
                        out.setdefault("user.식사", []).append(meal_entity)
                        print(f"[DEBUG] 식사 엔티티 추출 (개선): {meal_entity}")
                        return out
                    else:


                        m1 = re.search(r"(오늘|어제|내일)?\s*(아침|점심|저녁).*?([가-힣]+?)\s*(?:먹|먹어|먹었|먹었어|먹었어요|먹었습니다)", t)

                    m1_1 = re.search(r"(오늘|어제|내일)?\s*(아침|점심|저녁)에\s*([가-힣]+?)\s*(?:먹|먹어|먹었|먹었어|먹었어요|먹었습니다)", t)

                    m1_2 = re.search(r"(오늘|어제|내일)?\s*(아침|점심|저녁)에\s*([가-힣]+?)(?:먹어|먹었|먹었어|먹었어요|먹었습니다|먹어썽)", t)

                    m2 = re.search(r"(오늘|어제|내일)?\s*(아침|점심|저녁)\s*먹", t)

                    m2_1 = re.search(r"(오늘|어제|내일)?\s*(아침|점심|저녁)에\s*먹", t)

                    m3 = re.search(r"([가-힣]+)\s*먹", t)

                    m3_1 = re.search(r"([가-힣]+)\s*먹었고,?\s*(\d{1,2}시\s*반?)\s*에?\s*먹", t)
            
                    if m3_1:


                        menu_item = m3_1.group(1).strip()
                        extracted_time = m3_1.group(2).strip()
                        


                        if menu_item not in {"음식", "뭔가", "뭐", "약", "약물", "약품", "식후에", "식전에", "식후", "식전"}: 

                            inferred_meal = None
                            if extracted_time:
                                hour = self._extract_hour_from_time(extracted_time)
                                if hour and 6 <= hour < 11:
                                    inferred_meal = "아침"
                                elif hour and 11 <= hour < 15:
                                    inferred_meal = "점심"
                                elif hour and 15 <= hour < 22:
                                    inferred_meal = "저녁"
                            

                            if not inferred_meal:
                                inferred_meal = None
                        
                            meal_entity = {
                                "끼니": inferred_meal, "메뉴": [menu_item], "날짜": "오늘", "시간": extracted_time
                            }

                            out.setdefault("user.식사", []).append(meal_entity)
                    elif m2 or m2_1:

                        if m2_1:
                            rel_date, meal = m2_1.group(1), m2_1.group(2)
                        else:
                            rel_date, meal = m2.group(1), m2.group(2)
                        

                        extracted_time = self._extract_time_from_text(t)
                        

                        if not extracted_time and session_id in self.time_context:
                            context = self.time_context[session_id]
                            if context.get("last_meal") == meal and context.get("last_time"):
                                extracted_time = context["last_time"]
                                logger.debug(f"이전 컨텍스트에서 시간 정보 연결: {meal} {extracted_time}")
                        

                        if extracted_time:
                            if session_id not in self.time_context:
                                self.time_context[session_id] = {}
                            self.time_context[session_id]["last_time"] = extracted_time
                            self.time_context[session_id]["last_meal"] = meal
                            self.time_context[session_id]["last_menu"] = None
                        
                        out.setdefault("user.식사", []).append({
                            "끼니": meal, "메뉴": [], "날짜": rel_date, "시간": extracted_time
                        })
                    elif m1 or m1_1 or m1_2:
                        if m1_2:
                            rel_date, meal, menu_raw = m1_2.group(1), m1_2.group(2), m1_2.group(3)
                        elif m1_1:
                            rel_date, meal, menu_raw = m1_1.group(1), m1_1.group(2), m1_1.group(3)
                        else:
                            rel_date, meal, menu_raw = m1.group(1), m1.group(2), m1.group(3)
                        


                        menu_raw_processed = menu_raw
                        

                        complex_menu_patterns = [
                            r"샤인\s*머스켓", r"치킨\s*버거", r"피자\s*슬라이스", r"햄\s*버거", r"치즈\s*버거",
                            r"김치\s*찌개", r"된장\s*찌개", r"미역\s*국", r"계란\s*말이", r"김치\s*전"
                        ]
                        
                        for pattern in complex_menu_patterns:
                            menu_raw_processed = re.sub(pattern, lambda m: m.group(0).replace(" ", "_"), menu_raw_processed)
                        

                        menu_candidates = [x.strip().replace("_", " ") for x in re.split(r"[,와과랑및]", menu_raw_processed) if x.strip()]
                        

                        menu_stopwords = {
                            "그냥", "에는", "에서", "을", "를", "이", "가", "은", "는", "에", "의", "로", "으로",
                        "하고", "하면서", "먹었어", "먹었고", "먹었는데", "먹었어요", "먹었습니다",
                        "드셨어", "드셨고", "드셨는데", "드셨어요", "드셨습니다",
                        "했어", "했고", "했는데", "했어요", "했습니다",
                        "음식", "뭔가", "뭐", "약", "약물", "약품", "간식", "디저트"
                        }
                        
                        menus = [x for x in menu_candidates 
                                if (x not in STOPWORDS and x not in menu_stopwords and 
                                    len(x) > 1 and not re.match(r'^[0-9]+$', x) and
                                    not re.match(r'^[가-힣]{1}$', x))]
                        

                        extracted_time = self._extract_time_from_text(t)
                        

                        if not extracted_time and session_id in self.time_context:
                            context = self.time_context[session_id]
                            if context.get("last_meal") == meal and context.get("last_time"):
                                extracted_time = context["last_time"]
                                logger.debug(f"이전 컨텍스트에서 시간 정보 연결: {meal} {extracted_time}")
                        

                        if extracted_time:
                            if session_id not in self.time_context:
                                self.time_context[session_id] = {}
                            self.time_context[session_id]["last_time"] = extracted_time
                            self.time_context[session_id]["last_meal"] = meal
                        
                        out.setdefault("user.식사", []).append({
                            "끼니": meal, "메뉴": menus, "날짜": rel_date, "시간": extracted_time
                        })
                    elif m3:
                        menu_item = m3.group(1).strip()
                        

                        extracted_time = self._extract_time_from_text(t)
                        if extracted_time and menu_item in {"시에", "시", "분", "오전", "오후", "새벽", "밤"}:


                            if session_id in self.time_context:
                                context = self.time_context[session_id]
                                if context.get("last_meal"):
                                    meal = context["last_meal"]
                                    out.setdefault("user.식사", []).append({
                                        "끼니": meal, "메뉴": [], "날짜": "오늘", "시간": extracted_time
                                    })

                                    self.time_context[session_id]["last_time"] = extracted_time
                                    return out
                        


                        if menu_item not in {"음식", "뭔가", "뭐", "약", "약물", "약품", "시에", "시", "분", "오전", "오후", "새벽", "밤", "식후에", "식전에", "식후", "식전"}: 
                            extracted_time = self._extract_time_from_text(t)
                            

                            inferred_meal = None
                            if extracted_time:
                                hour = self._extract_hour_from_time(extracted_time)
                                if hour and 6 <= hour < 11:
                                    inferred_meal = "아침"
                                elif hour and 11 <= hour < 15:
                                    inferred_meal = "점심"
                                elif hour and 15 <= hour < 22:
                                    inferred_meal = "저녁"
                            

                            if not inferred_meal:
                                inferred_meal = None
                            
                            out.setdefault("user.식사", []).append({
                                "끼니": inferred_meal, "메뉴": [menu_item], "날짜": "오늘", "시간": extracted_time
                            })



            if "user.물건" not in out and "user.일정" not in out:
                try:

                    item_location_result = self._extract_item_location_rule(t)
                    if item_location_result:
                        out.update(item_location_result)
                    else:

                        item_command_result = self._extract_item_command_rule(t)
                        if item_command_result:
                            out.update(item_command_result)
                except Exception as e:
                    print(f"[WARN] 물건 추출 중 오류: {e}")

        except Exception as e:
            print(f"[ERROR] _rule_based_extract 전체 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return out


        if "user.일정" in out and "user.물건" in out:
            print(f"[DEBUG] [FINAL CHECK] 일정 엔티티 존재 → 물건 엔티티 제거: {out.get('user.물건')}")
            out.pop("user.물건", None)
        
        return out

    def _extract_entities_with_llm(self, user_input: str, session_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        try:

            prompt = f"""
다음 사용자 발화에서 개인정보를 추출해주세요.

사용자 발화: "{user_input}"

다음 정보가 있으면 추출하세요:
- 나이: "16살", "20세" 등 (숫자+살/세 형식)
- 학교: "곡반 중학교", "서울대학교", "서울고등학교" 등 (학교명 전체 추출)
  * 다양한 표현을 이해하세요: "다녀", "다니고", "재학중", "졸업했어" 등
  * 예: "나는 서울고등학교에 다녀" → "서울고등학교"
  * 학교명만 추출하고 조사("에", "에서")는 제외하세요
- 직업: "학생", "회사원", "선생님" 등
- 취미: "독서", "영화감상" 등

JSON 형식으로 응답:
{{
    "사용자": [{{"나이": "16살", "학교": "서울고등학교", "직업": "학생", "취미": "독서"}}]
}}

정보가 없으면 빈 배열로 표시:
{{
    "사용자": []
}}
"""
            
            response = self.llm.invoke(prompt)
            result_text = response.content.strip()
            print(f"[DEBUG] LLM 원본 응답: {result_text}")
            

            import json
            import re
            

            json_text = re.sub(r'```json\s*', '', result_text)
            json_text = re.sub(r'```\s*$', '', json_text).strip()
            
            try:
                entities = json.loads(json_text)
                print(f"[DEBUG] LLM 엔티티 추출 결과: {entities}")
                

                formatted_entities = {}
                for entity_type, entity_list in entities.items():
                    if entity_list:
                        formatted_entities[f"user.{entity_type}"] = entity_list
                
                return formatted_entities
                
            except json.JSONDecodeError as e:
                print(f"[DEBUG] LLM JSON 파싱 실패: {e}")
                return {}
                
        except Exception as e:
            print(f"[DEBUG] LLM 엔티티 추출 실패: {e}")
            return {}


    def _pre_extract_entities(self, user_input: str, session_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        merged: Dict[str, List[Dict[str, Any]]] = {}


        print(f"[DEBUG] LLM 엔티티 추출 시작 (entity_chain): '{user_input}'")
        structured_llm_out = {}
        try:
            if hasattr(self, 'entity_chain') and self.entity_chain:
                result = self.entity_chain.invoke({"utterance": user_input})
                print(f"[DEBUG] entity_chain 원본 응답: {result}")
                

                if isinstance(result, dict):
                    for key, value in result.items():
                        if key.startswith("user."):
                            structured_llm_out[key] = value if isinstance(value, list) else [value]
                        else:

                            entity_key = f"user.{key}" if key in ["약", "일정", "식사", "가족", "물건", "기념일", "취미", "취향", "건강상태"] else f"user.{key}"
                            structured_llm_out[entity_key] = value if isinstance(value, list) else [value]
                    
                    print(f"[DEBUG] entity_chain 변환된 결과: {structured_llm_out}")
        except Exception as e:
            print(f"[WARN] entity_chain 추출 실패: {e}")
            import traceback
            traceback.print_exc()
        

        user_info_llm_out = {}
        try:
            user_info_llm_out = self._extract_entities_with_llm(user_input, session_id)
            print(f"[DEBUG] 사용자 정보 LLM 추출 결과: {user_info_llm_out}")
        except Exception as e:
            print(f"[WARN] 사용자 정보 LLM 추출 실패: {e}")
        

        if structured_llm_out:
            merged = structured_llm_out.copy()

            if user_info_llm_out.get("user.사용자"):
                merged["user.사용자"] = user_info_llm_out["user.사용자"]
        elif user_info_llm_out:
            merged = user_info_llm_out.copy()
        

        if not merged or (not structured_llm_out and not user_info_llm_out):
            print(f"[DEBUG] LLM 결과 없음 또는 부족 → rule-based fallback 시도")
            rule_out = self._rule_based_extract(user_input, session_id)
            print(f"[DEBUG] _pre_extract_entities rule_out (fallback): {rule_out}")
            

            if merged:

                for key, value in rule_out.items():
                    if key not in merged:
                        merged[key] = value
                    else:

                        existing = merged[key]
                        new_items = [item for item in value if item not in existing]
                        if new_items:
                            merged[key].extend(new_items)
            else:
                merged = rule_out.copy()
        

        for entity_key, entity_list in merged.items():
            for entity in entity_list:
                if isinstance(entity, dict) and "질문" in entity:
                    print(f"[DEBUG] _pre_extract_entities에서 재질문 발견: {entity['질문']}")
                    return {entity_key: [entity]}
        

        for entity_key, entity_list in merged.items():
            if entity_list:
                for entity in entity_list:
                    if isinstance(entity, dict):

                        entity_type = entity_key.replace("user.", "")
                        

                        missing_info = self._check_missing_fields(entity_type, entity)
                        if missing_info["has_missing"]:
                            print(f"[DEBUG] Slot-filling 필요: {entity_type} - {missing_info['missing_fields']}")
                            return {
                                "success": False,
                                "incomplete": True,
                                "entity_type": entity_type,
                                "message": missing_info["message"],
                                "pending_data": {
                                    "entity_type": entity_type,
                                    "new_data": entity,
                                    "missing_fields": missing_info["missing_fields"],
                                    "session_id": session_id
                                }
                            }
        

        if "user.약물" in merged:
            print(f"[DEBUG] 약물 패턴 우선 처리: {merged['user.약물']}")
            return merged
        

        return merged

    def _dedup_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        unique = []
        for e in entities:

            def make_hashable(obj):
                if isinstance(obj, dict):
                    return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
                elif isinstance(obj, list):
                    return tuple(make_hashable(item) for item in obj)
                else:
                    return obj
            
            key = make_hashable(e)
            if key not in seen:
                seen.add(key)
                unique.append(e)
        return unique

    def _dedup_drug_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        drug_dict = {}
        for e in entities:
            drug_name = e.get("약명", "")
            if not drug_name:
                continue
                
            if drug_name not in drug_dict:
                drug_dict[drug_name] = e
            else:

                existing = drug_dict[drug_name]
                merged = {"약명": drug_name}
                

                if existing.get("복용") or e.get("복용"):
                    merged["복용"] = existing.get("복용", []) + e.get("복용", [])
                

                if existing.get("식사와의 관계") or e.get("식사와의 관계"):
                    merged["식사와의 관계"] = e.get("식사와의 관계") or existing.get("식사와의 관계")
                

                if existing.get("복용 기간") or e.get("복용 기간"):
                    merged["복용 기간"] = e.get("복용 기간") or existing.get("복용 기간")
                

                for key, value in existing.items():
                    if key not in merged and value:
                        merged[key] = value
                for key, value in e.items():
                    if key not in merged and value:
                        merged[key] = value
                
                drug_dict[drug_name] = merged
        
        return list(drug_dict.values())

    def _is_complete_entity(self, entity_key: str, entity: dict) -> bool:
        required_fields = {
            "user.사용자": ["이름"],
            "user.약": ["약명"],
            "user.일정": ["제목", "날짜"],
            "user.기념일": ["제목", "날짜"],
            "user.가족": ["관계"],
            "user.건강상태": ["증상"],
            "user.물건": ["이름"],
            "user.식사": ["끼니"],
            "user.취미": ["이름"],
            "user.취향": ["값"]
        }
        
        required = required_fields.get(entity_key, [])
        for field in required:
            if not entity.get(field) or entity.get(field) == "" or entity.get(field) is None:
                return False
        return True


    def _generate_followup_questions(self, entity_key: str, missing_fields: List[str], value: dict = None) -> List[str]:
        questions = []
        
        for field in missing_fields:

            if value and value.get(field):
                continue
                
            if entity_key == "user.사용자" and field == "이름":
                questions.append("이름을 알려주세요.")
            elif entity_key == "user.약" and field == "약명":
                questions.append("어떤 약을 복용하셨나요?")
            elif entity_key == "user.일정" and field == "제목":
                questions.append("일정의 제목은 무엇인가요?")
            elif entity_key == "user.일정" and field == "날짜":
                questions.append("언제인가요?")
            elif entity_key == "user.기념일" and field == "제목":
                questions.append("기념일의 제목은 무엇인가요?")
            elif entity_key == "user.기념일" and field == "날짜":
                questions.append("언제인가요?")
            elif entity_key == "user.가족" and field == "관계":
                questions.append("가족 관계를 알려주세요.")
            elif entity_key == "user.건강상태" and field == "증상":
                questions.append("어떤 증상이 있으신가요?")
            elif entity_key == "user.물건" and field == "이름":
                questions.append("어떤 물건인가요?")
            elif entity_key == "user.식사" and field == "끼니":
                questions.append("어떤 끼니인가요? (아침/점심/저녁)")
            elif entity_key == "user.식사" and field == "메뉴":
                questions.append("무엇을 드셨나요?")
            elif entity_key == "user.식사" and field == "시간":
                questions.append("몇 시에 드셨나요?")
            elif entity_key == "user.취미" and field == "이름":
                questions.append("취미가 무엇인가요?")
            elif entity_key == "user.취향" and field == "값":
                questions.append("어떤 것을 좋아하시나요?")
        
        return questions

    def _consolidate_followup_questions(self, questions: List[str]) -> str:
        if not questions:
            return ""
        
        if len(questions) == 1:
            return questions[0]
        

        entity_questions = {}
        for q in questions:
            if "약" in q:
                entity_questions.setdefault("약", []).append(q)
            elif "식사" in q or "끼니" in q or "드셨" in q:
                entity_questions.setdefault("식사", []).append(q)
            elif "일정" in q:
                entity_questions.setdefault("일정", []).append(q)
            elif "기념일" in q or "생일" in q or "기념" in q:
                entity_questions.setdefault("기념일", []).append(q)
            elif "가족" in q or "아빠" in q or "엄마" in q or "형" in q or "누나" in q or "언니" in q or "동생" in q:
                entity_questions.setdefault("가족", []).append(q)
            else:
                entity_questions.setdefault("기타", []).append(q)
        

        consolidated = []
        for entity, qs in entity_questions.items():
            if entity == "약":
                if len(qs) == 1:
                    consolidated.append(qs[0])
                else:

                    if qs:
                        consolidated.extend(qs)
            elif entity == "식사":
                if len(qs) == 1:
                    consolidated.append(qs[0])
                else:

                    has_meal_type = any("끼니" in q or "아침" in q or "점심" in q or "저녁" in q for q in qs)
                    has_menu = any("무엇" in q for q in qs)
                    has_time = any("몇 시" in q or "시간" in q for q in qs)
                    
                    if has_meal_type and has_menu and has_time:
                        consolidated.append("식사에 대해 알려주셔서 고마워요. 어떤 끼니에 무엇을 몇 시에 드셨나요?")
                    elif has_menu and has_time:
                        consolidated.append("식사에 대해 알려주셔서 고마워요. 무엇을 드셨고, 몇 시에 드셨나요?")
                    elif has_meal_type and has_menu:
                        consolidated.append("식사에 대해 알려주셔서 고마워요. 어떤 끼니에 무엇을 드셨나요?")
                    elif has_meal_type and has_time:
                        consolidated.append("식사에 대해 알려주셔서 고마워요. 어떤 끼니에 몇 시에 드셨나요?")
                    elif has_menu:
                        consolidated.append("식사에 대해 알려주셔서 고마워요. 무엇을 드셨나요?")
                    elif has_time:
                        consolidated.append("식사에 대해 알려주셔서 고마워요. 몇 시에 드셨나요?")
                    elif has_meal_type:
                        consolidated.append("식사에 대해 알려주셔서 고마워요. 어떤 끼니인가요?")
                    else:
                        consolidated.extend(qs)
            elif entity == "기념일":
                if len(qs) == 1:
                    consolidated.append(qs[0])
                else:
                    consolidated.append("기념일에 대해 알려주셔서 고마워요. 추가로 날짜나 관계 정보도 말씀해주실 수 있나요?")
            elif entity == "가족":
                if len(qs) == 1:
                    consolidated.append(qs[0])
                else:
                    consolidated.append("가족에 대해 알려주셔서 고마워요. 추가로 이름이나 관계 정보도 말씀해주실 수 있나요?")
            else:
                consolidated.extend(qs)
        
        return " ".join(consolidated)

    def _get_confirmed_user_name(self, session_id: str) -> str:
        try:

            docs = self.vectorstore.get()
            for i, doc_id in enumerate(docs.get("ids", [])):

                if "_user.사용자_" in doc_id:
                    data = json.loads(docs["documents"][i])

                    if (data.get("이름") and data.get("확인됨")):

                        if data.get("별칭"):
                            return data["별칭"]
                        return data["이름"]
        except Exception as e:
            print(f"[WARN] 사용자 이름 조회 실패: {e}")
        
        return "사용자"

    def _analyze_entity_context(self, user_input: str, existing_entity: dict, new_entity: dict, entity_key: str) -> dict:
        try:

            prompt = f"""
사용자의 발화를 분석하여 기존 엔티티와 새 엔티티가 같은 대상을 가리키는지 판단해주세요.

사용자 발화: "{user_input}"

기존 엔티티: {existing_entity}
새 엔티티: {new_entity}

분석 기준:
1. 사용자가 "다른", "새로운", "또 다른" 등의 표현을 사용했는가?
2. 사용자가 "같은", "그", "이전에 말한" 등의 표현을 사용했는가?
3. 문맥상 같은 대상을 가리키는 것 같은가?

응답 형식: {{"is_same_entity": true/false, "reason": "판단 이유"}}
"""
            
            response = self.llm.invoke(prompt)
            result = json.loads(response.content)
            
            print(f"[DEBUG] 문맥 분석 결과: {result}")
            return result
            
        except Exception as e:
            print(f"[WARN] 문맥 분석 실패: {e}")

            return {"is_same_entity": True, "reason": "분석 실패로 인한 기본값"}

    def _find_item_location(self, user_input: str, session_id: str) -> dict:
        try:

            item_keywords = []
            

            patterns = [
                r"(.+?)\s*(?:어디|위치|있어|두었|놓았)",
                r"(.+?)\s*(?:가져다|가져와|찾아)",
                r"(.+?)\s*(?:찾아|찾아줘)"
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, user_input)
                for match in matches:
                    item = match.strip()
                    if len(item) >= 1 and item not in ["어디", "위치", "있어", "두었", "놓았", "가져다", "가져와", "찾아", "찾아줘"]:
                        item_keywords.append(item)
            
            if not item_keywords:
                return None
            

            docs = self.vectorstore.similarity_search(" ".join(item_keywords), k=20)
            
            if not docs:
                return None
            

            for doc in docs:
                try:
                    data = json.loads(doc.page_content)
                    

                    if data.get("entity_key") == "user.물건":
                        item_name = data.get("이름", "")
                        location = data.get("위치", "")
                        
                        if location and any(keyword in item_name for keyword in item_keywords):

                            normalized_location = self._normalize_location(location)
                            return {"이름": item_name, "위치": normalized_location}
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            print(f"[WARN] 물건 위치 검색 실패: {e}")
            return None

    def _handle_location_query(self, user_input: str, session_id: str) -> str:
        location_info = self._find_item_location(user_input, session_id)
        if location_info:
            return f"{location_info['이름']}은 {location_info['위치']}에 있어요."
        else:
            return "죄송해요, 그 물건의 위치를 모르겠어요. 어디에 두었는지 알려주시면 기억해둘게요!"


    def _build_personalized_emotional_reply(self, user_input: str, session_id: str) -> str:
        try:

            user_name = self._get_confirmed_user_name(session_id)
            facts_text = self._get_facts_text(session_id)
            

            context_info = []
            if user_name:
                context_info.append(f"사용자의 이름은 {user_name}입니다.")
            if facts_text:
                context_info.append(f"저장된 정보: {facts_text}")
            
            context_text = "\n".join(context_info) if context_info else ""
            

            if context_text:
                prompt = (
                    "당신은 사용자의 감정을 깊이 이해하고 공감하는 생활 지원 로봇입니다.\n"
                    "사용자의 감정 상태를 파악하고, 그 감정에 맞는 따뜻하고 진심 어린 응답을 해주세요.\n"
                    "조언보다는 먼저 공감하고, 사용자가 혼자가 아니라는 것을 느끼도록 해주세요.\n"
                    "답변은 1-2문장으로 간결하게, **항상 존댓말로** 응답해주세요.\n"
                    "저장된 정보는 실제 VectorStore에 저장된 사실만 사용하세요.\n\n"
                    f"[사용자 정보]\n{context_text}\n\n"
                    f"사용자: {user_input}\n"
                    "로봇:"
                )
            else:

                user_name_confirmed = bool(self._get_confirmed_user_name(session_id))
                return build_emotional_reply(user_input, llm=self.llm, user_name_confirmed=user_name_confirmed)
            
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            print(f"[WARN] 개인화된 감정 응답 생성 실패: {e}")

            user_name_confirmed = bool(self._get_confirmed_user_name(session_id))
            return build_emotional_reply(user_input, llm=self.llm, user_name_confirmed=user_name_confirmed)

    def _prevent_name_family_conflict(self, entities: Dict[str, List[Dict[str, Any]]]) -> None:

        family_names = set()
        for family_entity in entities.get("user.가족", []):
            if "이름" in family_entity:
                family_names.add(family_entity["이름"])
        

        for user_entity in entities.get("user.사용자", []):
            if "이름" in user_entity:
                user_name = user_entity["이름"]
                if user_name in family_names:


                    user_entity.pop("이름", None)

    def _normalize_name(self, name: str) -> str:
        if not name: 
            return name
        

        name = name.strip().replace(" ", "")
        

        name = re.sub(r"(이라고\s*해|라고\s*해|라\s*해|이라고\s*불러|라고\s*불러|라\s*불러)$", "", name).strip()
        

        name = re.sub(r"(이야|이에요|입니다|예요|이고|이고요|야|다|어|아)$", "", name).strip()
        

        name = re.sub(r"(님|씨)$", "", name).strip()
        
        return name

    def _normalize_location(self, location: str) -> str:
        if not location:
            return location
        

        location = re.sub(r"(에|에서|으로|쪽에|안에|밖에|옆에|앞에|뒤에|아래에)+$", "", location).strip()
        

        relative_position_pattern = r"(.+?)(위|아래|옆|앞|뒤|왼쪽|오른쪽|가운데|중앙|중간)$"
        m = re.match(relative_position_pattern, location)
        
        if m:
            base_location = m.group(1).strip()
            relative_pos = m.group(2)
            

            base_normalized = self._normalize_base_location(base_location)
            

            return f"{base_normalized}({relative_pos})"
        

        return self._normalize_base_location(location)
    
    def _normalize_base_location(self, location: str) -> str:

        location_normalize_map = {

            "거실": "거실", "방": "거실", "응접실": "거실", "라운지": "거실",
            "침실": "침실", "자기방": "침실", "개인방": "침실",
            "부엌": "부엌", "주방": "부엌", "요리실": "부엌",
            "화장실": "화장실", "욕실": "화장실", "세면실": "화장실",
            "다용도실": "다용도실", "서재": "다용도실", "작업실": "다용도실",
            "베란다": "베란다", "발코니": "베란다", "테라스": "베란다",
            "지하": "지하", "지하실": "지하", "지하층": "지하",
            "옥상": "옥상", "루프탑": "옥상",
            

            "화장지": "화장지", "휴지": "화장지", "두루마리": "화장지",
            "냉장고": "냉장고", "냉동고": "냉장고",
            "책상": "책상", "데스크": "책상", "작업대": "책상",
            "침대": "침대", "베드": "침대", "매트리스": "침대",
            "소파": "소파", "쇼파": "소파", "의자": "소파",
            "테이블": "테이블", "탁자": "테이블", "식탁": "테이블",
            "옷장": "옷장", "장롱": "옷장", "드레스룸": "옷장",
            "서랍": "서랍", "서랍장": "서랍", "수납함": "서랍",
            "선반": "선반", "책장": "선반", "수납장": "선반",
            

            "앞": "앞", "앞쪽": "앞", "정면": "앞",
            "뒤": "뒤", "뒤쪽": "뒤", "후면": "뒤",
            "왼쪽": "왼쪽", "좌측": "왼쪽", "왼편": "왼쪽",
            "오른쪽": "오른쪽", "우측": "오른쪽", "오른편": "오른쪽",
            "위": "위", "위쪽": "위", "상단": "위",
            "아래": "아래", "아래쪽": "아래", "하단": "아래",
            "가운데": "가운데", "중앙": "가운데", "중간": "가운데",
            "옆": "옆", "옆쪽": "옆", "측면": "옆",
            

            "여기": "여기", "이곳": "여기", "현재위치": "여기",
            "저기": "저기", "그곳": "저기", "저쪽": "저기",
            "어디": "어디", "어디선가": "어디", "어딘가": "어디",
        }
        

        return location_normalize_map.get(location, location)

    def _merge_entity_values(self, old_value: dict, new_value: dict, entity_key: str) -> dict:
        if not old_value:
            return new_value
        if not new_value:
            return old_value
        

        merged = {**old_value, **new_value}
        

        if entity_key.endswith("사용자"):

            if "별칭" in new_value and "별칭" in old_value:

                if old_value["별칭"] != new_value["별칭"]:
                    merged["별칭"] = [old_value["별칭"], new_value["별칭"]]
                else:
                    merged["별칭"] = new_value["별칭"]
            elif "별칭" in old_value and "별칭" not in new_value:
                merged["별칭"] = old_value["별칭"]
            elif "별칭" in new_value and "별칭" not in old_value:
                merged["별칭"] = new_value["별칭"]
        
        elif entity_key.endswith("약"):

            if "복용" in old_value and "복용" in new_value:
                merged["복용"] = (old_value["복용"] or []) + (new_value["복용"] or [])
            elif "복용" in old_value:
                merged["복용"] = old_value["복용"]
            elif "복용" in new_value:
                merged["복용"] = new_value["복용"]
        
        elif entity_key.endswith("일정"):

            if "시간" in old_value and "시간" in new_value:
                old_time = self._normalize_time_field(old_value["시간"])
                new_time = self._normalize_time_field(new_value["시간"])
                merged["시간"] = sorted(list(set(old_time + new_time))) if (old_time or new_time) else None
            elif "시간" in old_value:
                merged["시간"] = old_value["시간"]
            elif "시간" in new_value:
                merged["시간"] = new_value["시간"]
        
        elif entity_key.endswith("식사"):

            if "메뉴" in old_value and "메뉴" in new_value:
                old_menus = old_value["메뉴"] if isinstance(old_value["메뉴"], list) else [old_value["메뉴"]]
                new_menus = new_value["메뉴"] if isinstance(new_value["메뉴"], list) else [new_value["메뉴"]]
                merged["메뉴"] = list(set(old_menus + new_menus))
            elif "메뉴" in old_value:
                merged["메뉴"] = old_value["메뉴"]
            elif "메뉴" in new_value:
                merged["메뉴"] = new_value["메뉴"]
        
        elif entity_key.endswith("물건"):

            if "설명" in old_value and "설명" in new_value:
                old_desc = old_value["설명"] if isinstance(old_value["설명"], list) else [old_value["설명"]]
                new_desc = new_value["설명"] if isinstance(new_value["설명"], list) else [new_value["설명"]]
                merged["설명"] = list(set(old_desc + new_desc))
            elif "설명" in old_value:
                merged["설명"] = old_value["설명"]
            elif "설명" in new_value:
                merged["설명"] = new_value["설명"]
        
        return merged

    def _extract_period(self, text: str) -> Optional[str]:
        period_patterns = [
            r"(일주일치|\d+일치|\d+주일치|\d+개월치|\d+년치)",
            r"(일주일|\d+일\s*치|\d+주\s*일\s*치|\d+개\s*월\s*치|\d+년\s*치)",
            r"(일주일분|\d+일분|\d+주분|\d+개월분|\d+년분)",
            r"(일주일\s*분|\d+일\s*분|\d+주\s*분|\d+개\s*월\s*분|\d+년\s*분)"
        ]
        
        for pattern in period_patterns:
            m = re.search(pattern, text)
            if m:
                return m.group(1)
        return None

    def _extract_meal_relation(self, text: str) -> Optional[str]:
        if "식후" in text:
            return "식후"
        if "식전" in text:
            return "식전"
        return None

    def _extract_drugs_with_info(self, text: str) -> List[Dict[str, Any]]:
        results = []
        seen_drugs = set()
        

        non_drug_words = {
            "예약", "약속", "약속시간", "약속장소", "약속일",
            "치약", "세정약", "세정제", "세정액", "세정용품",
            "약속", "약속시간", "약속장소", "약속일", "약속시간", "약속장소"
        }
        

        drug_patterns = [
            r"([가-힣A-Za-z]+약)",
            r"(아스피린|타이레놀|이부프로펜|아세트아미노펜)",
        ]
        
        all_drugs = []
        for pattern in drug_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in non_drug_words:
                    all_drugs.append(match)
        

        for drug in all_drugs:

            if drug in seen_drugs:
                continue
            seen_drugs.add(drug)
            
            drug_info = {"약명": drug}
            

            sentences = re.split(r'[.,]', text)
            
            for sentence in sentences:
                if drug in sentence:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    

                    dosages = self._extract_dosage(sentence)
                    if dosages:
                        drug_info["복용"] = dosages


                    meal_relation = self._extract_meal_relation(sentence)
                    if meal_relation:
                        drug_info["식사와의 관계"] = meal_relation


                    period = self._extract_period(sentence)
                    if period:
                        drug_info["복용 기간"] = period
                    
                    break

            results.append(drug_info)

        return results

    def _extract_dosage(self, text: str) -> List[Dict[str, str]]:
        dosage_patterns = [
            r"(하루\s*에?\s*\d+번|\d+번\s*복용)",
            r"(아침|점심|저녁)",
            r"(\d+시\s*\d+분?)",
        ]
        
        dosages = []
        seen_dosages = set()
        
        for pattern in dosage_patterns:
            matches = re.findall(pattern, text)
            for match in matches:

                normalized_match = re.sub(r'\s+', ' ', match.strip())
                if normalized_match not in seen_dosages:
                    seen_dosages.add(normalized_match)
                    dosages.append({"원문": normalized_match})
        

        return dosages if dosages else None

    def _add_to_date_cache(self, date_str: str, normalized_date: str) -> None:
        self.date_cache[date_str] = normalized_date
        

        if len(self.date_cache) > self.max_date_cache_size:

            oldest_key = next(iter(self.date_cache))
            del self.date_cache[oldest_key]
            logger.debug(f"날짜 캐시 크기 제한으로 오래된 항목 제거: '{oldest_key}'")

    def _normalize_date(self, date_str: str, session_id: str = None) -> str:
        if not date_str:
            return date_str
        

        if date_str in self.date_cache:
            logger.debug(f"날짜 캐시 hit: '{date_str}' → '{self.date_cache[date_str]}'")
            return self.date_cache[date_str]
        
        now = datetime.now()
        

        relative_dates = {
            "오늘": now,
            "현재": now,
            "지금": now,
            "내일": now + timedelta(days=1),
            "다음날": now + timedelta(days=1),
            "모레": now + timedelta(days=2),
            "이틀후": now + timedelta(days=2),
            "어제": now - timedelta(days=1),
            "하루전": now - timedelta(days=1),
            "그저께": now - timedelta(days=2),
            "이틀전": now - timedelta(days=2),
            "그제": now - timedelta(days=2),
            "3일전": now - timedelta(days=3),
            "일주일전": now - timedelta(days=7),
            "한주전": now - timedelta(days=7),
            "일주일후": now + timedelta(days=7),
            "한주후": now + timedelta(days=7)
        }
        

        if "다음 주" in date_str or "다음주" in date_str:

            if "금요일" in date_str:

                days_ahead = 4 - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "월요일" in date_str:
                days_ahead = 0 - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "화요일" in date_str:
                days_ahead = 1 - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "수요일" in date_str:
                days_ahead = 2 - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "목요일" in date_str:
                days_ahead = 3 - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "토요일" in date_str:
                days_ahead = 5 - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "일요일" in date_str:
                days_ahead = 6 - now.weekday()
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        

        if date_str in relative_dates:
            result = relative_dates[date_str].strftime('%Y-%m-%d')
            self._add_to_date_cache(date_str, result)
            return result
        

        weekday_map = {
            "월요일": 0, "화요일": 1, "수요일": 2, "목요일": 3, 
            "금요일": 4, "토요일": 5, "일요일": 6
        }
        

        weekday_pattern = r"(이번주|다음주|다다음주)\s*(월요일|화요일|수요일|목요일|금요일|토요일|일요일)"
        m = re.match(weekday_pattern, date_str.strip())
        if m:
            week_type = m.group(1)
            weekday_name = m.group(2)
            

            if week_type == "이번주":
                base = now
            elif week_type == "다음주":
                base = now + timedelta(weeks=1)
            else:
                base = now + timedelta(weeks=2)
            
            target_weekday = weekday_map[weekday_name]
            

            days_ahead = (target_weekday - base.weekday()) % 7
            target_date = base + timedelta(days=days_ahead)
            
            result = target_date.strftime('%Y-%m-%d')
            self._add_to_date_cache(date_str, result)
            return result
        

        single_weekday_pattern = r"(월요일|화요일|수요일|목요일|금요일|토요일|일요일)$"
        m = re.match(single_weekday_pattern, date_str.strip())
        if m:
            weekday_name = m.group(1)
            target_weekday = weekday_map[weekday_name]
            

            days_ahead = (target_weekday - now.weekday()) % 7
            target_date = now + timedelta(days=days_ahead)
            
            result = target_date.strftime('%Y-%m-%d')
            self._add_to_date_cache(date_str, result)
            return result
        


        month_day_pattern = r"(\d{1,2})월\s*(\d{1,2})일"
        m = re.match(month_day_pattern, date_str.strip())
        if m:
            month = int(m.group(1))
            day = int(m.group(2))
            

            try:
                target_date = datetime(now.year, month, day)

                if target_date < now:
                    target_date = datetime(now.year + 1, month, day)
                result = target_date.strftime('%Y-%m-%d')
                self._add_to_date_cache(date_str, result)
                return result
            except ValueError:
                pass
        

        year_month_day_pattern = r"(올해|내년)?\s*(\d{1,2})월\s*(\d{1,2})일"
        m = re.match(year_month_day_pattern, date_str.strip())
        if m:
            year_type = m.group(1)
            month = int(m.group(2))
            day = int(m.group(3))
            
            try:
                if year_type == "내년":
                    target_date = datetime(now.year + 1, month, day)
                else:
                    target_date = datetime(now.year, month, day)

                    if target_date < now:
                        target_date = datetime(now.year + 1, month, day)
                
                result = target_date.strftime('%Y-%m-%d')
                self._add_to_date_cache(date_str, result)
                return result
            except ValueError:
                pass
        

        relative_patterns = [

            (r"(\d+)\s*일\s*(뒤|후)", lambda m: now + timedelta(days=int(m.group(1)))),
            (r"일주일\s*(뒤|후)", lambda m: now + timedelta(days=7)),
            (r"한주\s*(뒤|후)", lambda m: now + timedelta(days=7)),

            (r"(\d+)\s*주\s*(뒤|후)", lambda m: now + timedelta(weeks=int(m.group(1)))),

            (r"(\d+)\s*개월\s*(뒤|후)", lambda m: now + timedelta(days=int(m.group(1)) * 30)),

            (r"(\d+)\s*년\s*(뒤|후)", lambda m: now + timedelta(days=int(m.group(1)) * 365)),
        ]
        
        for pattern, handler in relative_patterns:
            m = re.match(pattern, date_str.strip())
            if m:
                try:
                    target_date = handler(m)
                    result = target_date.strftime('%Y-%m-%d')
                    self._add_to_date_cache(date_str, result)
                    return result
                except (ValueError, OverflowError):
                    pass
        

        natural_date_patterns = [

            (r"(이번\s*달|다음\s*달|다다음\s*달)\s*(\d{1,2})일", self._parse_month_day_natural),


            (r"(이번\s*달|다음\s*달|다다음\s*달)\s*마지막\s*날", self._parse_month_last_day),

            (r"(이번\s*달|다음\s*달|다다음\s*달)\s*첫째\s*주\s*(월요일|화요일|수요일|목요일|금요일|토요일|일요일)", self._parse_month_first_week),
        ]
        
        for pattern, handler in natural_date_patterns:
            m = re.search(pattern, date_str.strip())
            if m:
                try:
                    result = handler(m, now)
                    if result:
                        logger.debug(f"자연어 날짜 처리 성공: '{date_str}' → '{result}'")
                        self._add_to_date_cache(date_str, result)
                        return result
                except Exception as e:
                    logger.debug(f"자연어 날짜 처리 실패: {str(e)[:50]}...")
                    pass
        

        try:
            parsed_date = dateparser.parse(
                date_str, 
                settings={
                    'RELATIVE_BASE': now,
                    'PREFER_DATES_FROM': 'future',
                    'DATE_ORDER': 'YMD'
                }, 
                languages=['ko', 'en']
            )
            if parsed_date:
                result = parsed_date.strftime('%Y-%m-%d')
                self._add_to_date_cache(date_str, result)
                return result
        except Exception:
            pass
        

        try:
            llm_prompt = f"""
사용자가 말한 날짜 표현: "{date_str}"
오늘 날짜: {now.strftime('%Y-%m-%d')}

1. YYYY-MM-DD 형식으로 변환하세요.
2. 확실하지 않으면 guess하지 말고 confidence를 "low"로 표시하세요.
3. 애매하거나 모호한 표현은 date를 null로 두고 confidence를 "low"로 설정하세요.

출력 JSON 형식:
{{
    "date": "<YYYY-MM-DD or null>",
    "confidence": "high" | "low"
}}

예시:
- "다음주 금요일" → {{"date": "2025-09-26", "confidence": "high"}}
- "다음 달 15일" → {{"date": "2025-10-15", "confidence": "high"}}
- "올해 크리스마스" → {{"date": "2025-12-25", "confidence": "high"}}
- "설날" → {{"date": null, "confidence": "low"}} (연도 불명확)
- "다음 생일" → {{"date": null, "confidence": "low"}} (구체적 날짜 불명확)
"""
            
            response = self.llm.invoke(llm_prompt)
            if hasattr(response, 'content'):

                try:
                    result = json.loads(str(response.content))

                    if result.get("confidence") == "high" and result.get("date"):
                        self._add_to_date_cache(date_str, result["date"])
                        return result["date"]

                    elif result.get("confidence") == "low":
                        return None
                except (json.JSONDecodeError, TypeError):
                    pass
        except Exception:
            pass
        

        return date_str

    def _parse_month_day_natural(self, match, now: datetime) -> str:
        month_type = match.group(1)
        day = int(match.group(2))
        
        if "이번" in month_type:
            target_month = now.month
            target_year = now.year
        elif "다음" in month_type:
            if now.month == 12:
                target_month = 1
                target_year = now.year + 1
            else:
                target_month = now.month + 1
                target_year = now.year
        else:
            if now.month >= 11:
                target_month = now.month + 2 - 12
                target_year = now.year + 1
            else:
                target_month = now.month + 2
                target_year = now.year
        
        try:
            target_date = datetime(target_year, target_month, day)
            return target_date.strftime('%Y-%m-%d')
        except ValueError:
            return None

    def _parse_month_last_day(self, match, now: datetime) -> str:
        month_type = match.group(1)
        
        if "이번" in month_type:
            target_month = now.month
            target_year = now.year
        elif "다음" in month_type:
            if now.month == 12:
                target_month = 1
                target_year = now.year + 1
            else:
                target_month = now.month + 1
                target_year = now.year
        else:
            if now.month >= 11:
                target_month = now.month + 2 - 12
                target_year = now.year + 1
            else:
                target_month = now.month + 2
                target_year = now.year
        

        if target_month == 12:
            next_month = 1
            next_year = target_year + 1
        else:
            next_month = target_month + 1
            next_year = target_year
        
        last_day = datetime(next_year, next_month, 1) - timedelta(days=1)
        return last_day.strftime('%Y-%m-%d')

    def _parse_month_first_week(self, match, now: datetime) -> str:
        month_type = match.group(1)
        weekday_name = match.group(2)
        
        weekday_map = {
            "월요일": 0, "화요일": 1, "수요일": 2, "목요일": 3, 
            "금요일": 4, "토요일": 5, "일요일": 6
        }
        target_weekday = weekday_map[weekday_name]
        
        if "이번" in month_type:
            target_month = now.month
            target_year = now.year
        elif "다음" in month_type:
            if now.month == 12:
                target_month = 1
                target_year = now.year + 1
            else:
                target_month = now.month + 1
                target_year = now.year
        else:
            if now.month >= 11:
                target_month = now.month + 2 - 12
                target_year = now.year + 1
            else:
                target_month = now.month + 2
                target_year = now.year
        

        first_day = datetime(target_year, target_month, 1)
        days_ahead = (target_weekday - first_day.weekday()) % 7
        target_date = first_day + timedelta(days=days_ahead)
        
        return target_date.strftime('%Y-%m-%d')

    def _check_date_normalization_failure(self, entities: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
        if not entities:
            return None
            

        date_entity_types = ["user.일정", "user.약", "user.식사", "user.기념일", "user.건강상태"]
        
        for entity_type, entity_list in entities.items():
            if entity_type in date_entity_types and isinstance(entity_list, list):
                for entity in entity_list:
                    if isinstance(entity, dict) and "날짜" in entity:
                        date_value = entity["날짜"]

                        if date_value and not self._is_normalized_date(date_value):

                            normalized_date = self._normalize_date(date_value)

                            if normalized_date == date_value or not self._is_normalized_date(normalized_date):

                                if entity_type == "user.일정" and "제목" in entity and entity["제목"] == date_value:
                                    continue

                                if re.match(r'\d{1,2}월\s*\d{1,2}일', date_value):
                                    continue

                                if entity_type == "user.일정" and any(re.search(pattern, date_value) for pattern in [r'(여행|파티|회의|약속|미팅|데이트|일정|스케줄|예약)', r'(병원|치과|약국|은행|우체국)']):
                                    continue
                                return f"'{date_value}'라는 날짜 표현이 명확하지 않네요. 구체적인 날짜로 말씀해주실래요? (예: '다음주 금요일', '12월 25일' 등)"
        
        return None

    def _is_normalized_date(self, date_str: str) -> bool:
        if not date_str:
            return False
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def _extract_name_llm(self, text: str) -> Optional[Dict[str, str]]:
        try:
            llm_prompt = f"""
다음 사용자 발화에서 **사용자 본인의 이름과 별칭**만 추출해주세요.

발화: "{text}"

중요 규칙:
1. **사용자 본인의 이름만** 추출 (한글 2-4글자, 영문 이름, 별명 등)
2. **다른 사람의 이름은 절대 제외** (가족, 친구, 동료, 선생님 등 모든 관계의 사람 이름)
3. **"내 이름은", "나는", "저는", "난", "내가" 등이 포함된 경우만** 사용자 이름으로 인식
4. **문맥을 정확히 파악하여 사용자 본인 이름만 추출**
5. **관계가 언급된 경우 (동생, 엄마, 아빠, 남자친구, 동료 등) 그 사람의 이름은 사용자 이름이 아님**
6. **특히 주의할 것:**
   - "우리 엄마 이름은 김영희야" → 추출하지 않음 (가족 이름)
   - "나는 홍길동이야" → 추출함 (사용자 이름)
   - "동생 철수" → 추출하지 않음 (가족 이름)
   - "내 동생은 현우야" → 추출하지 않음 (가족 이름)
7. 별칭이 있다면 함께 추출 ("편하게 서연이라고 불러도 돼" → 별칭: "서연")
8. 사용자 본인 이름이 없으면 null 반환

JSON 형식으로 응답:
{{"name": "사용자본인이름", "alias": "별칭" 또는 null, "confidence": 0.0-1.0}}

confidence는 이름 추출의 확신도를 나타냅니다:
- 0.9-1.0: 매우 확신 (명확한 이름 표현)
- 0.7-0.8: 높은 확신 (이름 표현이 있음)
- 0.5-0.6: 중간 확신 (추측 가능)
- 0.0-0.4: 낮은 확신 (불확실)

예시 (올바른 추출):
- "내 이름은 사실 권서연인데" → {{"name": "권서연", "alias": null, "confidence": 0.9}}
- "편하게 서연이라고 불러" → {{"name": null, "alias": "서연", "confidence": 0.8}}
- "나 권서연이야" → {{"name": "권서연", "alias": null, "confidence": 0.9}}
- "내 이름은 권서연이야. 근데 편하게 서연이라고 불러도 돼" → {{"name": "권서연", "alias": "서연", "confidence": 0.9}}

예시 (가족 이름 - 추출하지 않음):
- "우리 동생 이름은 임성현이고" → {{"name": null, "alias": null, "confidence": 0.9}}
- "내동생이름은 엄성현이야" → {{"name": null, "alias": null, "confidence": 0.9}}
- "내엄마이름은 전지현이야" → {{"name": null, "alias": null, "confidence": 0.9}}
- "엄마 이름은 전지현이야" → {{"name": null, "alias": null, "confidence": 0.9}}
- "아빠는 김민수라고 해" → {{"name": null, "alias": null, "confidence": 0.9}}
- "우리 형은 김철수야" → {{"name": null, "alias": null, "confidence": 0.9}}
- "동생 현우" → {{"name": null, "alias": null, "confidence": 0.9}}
- "할머니 최영희" → {{"name": null, "alias": null, "confidence": 0.9}}

예시 (기타 - 추출하지 않음):
- "사실 좋아해" → {{"name": null, "alias": null, "confidence": 0.2}}
- "편하게 불러도 돼" → {{"name": null, "alias": null, "confidence": 0.1}}
"""
            
            response = self.llm.invoke(llm_prompt)
            if hasattr(response, 'content'):

                try:
                    result = json.loads(str(response.content))
                    name = result.get('name')
                    alias = result.get('alias')
                    confidence = result.get('confidence', 0.0)
                    

                    if confidence >= 0.7 and name and self._is_valid_name(name):

                        normalized_name = self._normalize_name(name)
                        if normalized_name and self._is_valid_name(normalized_name):
                            logger.debug(f"LLM 이름 추출 성공: name='{normalized_name}', alias='{alias}', confidence={confidence}")
                            return {"name": normalized_name, "alias": alias, "confidence": confidence}
                        else:
                            logger.debug(f"LLM 이름 추출 실패: 정규화 후 유효하지 않은 이름 '{normalized_name}'")
                    else:
                        logger.debug(f"LLM 이름 추출 실패: 낮은 확신도({confidence}) 또는 유효하지 않은 이름 '{name}'")
                        
                except (json.JSONDecodeError, TypeError) as e:
                    logger.warning(f"이름 추출 JSON 파싱 실패: {str(e)[:50]}...")
        except Exception as e:
            logger.warning(f"이름 추출 실패: {str(e)[:50]}...")
        return None

    def _analyze_emotional_state(self, text: str) -> dict:
        negative_keywords = ["피곤", "힘들", "어지럽", "바닥", "슬퍼", "우울", "짜증", "화나", "답답해", "괴로워", "아픔", "상처", "실망"]
        positive_keywords = ["좋아", "기뻐", "행복", "신나", "즐거워", "만족", "뿌듯", "기쁘", "웃음", "즐겁"]
        
        text_lower = text.lower()
        negative_count = sum(1 for word in negative_keywords if word in text_lower)
        positive_count = sum(1 for word in positive_keywords if word in text_lower)
        
        if negative_count > positive_count:
            return {"mood": "negative", "intensity": min(negative_count * 0.3, 1.0)}
        elif positive_count > negative_count:
            return {"mood": "positive", "intensity": min(positive_count * 0.3, 1.0)}
        else:
            return {"mood": "neutral", "intensity": 0.5}

    def _should_maintain_emotional_context(self, session_id: str, current_category: str) -> bool:
        if session_id not in self.emotional_state:
            return False
        
        emotional_info = self.emotional_state[session_id]

        if (emotional_info.get("intensity", 0) > 0.6 and 
            emotional_info.get("last_emotional_turn", 0) <= 3 and
            current_category in ["cognitive", "physical"]):
            return True
        return False

    def _is_valid_name(self, name: str) -> bool:
        if not name or len(name) < 2:
            return False
        

        invalid_words = {
            "사실", "편하게", "그냥", "정말", "진짜", "완전", "너무", "정말로",
            "그러면", "그래서", "그런데", "하지만", "그리고", "그러나",
            "좋아", "싫어", "좋다", "싫다", "맞다", "틀리다",
            "이름", "나", "내", "저", "제", "우리", "너", "당신"
        }
        
        if name in invalid_words:
            return False
        

        if re.match(r'^[가-힣]{2,5}$', name):
            return True
        

        if re.match(r'^[a-zA-Z]{2,15}$', name):
            return True
        
        return False

    def _extract_nickname(self, text: str) -> Optional[str]:

        return None

    def _is_name_question(self, text: str) -> bool:
        question_patterns = [
            r"내\s*이름이?\s*뭐라고?",
            r"내\s*이름\s*알아?",
            r"내\s*이름\s*뭐야?",
            r"내\s*이름\s*뭐지?",
            r"내\s*이름\s*기억해?",
            r"내\s*이름\s*뭔지\s*알아?",
            r"내\s*이름\s*뭐였지?",
            r"내\s*이름\s*뭐였어?",
            r"내\s*이름\s*뭐였죠?",
            r"내\s*이름\s*뭐였나?",
            r"내\s*이름\s*뭐였더라?"
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in question_patterns)

    def _merge_user_entities(self, existing_entities: List[Dict], new_entity: Dict) -> List[Dict]:
        if not existing_entities:
            return [new_entity]
        

        existing = existing_entities[0]
        new_name = new_entity.get("이름", "")
        new_aliases = new_entity.get("별칭", [])
        

        if existing.get("이름") == new_name:

            if new_aliases:
                existing_aliases = existing.get("별칭", [])
                for alias in new_aliases:
                    if alias not in existing_aliases:
                        existing_aliases.append(alias)
                existing["별칭"] = existing_aliases
            return existing_entities
        

        existing_name = existing.get("이름", "")
        existing_aliases = existing.get("별칭", [])
        

        if new_name in existing_aliases:

            existing_aliases.remove(new_name)
            existing["이름"] = new_name
            existing["별칭"] = existing_aliases
            return existing_entities
        

        if existing_name in new_aliases:

            existing_aliases = existing.get("별칭", [])
            if existing_name not in existing_aliases:
                existing_aliases.append(existing_name)
            existing["이름"] = new_name
            existing["별칭"] = existing_aliases
            return existing_entities
        

        if self._is_name_variant(existing_name, new_name):

            if len(existing_name) > len(new_name):
                existing_aliases = existing.get("별칭", [])
                if new_name not in existing_aliases:
                    existing_aliases.append(new_name)
                existing["별칭"] = existing_aliases
            else:
                existing_aliases = existing.get("별칭", [])
                if existing_name not in existing_aliases:
                    existing_aliases.append(existing_name)
                existing["이름"] = new_name
                existing["별칭"] = existing_aliases
            return existing_entities
        

        return existing_entities + [new_entity]

    def _is_name_variant(self, name1: str, name2: str) -> bool:
        if not name1 or not name2:
            return False
        

        clean1 = name1.replace(" ", "")
        clean2 = name2.replace(" ", "")
        

        return clean1.endswith(clean2) or clean2.endswith(clean1)

    def _get_existing_user_entities(self, session_id: str) -> List[Dict]:
        try:
            docs = self.vectorstore.get()
            existing_users = []
            
            for i, doc_id in enumerate(docs.get("ids", [])):
                if doc_id.startswith(f"{session_id}_user.사용자"):
                    try:
                        data = json.loads(docs["documents"][i])
                        if (data.get("entity_key") == "user.사용자" and 
                            data.get("session_id") == session_id):
                            existing_users.append(data)
                    except (json.JSONDecodeError, TypeError):
                        continue
            
            return existing_users
        except Exception as e:
            print(f"[WARN] 기존 사용자 엔티티 조회 실패: {e}")
            return []

    def _process_single_entity(self, entity_key: str, filtered_value: Dict, session_id: str, questions: List[str], has_schedule: bool) -> bool:
        try:

            if entity_key.endswith("일정"):
                has_schedule = True
            

            

            if entity_key.endswith("식사") and "메뉴" in filtered_value:
                menus = filtered_value["메뉴"]
                if isinstance(menus, list):

                    filtered_menus = [menu for menu in menus if not menu.endswith("약")]
                    if not filtered_menus:

                        print(f"[INFO] 약을 식사로 착각한 엔티티 제거: {entity_key} - {filtered_value}")
                        return has_schedule
                    filtered_value["메뉴"] = filtered_menus
            

            missing_fields = self._check_missing_fields(entity_key, filtered_value)

            if missing_fields:

                logger.debug(f"누락된 필드 감지: {entity_key} - {missing_fields}, 값: {filtered_value}")
                followup_questions = self._generate_followup_questions(entity_key, missing_fields, filtered_value)
                questions.extend(followup_questions)
                

                if entity_key.endswith("식사") and (
                    ("메뉴" in missing_fields and filtered_value.get("메뉴") == []) or
                    ("시간" in missing_fields and not filtered_value.get("시간"))
                ):

                    final_value = self._add_to_vstore(
                        entity_key, filtered_value,
                        {"session_id": session_id, "entity_key": entity_key, "type": "entity", "created_at": datetime.now().isoformat()},
                        strategy="merge",
                        session_id=session_id
                    )
                    logger.debug(f"식사 엔티티 임시 저장 (메뉴/시간 누락): {filtered_value}")
                    

                    if "시간" in missing_fields and not filtered_value.get("시간"):
                        self.pending_question[session_id] = {
                            "기존_엔티티": final_value,
                            "새_엔티티": final_value,
                            "entity_key": entity_key
                        }
                        print(f"[DEBUG] 재질문 상태 설정: {entity_key} - 시간 누락")
                else:
                    return has_schedule


            final_value = self._add_to_vstore(
                entity_key, filtered_value,
                {"session_id": session_id, "entity_key": entity_key, "type": "entity", "created_at": datetime.now().isoformat()},
                strategy="merge"
            )


            if entity_key.endswith(".약"):
                if final_value.get("복용"):
                    enriched = [self._enrich_dose_dict(d) for d in final_value["복용"]]
                    final_value["복용"] = enriched
                    

                    try:
                        self._add_to_vstore(
                            entity_key=entity_key,
                            value=final_value,
                            metadata={"session_id": session_id, "type": "entity"},
                            strategy="merge",
                            identity=final_value.get("약명"),
                            session_id=session_id
                        )

                    except Exception as e:
                        print(f"[WARN] 복용 정보 enrich 후 업데이트 실패: {e}")
                    


                    pass
            
            return has_schedule
            
        except Exception as e:
            print(f"[ERROR] 엔티티 처리 실패: {e}")
            return has_schedule

    def _get_cached_classification(self, text: str, similarity_threshold: float = 0.85) -> Optional[Dict]:
        if not self.cache_texts or self.cache_embeddings is None:
            return None
        
        try:

            input_embedding = self.vectorizer.transform([text])
            

            similarities = cosine_similarity(input_embedding, self.cache_embeddings)[0]
            

            max_similarity_idx = np.argmax(similarities)
            max_similarity = similarities[max_similarity_idx]
            
            if max_similarity >= similarity_threshold:
                cached_text = self.cache_texts[max_similarity_idx]
                cached_result = self.classification_cache.get(cached_text)
                if cached_result:
                    logger.debug(f"유사도 {max_similarity:.3f}로 캐시된 결과 사용: '{cached_text}' → '{text}'")
                    return cached_result.to_dict()
        
        except Exception as e:
            logger.warning(f"캐시 검색 실패: {e}")
        
        return None

    def _add_to_cache(self, text: str, result: Dict) -> None:
        try:

            self.classification_cache[text] = result
            

            self.cache_texts.append(text)
            

            if len(self.cache_texts) > 100:

                old_text = self.cache_texts.pop(0)
                self.classification_cache.pop(old_text, None)
            

            if self.cache_texts:
                self.cache_embeddings = self.vectorizer.fit_transform(self.cache_texts)
            
        except Exception as e:
            logger.warning(f"캐시 추가 실패: {e}")

    def _sync_to_exact_cache(self, user_input: str, result_dict: Dict, pre_entities: Dict = None) -> None:
        try:

            from .task_classifier import _add_to_cache as add_exact_cache, ClassificationResult
            

            import re
            norm_text = re.sub(r"\s+", " ", user_input.strip().lower())
            

            category = result_dict.get("category", "query")
            confidence = result_dict.get("confidence", 0.5)
            probabilities = result_dict.get("probabilities", {category: 0.5})
            

            if confidence < CONFIDENCE_THRESHOLD:
                logger.warning(f"낮은 confidence로 인한 fallback: {confidence:.2f} < {CONFIDENCE_THRESHOLD}")

                from .task_classifier import classify
                category, _ = classify_hybrid(user_input, pre_entities)
                confidence = 0.5
                probabilities = {category: 0.5}
            
            classification_result = ClassificationResult(category, confidence, probabilities)
            

            add_exact_cache(norm_text, classification_result)
            logger.debug(f"정확 캐시 동기화: '{norm_text[:30]}...'")
            
        except Exception as e:
            logger.error(f"정확 캐시 동기화 실패: {str(e)[:50]}...")

    def _classify_with_cache(self, user_input: str, pre_entities: Dict[str, List[Dict[str, Any]]] = None) -> Dict:


        cached_result = self._get_cached_classification(user_input)
        if cached_result:
            logger.debug(f"유사 캐시 hit: '{user_input[:30]}...'")

            self._sync_to_exact_cache(user_input, cached_result, pre_entities)
            return cached_result
        

        try:
            classification_result = classify_hybrid(user_input, pre_entities)
            result_dict = classification_result.to_dict()
            

            self._add_to_cache(user_input, classification_result)
            self._sync_to_exact_cache(user_input, result_dict, pre_entities)
            
            return result_dict
            
        except Exception as e:
            logger.error(f"하이브리드 분류 실패, 기본 분류 사용: {e}")

            category, _ = classify_hybrid(user_input, pre_entities)
            fallback_result = {
                "category": category,
                "confidence": 0.5,
                "probabilities": {category: 0.5}
            }
            

            self._sync_to_exact_cache(user_input, fallback_result)
            return fallback_result

    def _normalize_duration(self, duration_str: str) -> str:
        if not duration_str:
            return duration_str
        

        duration_mapping = {
            "일주일치": "7일치",
            "일주일": "7일치", 
            "1주일치": "7일치",
            "1주일": "7일치",
            "이주일치": "14일치",
            "2주일치": "14일치",
            "한달치": "30일치",
            "1개월치": "30일치",
            "두달치": "60일치",
            "2개월치": "60일치",
            "세달치": "90일치",
            "3개월치": "90일치",
            "반년치": "180일치",
            "6개월치": "180일치",
            "일년치": "365일치",
            "1년치": "365일치"
        }
        

        if duration_str in duration_mapping:
            return duration_mapping[duration_str]
        

        import re
        patterns = [
            (r"(\d+)주일치", r"\1주일치"),
            (r"(\d+)주일", r"\1주일치"),
            (r"(\d+)개월치", r"\1개월치"),
            (r"(\d+)년치", r"\1년치"),
            (r"(\d+)일치", r"\1일치"),
            (r"(\d+)일분", r"\1일치"),
            (r"(\d+)주분", r"\1주일치"),
            (r"(\d+)개월분", r"\1개월치"),
            (r"(\d+)년분", r"\1년치")
        ]
        
        for pattern, replacement in patterns:
            if re.match(pattern, duration_str):
                return re.sub(pattern, replacement, duration_str)
        

        return duration_str

    def _extract_duration_from_dosage(self, dosage_list: List[dict]) -> Tuple[List[dict], str]:
        if not dosage_list:
            return dosage_list, None
        
        period_patterns = [
            r"(일주일치|\d+일치|\d+주일치|\d+개월치|\d+년치)",
            r"(일주일|\d+일\s*치|\d+주\s*일\s*치|\d+개\s*월\s*치|\d+년\s*치)",
            r"(일주일분|\d+일분|\d+주분|\d+개월분|\d+년분)",
            r"(일주일\s*분|\d+일\s*분|\d+주\s*분|\d+개\s*월\s*분|\d+년\s*분)"
        ]
        
        filtered_dosage = []
        extracted_period = None
        
        for dosage in dosage_list:
            if isinstance(dosage, dict) and "원문" in dosage:
                text = dosage["원문"]
                is_period = False
                
                for pattern in period_patterns:
                    if re.search(pattern, text):
                        if not extracted_period:
                            extracted_period = self._normalize_duration(text)
                        is_period = True
                        break
                
                if not is_period:
                    filtered_dosage.append(dosage)
            else:
                filtered_dosage.append(dosage)
        
        return filtered_dosage, extracted_period

    def _is_valid_entity(self, entity_key: str, value: dict) -> bool:
        if entity_key.endswith("사용자") and "이름" in value:
            name = value["이름"]

            if name in NAME_BLACKLIST or len(name) < 2:
                return False

            if len(name) == 1:
                return False
        
        if entity_key.endswith("물건") and "이름" in value:
            item_name = value["이름"]

            if item_name in {"물건", "거", "것", "뭐", "뭔가", "화", "알고", "다시"} or len(item_name) < 1:
                return False
        
        if entity_key.endswith("식사") and "메뉴" in value:
            menus = value["메뉴"]
            if isinstance(menus, list):

                valid_menus = [m for m in menus if m and m not in STOPWORDS and (len(m) > 1 or m == "밥")]
                if not valid_menus:
                    return False
        
        return True

    def maintenance_dedup_user(self, session_id: str):
        try:

            docs = self.vectorstore.similarity_search("사용자 이름", k=100)
            
            valid_docs = []
            invalid_ids = []
            
            for doc in docs:
                try:
                    data = json.loads(doc.page_content)

                    if (data.get("session_id") == session_id and 
                        data.get("entity_key") == "user.사용자" and
                        self._is_valid_entity("user.사용자", data)):
                        valid_docs.append(doc)
                    elif (data.get("session_id") == session_id and 
                          data.get("entity_key") == "user.사용자"):
                        invalid_ids.append(doc.metadata.get("id"))
                except Exception:
                    invalid_ids.append(doc.metadata.get("id"))
            

            if invalid_ids:
                self.vectorstore.delete(ids=invalid_ids)
                print(f"[MAINT] 잘못된 사용자 엔티티 {len(invalid_ids)}개 삭제")
            
            print(f"[MAINT] 유효한 사용자 엔티티 {len(valid_docs)}개 유지")
            
        except Exception as e:
            print(f"[WARN] 사용자 엔티티 정리 실패: {e}")

    def _extract_time_from_text(self, text: str) -> str:
        time_patterns = [
            r"(\d{1,2}시\s*반)",
            r"(\d{1,2}시\s*\d{1,2}분)",
            r"(\d{1,2}:\d{2})",
            r"(\d{1,2}시)",
            r"(오전\s*\d{1,2}시)",
            r"(오후\s*\d{1,2}시)",
            r"(새벽\s*\d{1,2}시)",
            r"(밤\s*\d{1,2}시)"
        ]
        for pattern in time_patterns:
            time_match = re.search(pattern, text)
            if time_match:
                return time_match.group(1)
        return None

    def _extract_menu_from_text(self, text: str) -> str:

        menu_patterns = [
            r"([가-힣]{2,10})",
        ]
        

        stopwords = {
             "고마워", "감사", "죄송", "미안", "알겠", "네", "아니", "그래", "맞아",
            "오늘", "어제", "내일", "아침", "점심", "저녁", "시간", "몇시", "언제",
            "먹었", "먹어", "드셨", "드셔", "식사", "밥", "음식", "메뉴"
        }
        
        for pattern in menu_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in stopwords and len(match) >= 2:
                    return match.strip()

        return None


    def _extract_hour_from_time(self, time_str: str) -> Optional[int]:
        if not time_str:
            return None
        
        try:
            if ":" in time_str:
                hour = int(time_str.split(":")[0])
                return hour
            elif "시" in time_str:
                hour = int(re.search(r"(\d{1,2})시", time_str).group(1))
                return hour
            else:
                return int(time_str)
        except (ValueError, IndexError, AttributeError):
            return None

    def _normalize_time_format(self, time_str: str) -> str:
        if not time_str:
            return time_str
        

        if re.match(r"\d{1,2}:\d{2}", time_str):
            hour, minute = time_str.split(":")
            return f"{hour}시 {minute}분"
        
        return time_str

    def _normalize_time_field(self, time_value) -> List[str]:
        if not time_value:
            return []
        if isinstance(time_value, list):
            return [str(t) for t in time_value if t]
        return [str(time_value)]

    def _entity_identity(self, entity_key: str, v: dict) -> str:
        if entity_key.endswith("사용자"): return "user_name"
        if entity_key.endswith("일정"):  return f"{v.get('제목')}|{v.get('날짜')}"
        if entity_key.endswith("물건"):  return v.get("이름")
        if entity_key.endswith("식사"):  return f"{v.get('날짜')}|{v.get('끼니')}"
        if entity_key.endswith("약"):    return v.get("약명")
        if entity_key.endswith("가족"):  return f"{v.get('관계')}|{v.get('이름')}"
        if entity_key.endswith("기념일"):return f"{v.get('관계')}|{v.get('제목')}|{v.get('날짜')}"
        if entity_key.endswith("취미"):  return v.get("이름")
        if entity_key.endswith("취향"):  return f"{v.get('종류')}|{v.get('값')}"
        if entity_key.endswith("건강상태"):
            return f"{v.get('질병')}|{v.get('증상')}|{v.get('기간')}"
        return json.dumps(v, ensure_ascii=False)

    def _squash_entities(self, session_id: str, entities: Dict[str, List[Dict[str, Any]]], user_input: str = None) -> Dict[str, List[Dict[str, Any]]]:
        out = {}
        for k, vals in entities.items():
            by_id: Dict[str, dict] = {}
            for v in vals:

                filtered_v = self._filter_meaningful_data(v, user_input)
                if not filtered_v:
                    continue
                    
                id_ = self._entity_identity(k, filtered_v)
                base = by_id.get(id_, {})

                for fk, fv in filtered_v.items():
                    if fv in (None, "", []): 
                        continue
                    if isinstance(fv, list) and isinstance(base.get(fk), list):

                        combined = base[fk] + fv
                        base[fk] = self._dedup_entities(combined) if all(isinstance(item, dict) for item in combined) else list({*base[fk], *fv})
                    else:

                        if not base.get(fk) or (isinstance(fv, str) and len(str(fv)) > len(str(base.get(fk)))):
                            base[fk] = fv
                by_id[id_] = base
            

            if k.endswith("식사"):

                unique_meals = []
                seen_combinations = set()
                for v in by_id.values():
                    meal_key = f"{v.get('끼니', '')}_{v.get('날짜', '')}_{v.get('시간', '')}"
                    if meal_key not in seen_combinations or not meal_key.strip('_'):
                        unique_meals.append(v)
                        seen_combinations.add(meal_key)
                out[k] = unique_meals
            else:

                filtered = []
                for v in by_id.values():
                    missing = self._check_missing_fields(k, v)
                    if not missing:
                        filtered.append(v)
                    else:


                        pass
                out[k] = filtered if filtered else list(by_id.values())
        

        if "user.약" in out:
            out["user.약"] = self._dedup_drug_entities(out["user.약"])
        
        return out


    def _enrich_dose_dict(self, d: dict) -> dict:
        txt = d.get("원문","") or ""

        m = re.search(r"하루\s*(\d+)\s*번", txt)
        if m: d["횟수"] = int(m.group(1))
        kor = {"한":1,"두":2,"세":3,"네":4,"다섯":5}
        m = re.search(r"하루\s*([한두세네다섯])\s*번", txt)
        if m: d["횟수"] = kor[m.group(1)]

        if "식후" in txt: d["식전후"] = "식후"
        elif "식전" in txt: d["식전후"] = "식전"

        if any(k in txt for k in ["아침","점심","저녁"]):
            d.setdefault("시간대", [])
            for k in ["아침","점심","저녁"]:
                if k in txt and k not in d["시간대"]:
                    d["시간대"].append(k)
        return d


    def _filter_meaningful_data(self, value: dict, user_input: str = None) -> dict:
        if not value:
            return {}
        

        if not isinstance(value, dict):
            print(f"[ERROR] _filter_meaningful_data: 예상치 못한 데이터 타입 {type(value)}: {value}")
            return {}
        
        print(f"[DEBUG] _filter_meaningful_data 호출: {value}")
        
        filtered = {}
        for key, val in value.items():
            if val is None or val == "" or val == "N/A" or val == "null":
                continue
            if isinstance(val, str) and val.strip() in ["", "N/A", "null", "없음", "모름", "모르겠어", "N/A", "null"]:
                continue
            if isinstance(val, list) and not val:
                continue
            if isinstance(val, list) and all(item in ["", "N/A", "null", "없음", "모름"] for item in val):
                continue
            filtered[key] = val
        

        if "약명" in filtered and ("복용" in filtered or "식사와의 관계" in filtered):
            return filtered
        

        if any(key in filtered for key in ["시간대", "복용", "날짜"]) and any(key in filtered for key in ["시간대", "복용", "날짜"]):
            return filtered
        

        if not filtered:
            return {}
        
        return filtered

    def _merge_meal_entity(self, existing: dict, new: dict) -> dict:

        if existing is None or not isinstance(existing, dict):
            return new
        merged = existing.copy()
        

        if "메뉴" in new and new["메뉴"]:
            existing_menus = existing.get("메뉴", [])
            new_menus = new["메뉴"] if isinstance(new["메뉴"], list) else [new["메뉴"]]
            

            all_menus = list(set(existing_menus + new_menus))
            merged["메뉴"] = all_menus
        

        for key, val in new.items():
            if key != "메뉴" and val and val not in ["", "N/A", "null"]:
                merged[key] = val
        
        return merged

    def _add_to_vstore(self, entity_key: str, value: dict, metadata: dict, strategy: str = "merge", identity: Optional[str] = None, user_input: str = None, session_id: Optional[str] = None) -> dict:
        try:

            filtered_value = self._filter_meaningful_data(value, user_input)
            if not filtered_value:
                logger.debug(f"의미없는 데이터 필터링: {entity_key} - {value}")
                return value
        except Exception as e:
            if str(e).startswith("QUESTION:"):
                question = str(e)[9:]
                print(f"[DEBUG] _add_to_vstore에서 재질문 예외 처리: {question}")
                return {"질문": question}
            raise e
        

        if isinstance(filtered_value, dict) and "질문" in filtered_value:
            print(f"[DEBUG] _add_to_vstore에서 재질문 반환: {filtered_value['질문']}")

            if session_id:
                self.current_question[session_id] = filtered_value["질문"]
            return filtered_value
        

        if not entity_key.endswith("식사") and not self._is_complete_entity(entity_key, filtered_value):
            logger.debug(f"불완전한 엔티티 저장 거부: {entity_key} - {filtered_value}")
            return filtered_value
        
        base_key = f"{metadata.get('session_id', '')}_{entity_key}"
        if identity is None:

            if entity_key.endswith("사용자"):
                identity = "user_name"
            elif entity_key.endswith("일정"):
                identity = f"{filtered_value.get('제목')}|{filtered_value.get('날짜')}"
            elif entity_key.endswith("물건"):
                identity = filtered_value.get("이름")
            elif entity_key.endswith("식사"):
                identity = f"{filtered_value.get('날짜')}|{filtered_value.get('끼니')}"
            elif entity_key.endswith("약"):
                identity = filtered_value.get("약명")
            elif entity_key.endswith("가족"):
                identity = f"{filtered_value.get('관계')}|{filtered_value.get('이름')}"
            elif entity_key.endswith("기념일"):
                identity = f"{filtered_value.get('관계')}|{filtered_value.get('제목')}|{filtered_value.get('날짜')}"
            elif entity_key.endswith("취미"):
                identity = filtered_value.get("이름")
            elif entity_key.endswith("취향"):
                identity = f"{filtered_value.get('종류')}|{filtered_value.get('값')}"
            elif entity_key.endswith("건강상태"):
                parts = [filtered_value.get("질병"), filtered_value.get("증상"), filtered_value.get("기간")]
                identity = "|".join([p for p in parts if p]) or hashlib.md5(json.dumps(filtered_value, ensure_ascii=False).encode()).hexdigest()

        unique_key = f"{base_key}_{hashlib.md5(str(identity).encode()).hexdigest()}"


        session_id = metadata.get('session_id', 'default')
        if "날짜" in filtered_value and filtered_value["날짜"]:
            filtered_value["날짜"] = self._normalize_date(filtered_value["날짜"], session_id)
        elif entity_key.endswith(("일정", "식사", "기념일", "약", "건강상태", "취미", "취향")):

            filtered_value["날짜"] = self._normalize_date("오늘", session_id)
        

        if entity_key.endswith("물건") and "위치" in filtered_value:
            filtered_value["위치"] = self._normalize_location(filtered_value["위치"])

        try:

            if entity_key.endswith("사용자") and "이름" in filtered_value:
                filtered_value["이름"] = self._normalize_name(filtered_value["이름"])
                

                if hasattr(self, 'excel_cache') and session_id in self.excel_cache:
                    cache_entities = self.excel_cache[session_id].get("사용자", [])
                    for cached_entity in cache_entities:
                        if isinstance(cached_entity, dict) and cached_entity.get("이름") == filtered_value.get("이름"):
                            logger.debug(f"사용자 정보 중복 방지: '{filtered_value.get('이름')}' 이미 존재 (세션: {session_id})")
                            return self._merge_entity_values(cached_entity, filtered_value, "user.사용자")
                

            

            if entity_key.endswith("가족") and "이름" in filtered_value and "관계" in filtered_value:
                filtered_value["이름"] = self._normalize_name(filtered_value["이름"])
                

                if hasattr(self, 'excel_cache') and session_id in self.excel_cache:
                    cache_entities = self.excel_cache[session_id].get("가족", [])
                    for cached_entity in cache_entities:
                        if (isinstance(cached_entity, dict) and 
                            cached_entity.get("이름") == filtered_value.get("이름") and
                            cached_entity.get("관계") == filtered_value.get("관계")):
                            logger.debug(f"가족 정보 중복 방지: '{filtered_value.get('관계')} {filtered_value.get('이름')}' 이미 존재 (세션: {session_id})")
                            return self._merge_entity_values(cached_entity, filtered_value, "user.가족")
                

            

            if "이름" in filtered_value and filtered_value.get("이름"):
                print(f"[DEBUG] 동적 중복 검사 시작: entity_key={entity_key}, session_id={session_id}, 이름={filtered_value.get('이름')}")
                
                entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
                existing_entities = []
                

                if hasattr(self, 'excel_cache'):

                    if entity_key.endswith("물건"):

                        for sess_id, cache_data in self.excel_cache.items():
                            cache_entities = cache_data.get(entity_type, [])
                            for cached_entity in cache_entities:
                                if isinstance(cached_entity, dict) and cached_entity.get("이름") == filtered_value.get("이름"):
                                    existing_entities.append(cached_entity)
                                    print(f"[DEBUG] 기존 엔티티 발견: {cached_entity}")
                    else:

                        if session_id in self.excel_cache:
                            cache_entities = self.excel_cache[session_id].get(entity_type, [])
                            for cached_entity in cache_entities:
                                if isinstance(cached_entity, dict):
                                    existing_entities.append(cached_entity)
                                    print(f"[DEBUG] 기존 엔티티 발견: {cached_entity}")
                

                if existing_entities:
                    existing_name = existing_entities[0].get("이름", "")
                    new_name = filtered_value.get("이름", "")
                    entity_type = entity_key.replace("user.", "")
                    
                    print(f"[DEBUG] 중복 엔티티 발견: 기존='{existing_name}', 새='{new_name}' - 최신 정보로 자동 업데이트")
                    

                    if existing_name == new_name:

                        merged = self._merge_entity_values(existing_entities[0], filtered_value, entity_key)
                        print(f"[DEBUG] 중복 엔티티 자동 병합 완료: {merged}")

                        filtered_value = merged

                        if session_id in self.excel_cache:
                            cache_entities = self.excel_cache[session_id].get(entity_type, [])
                            for i, cached_entity in enumerate(cache_entities):
                                if isinstance(cached_entity, dict) and cached_entity.get("이름") == existing_name:
                                    cache_entities[i] = merged
                                    break

                    else:

                        print(f"[DEBUG] 다른 이름의 엔티티 - 새 엔티티로 저장: {new_name}")

            

            if entity_key.endswith("약") and "약명" in filtered_value:

                if hasattr(self, 'excel_cache') and session_id in self.excel_cache:
                    cache_entities = self.excel_cache[session_id].get("약", [])
                    for cached_entity in cache_entities:
                        if isinstance(cached_entity, dict) and cached_entity.get("약명") == filtered_value.get("약명"):
                            logger.debug(f"약 정보 중복 발견: '{filtered_value.get('약명')}' 이미 존재 - 최신 정보로 자동 병합 (세션: {session_id})")

                            merged = self._merge_entity_values(cached_entity, filtered_value, "user.약")

                            filtered_value = merged

                            for i, entity in enumerate(cache_entities):
                                if isinstance(entity, dict) and entity.get("약명") == filtered_value.get("약명"):
                                    cache_entities[i] = merged
                                    break

                            break
            

            if entity_key.endswith("식사"):

                if hasattr(self, 'excel_cache') and session_id in self.excel_cache:
                    cache_entities = self.excel_cache[session_id].get("식사", [])
                    for cached_entity in cache_entities:
                        if not isinstance(cached_entity, dict):
                            continue
                        


                        if (filtered_value.get("날짜") and filtered_value.get("끼니") and 
                            cached_entity.get("날짜") == filtered_value.get("날짜") and
                            cached_entity.get("끼니") == filtered_value.get("끼니")):
                            logger.debug(f"식사 정보 중복 방지: '{filtered_value.get('날짜')} {filtered_value.get('끼니')}' 이미 존재 (세션: {session_id})")
                            merged_meal = self._merge_meal_entity(cached_entity, filtered_value)

                            cache_entities.remove(cached_entity)
                            cache_entities.append(merged_meal)
                            return merged_meal
                        

                        elif (filtered_value.get("메뉴") and cached_entity.get("메뉴") and
                              not filtered_value.get("끼니") and not cached_entity.get("끼니") and
                              filtered_value.get("메뉴") == cached_entity.get("메뉴")):
                            logger.debug(f"식사 정보 중복 방지: 메뉴 '{filtered_value.get('메뉴')}' 이미 존재 (세션: {session_id})")
                            merged_meal = self._merge_meal_entity(cached_entity, filtered_value)

                            cache_entities.remove(cached_entity)
                            cache_entities.append(merged_meal)
                            return merged_meal
                        

                        elif (filtered_value.get("시간") and not filtered_value.get("메뉴") and 
                              not filtered_value.get("끼니") and cached_entity.get("끼니")):
                            logger.debug(f"식사 시간 업데이트: 기존 식사에 시간 '{filtered_value.get('시간')}' 추가 (세션: {session_id})")
                            merged_meal = self._merge_meal_entity(cached_entity, filtered_value)

                            cache_entities.remove(cached_entity)
                            cache_entities.append(merged_meal)
                            return merged_meal
            

            if strategy == "merge":

                entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
                old_val = None
                
                if hasattr(self, 'excel_cache') and session_id in self.excel_cache:
                    cache_entities = self.excel_cache[session_id].get(entity_type, [])

                    if identity and cache_entities:
                        for cached_entity in cache_entities:
                            if isinstance(cached_entity, dict) and str(identity) in str(cached_entity):
                                old_val = cached_entity
                                break
                    
                    if old_val:

                        if entity_key.endswith("사용자"):

                            if old_val.get("이름") and old_val.get("이름") == value.get("이름"):

                                logger.debug(f"사용자 정보 중복 방지: '{value.get('이름')}' 이미 존재")
                                return self._merge_entity_values(old_val, value, "user.사용자")
                            
                            if old_val.get("이름") and old_val.get("이름") != value.get("이름"):

                                print(f"[WARN] 사용자 이름 충돌: 기존 '{old_val.get('이름')}' vs 새 '{filtered_value.get('이름')}' - 재질문으로 처리")
                                return old_val

                            filtered_value = self._merge_entity_values(old_val, filtered_value, entity_key)
                        

                        elif entity_key.endswith("약"):

                            combined_dosage = (old_val.get("복용") or []) + (filtered_value.get("복용") or [])

                        filtered_dosage, extracted_period = self._extract_duration_from_dosage(combined_dosage)
                        filtered_value["복용"] = filtered_dosage
                        

                        if extracted_period:
                            filtered_value["복용 기간"] = extracted_period
                        elif filtered_value.get("복용 기간"):
                            filtered_value["복용 기간"] = self._normalize_duration(filtered_value["복용 기간"])
                        elif old_val.get("복용 기간"):
                            filtered_value["복용 기간"] = self._normalize_duration(old_val["복용 기간"])
                        

                        filtered_value = self._merge_entity_values(old_val, filtered_value, entity_key)
                    

                    elif entity_key.endswith("일정"):
                        if old_val.get("제목") == filtered_value.get("제목") and old_val.get("날짜") == filtered_value.get("날짜"):

                            filtered_value = self._merge_entity_values(old_val, filtered_value, entity_key)
                    

                    elif entity_key.endswith("기념일"):
                        if (old_val.get("관계") == filtered_value.get("관계") and 
                            old_val.get("제목") == filtered_value.get("제목") and 
                            old_val.get("날짜") == filtered_value.get("날짜")):
                            filtered_value = {**old_val, **filtered_value}
                    

                    elif entity_key.endswith("식사"):
                        if (old_val.get("날짜") == filtered_value.get("날짜") and 
                            old_val.get("끼니") == filtered_value.get("끼니")):

                            old_menus = old_val.get("메뉴", [])
                            new_menus = filtered_value.get("메뉴", [])
                            filtered_value["메뉴"] = list(set(old_menus + new_menus))
                            

                            if filtered_value.get("시간"):

                                filtered_value["시간"] = self._normalize_time_field(filtered_value.get("시간"))
                            elif old_val.get("시간"):

                                filtered_value["시간"] = self._normalize_time_field(old_val.get("시간"))
                            
                            filtered_value = {**old_val, **filtered_value}
                    

                    elif entity_key.endswith("물건"):
                        if old_val.get("이름") == filtered_value.get("이름"):

                            if filtered_value.get("위치") and (
                                not old_val.get("위치") or 
                                len(str(filtered_value.get("위치"))) > len(str(old_val.get("위치")))
                            ):
                                old_val["위치"] = filtered_value.get("위치")
                            filtered_value = {**old_val, **filtered_value}
                    

                    elif entity_key.endswith("건강상태"):
                        if old_val.get("증상") == filtered_value.get("증상"):

                            severity_order = {"경미": 1, "보통": 2, "심함": 3, "매우심함": 4}
                            old_sev = severity_order.get(old_val.get("정도"), 0)
                            new_sev = severity_order.get(filtered_value.get("정도"), 0)
                            if new_sev > old_sev:
                                old_val["정도"] = filtered_value.get("정도")
                            filtered_value = {**old_val, **filtered_value}
        except Exception as e:
            print("[WARN] merge 실패:", e)


        session_id = metadata.get('session_id', 'default')
        try:
            user_name = self.user_names.get(session_id or "default")
            if not user_name or user_name == "사용자":
                print(f"[WARN] 사용자 이름이 설정되지 않아 엔티티를 저장할 수 없습니다. (session_id: {session_id})")
                return filtered_value
            

            entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
            



            self.excel_manager.save_entity_data(user_name, entity_type, filtered_value)
            

            if not hasattr(self, 'excel_cache'):
                self.excel_cache = {}
            session_cache = self.excel_cache.setdefault(session_id, {})
            

            if entity_type not in session_cache:
                session_cache[entity_type] = []
            

            existing = None
            for item in session_cache[entity_type]:
                if isinstance(item, dict):

                    if identity and str(identity) in str(item):
                        existing = item
                        break
            
            if existing and strategy == "merge":

                merged = self._merge_entity_values(existing, filtered_value, entity_key)
                session_cache[entity_type] = [
                    merged if (isinstance(item, dict) and str(identity) in str(item)) else item
                    for item in session_cache[entity_type]
                ]
                filtered_value = merged
            elif not existing:

                session_cache[entity_type].append(filtered_value)
            
            print(f"[INFO] 엑셀 백엔드로 엔티티 저장 완료: ({entity_type}) → {user_name}.xlsx")
        except Exception as e:
            print(f"[ERROR] _add_to_vstore 엑셀 저장 실패: {e}")
            import traceback
            traceback.print_exc()
        
        return filtered_value

    def _get_facts_text(self, session_id: str) -> str:
        if not hasattr(self, "excel_cache"):
            return ""
        sess = self.excel_cache.get(session_id, {})
        facts = []
        if sess.get("사용자"):
            u = sess["사용자"][0]
            facts.append(f"이름은 {u.get('이름')}이고, 나이는 {u.get('나이')}입니다.")
        if sess.get("물건"):
            for it in sess["물건"][-3:]:
                facts.append(f"{it.get('이름')}은 {it.get('위치')}에 있습니다.")
        if sess.get("식사"):
            for meal in sess["식사"][-3:]:
                facts.append(f"{meal.get('날짜')} {meal.get('끼니')}에는 {', '.join(meal.get('메뉴', []))}을 먹었습니다.")
        return "\n".join(facts)


    def _upsert_entities_and_get_confirms(self, session_id: str, entities: Dict[str, List[Dict[str, Any]]], user_input: str = None) -> Tuple[List[str], bool]:
        questions: List[str] = []
        has_schedule = False
        

        if not isinstance(entities, dict):
            print(f"[ERROR] _upsert_entities_and_get_confirms: entities가 dict가 아님 {type(entities)}: {entities}")
            return [], False
        

        for entity_key, entity_list in entities.items():
            for entity in entity_list:
                if isinstance(entity, dict) and "질문" in entity:
                    print(f"[DEBUG] 재질문 처리 (상위): {entity['질문']}")
                    questions.append(entity["질문"])
                    return questions, has_schedule
        

        if session_id in self.current_question:
            question = self.current_question[session_id]
            print(f"[DEBUG] _upsert_entities_and_get_confirms에서 전역 재질문 발견: {question}")
            questions.append(question)
            return questions, has_schedule


        correction_keywords = ["이미 말했", "이미 말했는데", "이미 답했", "이미 답했는데", "아까 말했", "아까 말했는데"]
        is_correction = any(keyword in user_input for keyword in correction_keywords)
        

        skip_keywords = ["모르겠", "없어", "몰라", "없다", "모름", "기억 안나", "기억안나", "잘 모르", "잘모르"]
        is_skip_response = any(keyword in user_input for keyword in skip_keywords)
        
        if is_correction:
            logger.debug("정정 요청 감지: 사용자가 이미 답변했다고 명시함")

            return [], has_schedule
        
        if is_skip_response:
            logger.debug("모름/없음 응답 감지: 사용자가 모르거나 없다고 답함")

            for entity_key, values in entities.items():
                for value in values:
                    try:

                        filtered_value = self._filter_meaningful_data(value, user_input)
                        if not filtered_value:
                            continue
                        

                        if isinstance(filtered_value, dict) and "질문" in filtered_value:
                            print(f"[DEBUG] 재질문 처리: {filtered_value['질문']}")
                            questions.append(filtered_value["질문"])
                            return questions, has_schedule
                    except Exception as e:
                        if str(e).startswith("QUESTION:"):
                            question = str(e)[9:]
                            print(f"[DEBUG] _upsert_entities_and_get_confirms에서 재질문 예외 처리: {question}")
                            questions.append(question)
                            return questions, has_schedule
                        raise e
                    
                    missing_fields = self._check_missing_fields(entity_key, filtered_value)
                    if missing_fields:

                        for field in missing_fields:
                            if field == "시간":
                                filtered_value[field] = "미정"
                            elif field == "날짜":
                                filtered_value[field] = "오늘"
                            elif field == "약명":
                                filtered_value[field] = "미정"
                            elif field == "제목":
                                filtered_value[field] = "미정"
                            else:
                                filtered_value[field] = "미정"
                        

                        self._add_to_vstore(
                            entity_key, filtered_value,
                            {"session_id": session_id, "entity_key": entity_key, "type": "entity", "created_at": datetime.now().isoformat()},
                            session_id=session_id,
                            strategy="merge"
                        )
                        
                        if entity_key.endswith("일정"):
                            has_schedule = True
            
            return [], has_schedule

        self._prevent_name_family_conflict(entities)

        for entity_key, values in entities.items():

            

            if entity_key == "user.사용자":

                existing_users = self._get_existing_user_entities(session_id)
                

                merged_users = existing_users
                for value in values:
                    filtered_value = self._filter_meaningful_data(value)
                    if not filtered_value:
                        continue
                    
                    if not self._is_valid_entity(entity_key, filtered_value):
                        continue
                    

                    merged_users = self._merge_user_entities(merged_users, filtered_value)
                

                for user_entity in merged_users:

                    filtered_value = self._filter_meaningful_data(user_entity)
                    if not filtered_value:
                        continue
                    

                    if not self._is_valid_entity(entity_key, filtered_value):
                        continue
                    

                    self._process_single_entity(entity_key, filtered_value, session_id, questions, has_schedule)
            else:

                for value in values:

                    filtered_value = self._filter_meaningful_data(value)
                    if not filtered_value:
                        print(f"[INFO] 의미없는 데이터 필터링: {entity_key} - {value}")
                        continue
                    

                    if not self._is_valid_entity(entity_key, filtered_value):
                        print(f"[INFO] 유효하지 않은 엔티티 스킵: {entity_key} - {filtered_value}")
                        continue
                        

                    self._process_single_entity(entity_key, filtered_value, session_id, questions, has_schedule)
                

                if entity_key.endswith("일정"):
                    has_schedule = True
                

                

                if entity_key.endswith("식사") and "메뉴" in filtered_value:
                    menus = filtered_value["메뉴"]
                    if isinstance(menus, list):

                        filtered_menus = [menu for menu in menus if not menu.endswith("약")]
                        if not filtered_menus:

                            print(f"[INFO] 약을 식사로 착각한 엔티티 제거: {entity_key} - {filtered_value}")
                            continue
                        filtered_value["메뉴"] = filtered_menus
                

                missing_fields = self._check_missing_fields(entity_key, filtered_value)

                if missing_fields:

                    logger.debug(f"누락된 필드 감지: {entity_key} - {missing_fields}, 값: {filtered_value}")
                    followup_questions = self._generate_followup_questions(entity_key, missing_fields, filtered_value)
                    questions.extend(followup_questions)
                    

                    if entity_key.endswith("식사") and (
                        ("메뉴" in missing_fields and filtered_value.get("메뉴") == []) or
                        ("시간" in missing_fields and not filtered_value.get("시간"))
                    ):

                        final_value = self._add_to_vstore(
                            entity_key, filtered_value,
                            {"session_id": session_id, "entity_key": entity_key, "type": "entity", "created_at": datetime.now().isoformat()},
                            strategy="merge"
                        )
                        logger.debug(f"식사 엔티티 임시 저장 (메뉴/시간 누락): {filtered_value}")
                        

                        if "시간" in missing_fields and not filtered_value.get("시간"):
                            self.pending_question[session_id] = {
                                "기존_엔티티": final_value,
                                "새_엔티티": final_value,
                                "entity_key": entity_key
                            }
                            print(f"[DEBUG] 재질문 상태 설정: {entity_key} - 시간 누락")
                    else:
                        continue


                final_value = self._add_to_vstore(
                    entity_key, filtered_value,
                    {"session_id": session_id, "entity_key": entity_key, "type": "entity", "created_at": datetime.now().isoformat()},
                    strategy="merge"
                )


                if entity_key.endswith(".약"):
                    if final_value.get("복용"):
                        enriched = [self._enrich_dose_dict(d) for d in final_value["복용"]]
                        final_value["복용"] = enriched
                        

                        try:
                            self._add_to_vstore(
                                entity_key=entity_key,
                                value=final_value,
                                session_id=session_id,
                                metadata={"session_id": session_id, "type": "entity"},
                                strategy="merge",
                                identity=final_value.get("약명")
                            )

                        except Exception as e:
                            print(f"[WARN] 복용 정보 enrich 후 업데이트 실패: {e}")
                        


                        has_complete_info = True
                        





                    if not final_value.get("복용") and not final_value.get("식사와의 관계"):
                        questions.append(f"{final_value.get('약명','약')}은 언제, 하루 몇 번 복용하나요?")



        return list(dict.fromkeys(questions)), has_schedule


    def save_final_summary(self, session_id: str):
        print(f"[DEBUG] save_final_summary 시작: session_id={session_id}")
        print(f"[DEBUG] auto_export_enabled: {self.cfg.auto_export_enabled}")
        

        try:
            import sqlite3
            conn = sqlite3.connect(self.sqlite_path)
            c = conn.cursor()
            c.execute("SELECT id, session_id, role, content, message FROM message_store WHERE session_id = ? ORDER BY id", (session_id,))
            messages = c.fetchall()
            conn.close()
            
            print(f"[DEBUG] message_store에서 조회된 메시지 수: {len(messages)}")
            
            if len(messages) == 0:
                print(f"[INFO] 세션 {session_id}에 대화 기록이 없습니다.")
                return
                

            texts = []
            for msg in messages:
                role = msg[2]
                content = msg[3]
                message = msg[4]
                

                if content:
                    texts.append(content)
                elif message:
                    try:
                        import json
                        msg_data = json.loads(message)
                        if 'data' in msg_data and 'content' in msg_data['data']:
                            texts.append(msg_data['data']['content'])
                    except (json.JSONDecodeError, KeyError):
                        pass
            
            print(f"[DEBUG] texts 길이: {len(texts)}")
            
        except Exception as e:
            print(f"[ERROR] message_store 조회 실패: {e}")
            return
        

        if self.cfg.auto_export_enabled:
            print(f"[DEBUG] 자동 추출 시작: session_id={session_id}")
            try:
                self.export_conversation_to_excel(session_id)
                print(f"[INFO] 대화 기록이 엑셀 파일로 저장되었습니다: conversation_extract/{session_id}.xlsx")
            except Exception as e:
                print(f"[ERROR] 엑셀 파일 생성 실패: {e}")
        

        confirmed_name = self._get_confirmed_user_name(session_id)
        

        emotional_context = ""
        if session_id in self.emotional_state:
            emotional_info = self.emotional_state[session_id]
            if emotional_info.get("mood") and emotional_info.get("intensity", 0) > 0.5:
                mood = emotional_info["mood"]
                intensity = emotional_info["intensity"]
                if mood == "negative":
                    emotional_context = f" 사용자는 피곤함, 어지러움, 우울함 등의 부정적 감정을 표현했습니다."
                elif mood == "positive":
                    emotional_context = f" 사용자는 기쁨, 만족감 등의 긍정적 감정을 표현했습니다."
        

        system_prompt = (
            "다음 대화를 정확히 요약하세요.\n\n"
            "예시:\n"
            "사용자: '내 이름은 김철수야'\n"
            "AI: '김철수님의 이름을 저장했어요'\n"
            "요약: 사용자가 자신의 이름을 '김철수'라고 소개했습니다.\n\n"
            "규칙:\n"
            "- 대화에 있는 내용만 기록하세요\n"
            "- 이름, 약명, 음식명은 정확히 그대로 기록하세요\n"
            "- '오늘'은 현재 날짜로 변환하세요\n"
        )
        
        if confirmed_name:
            system_prompt += f"5) 사용자의 이름은 '{confirmed_name}'입니다. 다른 이름으로 추측하지 마세요.\n"
        
        system_prompt += f"세션 ID: {session_id}"
        

        escaped_texts = []
        for text in texts:

            escaped_text = text.replace("{", "{{").replace("}", "}}")
            escaped_texts.append(escaped_text)
        
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"세션 {session_id}의 대화 내용{emotional_context}:\n" + "\n".join(escaped_texts)),
        ])
        chain = summary_prompt | self.llm | StrOutputParser()
        summary_text = chain.invoke({})
        
        conn = sqlite3.connect(self.sqlite_path)
        c = conn.cursor()
        

        c.execute("INSERT INTO conversation_summary (session_id, summary, created_at, updated_at) VALUES (?, ?, ?, ?)", 
                 (session_id, summary_text, datetime.now().isoformat(), datetime.now().isoformat()))

        
        conn.commit()
        conn.close()
        

        print(f"[DEBUG] 자동 추출 시작: session_id={session_id}, auto_export_enabled={self.cfg.auto_export_enabled}")
        self.auto_export_conversation(session_id)


    def build_chain(self) -> Runnable:
        system_tmpl = (
            "당신은 생활 지원 로봇입니다.\n"
            "최근 대화 요약: {summary}\n"
            "저장된 엔티티: {entities}\n"
            "이번 턴 엔티티: {staged_entities}\n"
            "검색 컨텍스트:\n{retrieved}\n\n"
            "규칙:\n"
            "- 저장된 정보를 우선 활용.\n"
            "- 모르는 정보가 있으면 '아직 그 정보는 몰라요. 알려주시면 기억해둘게요!'라고 자연스럽게 답변.\n"
            "- 답변은 간결한 한국어."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_tmpl),
            MessagesPlaceholder("history"),
            ("human", "{input}"),
        ])
        return (
            {
                "summary": lambda x: x.get("summary", "(요약 없음)"),
                "entities": lambda x: x.get("entities", "[]"),
                "staged_entities": lambda x: x.get("staged_entities", "[]"),
                "retrieved": lambda x: x.get("retrieved", ""),
                "history": lambda x: x.get("history", []),
                "input": lambda x: x["input"],
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def _get_all_medications(self, session_id: str) -> List[dict]:
        try:
            user_name = self.user_names.get(session_id or "default")
            if not user_name:
                return []
            df = self.excel_manager.load_sheet_data(user_name, "복약정보")
            if df is None or df.empty:
                return []
            return df.to_dict("records")
        except Exception as e:
            print(f"[ERROR] 약 정보 조회 실패: {e}")
            return []


    def _format_entities_for_output(self, user_input: str, ents: List[Document], session_id: str = "default") -> str:
        if not ents:

            user_name = self._get_confirmed_user_name(session_id)
            if user_name and user_name != "사용자":
                return f"아직 그 정보는 몰라요, {user_name}님. 알려주시면 기억해둘게요!"
            else:
                return "아직 그 정보는 몰라요. 알려주시면 기억해둘게요!"
        lines = []
        for d in ents:
            try:
                val = json.loads(d.page_content)

            except Exception as e:
                print(f"[DEBUG] JSON 파싱 실패: {e}")
                continue
            etype = d.metadata.get("entity_key", "")

            

            if etype.endswith("사용자") and val.get("이름"):

                lines.append(f"네, {val['이름']}님이에요.")
            

            elif etype.endswith("물건"):
                if val.get("위치"):

                    lines.append(f"{val.get('이름')}은 {val.get('위치')}에 있어요.")
                else:
                    lines.append(f"{val.get('이름')}에 대해 알고 있어요.")
            

            elif etype.endswith("일정"):
                title = val.get("제목", "일정")
                date = val.get("날짜", "")
                time = val.get("시간", "")
                location = val.get("장소", "")
                

                if date:
                               date = self._normalize_date(date, session_id)
                
                parts = [title]
                if date:
                    parts.append(f"{date}에")
                if time:

                    if isinstance(time, list):
                        time = ', '.join(time)
                    parts.append(f"{time}에")
                if location:
                    parts.append(f"{location}에서")
                
                lines.append(" ".join(parts) + " 예정이에요.")
            

            elif etype.endswith("약"):
                drug_name = val.get("약명", "약")
                doses = val.get("복용", [])
                period = val.get("복용 기간", "")
                
                if doses:
                    dose_info = []
                    for dose in doses:
                        if dose.get("원문"):
                            dose_info.append(dose["원문"])
                    if dose_info:
                        lines.append(f"{drug_name}을 {', '.join(dose_info)} 복용하시는군요.")
                else:
                    lines.append(f"{drug_name}에 대해 알고 있어요.")
                
                if period:
                    lines.append(f"복용 기간은 {period}이에요.")
                else:
                    lines.append("복용 기간은 아직 안 알려주셨어요.")
            

            elif etype.endswith("식사"):
                meal = val.get("끼니", "")
                menus = val.get("메뉴", [])
                date = val.get("날짜", "")
                time = val.get("시간", "")
                

                if date:
                               date = self._normalize_date(date, session_id)
                
                parts = []
                if date:
                    parts.append(f"{date}")
                if meal:
                    parts.append(f"{meal}에")
                if time:

                    if isinstance(time, list):
                        time = ', '.join(time)
                    parts.append(f"{time}에")
                if menus:
                    parts.append(f"{', '.join(menus)}을 드셨어요.")
                
                if parts:
                    lines.append(" ".join(parts))
                else:
                    lines.append("식사 정보를 알고 있어요.")
            

            elif etype.endswith("가족"):
                relation = val.get("관계", "")
                name = val.get("이름", "")
                if relation and name:
                    lines.append(f"{relation} {name}님에 대해 알고 있어요.")
                elif relation:
                    lines.append(f"{relation}에 대해 알고 있어요.")
            


            elif not etype.endswith(("사용자", "물건", "일정", "약", "식사", "가족", "기념일", "취미", "취향", "건강상태")):
                name = val.get("이름", "")
                relation_type = etype.replace("user.", "")
                if name:
                    lines.append(f"{relation_type} {name}님에 대해 알고 있어요.")
                else:
                    lines.append(f"{relation_type}에 대해 알고 있어요.")
            

            elif etype.endswith("기념일"):
                title = val.get("제목", "")
                date = val.get("날짜", "")
                relation = val.get("관계", "")
                

                if date:
                               date = self._normalize_date(date, session_id)
                
                parts = []
                if relation:
                    parts.append(f"{relation}의")
                if title:
                    parts.append(f"{title}")
                if date:
                    parts.append(f"{date}에")
                
                if parts:
                    lines.append(" ".join(parts) + " 기념일이에요.")
            

            elif etype.endswith("건강상태"):
                symptom = val.get("증상", "")
                severity = val.get("정도", "")
                period = val.get("기간", "")
                
                parts = []
                if symptom:
                    parts.append(f"{symptom} 증상")
                if severity:
                    parts.append(f"{severity}한 정도")
                if period:
                    parts.append(f"{period} 동안")
                
                if parts:
                    lines.append(" ".join(parts) + "이 있으시군요.")
            

            elif etype.endswith("취미"):
                hobby = val.get("이름", "")
                if hobby:
                    lines.append(f"{hobby} 취미를 가지고 계시는군요.")
            

            elif etype.endswith("취향"):
                category = val.get("종류", "")
                value = val.get("값", "")
                if category and value:
                    lines.append(f"{category}에서 {value}을 좋아하시는군요.")
                elif value:
                    lines.append(f"{value}을 좋아하시는군요.")
        

        result = " ".join(lines) if lines else "아직 그건 몰라요."

        return result


    def process_user_input(self, user_text: str, session_id: str = "default") -> str:
        print(f"[DEBUG] process_user_input 호출됨: '{user_text}'")
        try:

            if not hasattr(self, '_session_initialized'):
                self._session_initialized = set()
            
            if session_id not in self._session_initialized:
                print(f"[DEBUG] 세션 {session_id} 초기화 완료")
                self._session_initialized.add(session_id)
            

            if session_id in self.pending_question:
                print(f"[DEBUG] pending_question 발견: {self.pending_question[session_id]}")

                import re
                yes_pattern = re.compile(r"^(응|네|좋아|그래|ㅇㅇ|웅|맞아)\s*$", re.IGNORECASE)
                no_pattern = re.compile(r"^(아니|괜찮아|됐어|ㄴㄴ|싫어)\s*$", re.IGNORECASE)
                
                if yes_pattern.match(user_text.strip()) or no_pattern.match(user_text.strip()):
                    followup = handle_pending_answer(user_text, self, session_id)
                    if followup:
                        return followup
                else:
                    print(f"[DEBUG] pending_question이 있지만 확인 응답이 아님: '{user_text}' - 일반 처리로 진행")


            print(f"[DEBUG] 분류 시작")
            

            cached_result = self._get_cached_classification(user_text)
            if cached_result:
                print(f"[DEBUG] 유사 캐시 사용: {cached_result['category']} (신뢰도: {cached_result['confidence']:.2f})")
                from .task_classifier import ClassificationResult
                result = ClassificationResult(
                    category=cached_result["category"],
                    confidence=cached_result["confidence"],
                    probabilities=cached_result.get("probabilities", {}),
                    reasoning=cached_result["reasoning"]
                )
            else:

                from .task_classifier import classify_hybrid
                result = classify_hybrid(user_text, None)
                print(f"[DEBUG] LLM 분류 결과: '{user_text}' -> {result.category} (신뢰도: {result.confidence:.2f})")
                

                self._add_to_cache(user_text, {
                    "category": result.category,
                    "confidence": result.confidence,
                    "probabilities": result.probabilities,
                    "reasoning": result.reasoning
                })
        except Exception as e:
            print(f"[ERROR] process_user_input 초기 처리 실패: {e}")
            import traceback
            traceback.print_exc()
            return "죄송해요, 처리 중 오류가 발생했어요."


        print(f"[DEBUG] 카테고리별 처리 시작: {result.category}")
        

        self.conversation_memory.chat_memory.add_user_message(user_text)
        
        if result.category == "cognitive":
            print(f"[DEBUG] cognitive 처리 호출")
            try:
                from .support_chains import handle_cognitive_task_with_lcel
                response = handle_cognitive_task_with_lcel(user_text, self, session_id)

                self.conversation_memory.chat_memory.add_ai_message(response)

                return response
            except Exception as e:
                import traceback
                print(f"[ERROR] cognitive 처리 실패: {traceback.format_exc()}")

                error_response = "죄송해요, 처리 중 오류가 있었어요. 다시 한 번 말씀해 주시겠어요?"

                self.conversation_memory.chat_memory.add_ai_message(error_response)
                return error_response
        elif result.category == "emotional":
            print(f"[DEBUG] emotional 처리 호출")
            try:
                response = self._handle_emotional_task(user_text, session_id)

                self.conversation_memory.chat_memory.add_ai_message(response)


                return response
            except Exception as e:
                import traceback
                print(f"[ERROR] emotional 처리 실패: {traceback.format_exc()}")

                error_response = "지금 많이 힘드셨죠. 곁에서 같이 이야기 들어드릴게요. 어떤 점이 가장 힘들었나요?"

                self.conversation_memory.chat_memory.add_ai_message(error_response)
                return error_response
        elif result.category == "physical":
            print(f"[DEBUG] physical 처리 호출")
            
            response = handle_physical_task(user_text, self, session_id)
            

            if isinstance(response, dict):
                message = response.get("message", str(response))

                self.conversation_memory.chat_memory.add_ai_message(message)


                return response
            else:

                self.conversation_memory.chat_memory.add_ai_message(response)


                return response
        elif result.category == "query":
            print(f"[DEBUG] query 처리 호출")
            from .support_chains import handle_query_with_lcel
            response = handle_query_with_lcel(user_text, self, session_id)

            self.conversation_memory.chat_memory.add_ai_message(response)


            return response
        else:
            print(f"[DEBUG] 알 수 없는 카테고리: {result.category}")
            return "죄송해요, 잘 이해하지 못했어요."

    
    def _handle_emotional_task(self, user_text: str, session_id: str) -> str:
        try:

            if hasattr(self, 'pending_question') and self.pending_question.get(session_id):
                pending_data = self.pending_question[session_id]
                print(f"[DEBUG] 중복 응답 처리 (emotional): {user_text}")
                result = self.handle_duplicate_answer(user_text, pending_data)
                

                if session_id in self.pending_question:
                    del self.pending_question[session_id]
                
                return result["message"]
            

            memory_vars = self.conversation_memory.load_memory_variables({})
            conversation_history = memory_vars.get('history', '')
            print(f"[DEBUG] Emotional LCEL history 길이: {len(conversation_history)}")
            

            self._save_message(session_id, "human", user_text)
            

            conversation_history = self._convert_conversation_history_to_string(conversation_history)
            

            entities = self._pre_extract_entities(user_text, session_id)
            print(f"[DEBUG] emotional에서 추출된 엔티티: {entities}")
            

            if isinstance(entities, dict) and entities.get("success") == False and entities.get("incomplete"):
                print(f"[DEBUG] Slot-filling 필요 (emotional): {entities['message']}")

                self.pending_question[session_id] = entities.get("pending_data", {})
                return entities["message"]
            
            if entities:

                if "user.사용자" in entities:
                    for user_entity in entities["user.사용자"]:
                        name = user_entity.get("이름", "")
                        if name:
                            save_result = self.save_entity_to_vectorstore(
                                entity_type="사용자",
                                data={"이름": name, "확인됨": user_entity.get("확인됨", True)},
                                session_id=session_id
                            )
                            if save_result.get("duplicate"):

                                self.pending_question[session_id] = save_result.get("pending_data", {})
                                return save_result["message"]
                

                if "user.건강상태" in entities:
                    for emotion in entities["user.건강상태"]:
                        emotion_state = emotion.get("증상", "")
                        if emotion_state:


                            from life_assist_dm.life_assist_dm.support_chains import _summarize_emotion_context_for_save
                            info_summary = _summarize_emotion_context_for_save(user_text, self.llm if hasattr(self, 'llm') else None)
                            
                            save_result = self.save_entity_to_vectorstore(
                                entity_type="정서",
                                data={
                                "감정": emotion_state,
                                "정보": info_summary
                            },
                            session_id=session_id
                        )
                            if save_result.get("duplicate"):

                                self.pending_question[session_id] = save_result.get("pending_data", {})
                                return save_result["message"]
                            else:
                                print(f"[DEBUG] 정서 저장됨: {emotion_state}")
            

            if conversation_history and len(conversation_history.strip()) > 0:

                from .support_chains import build_emotional_reply
                user_name_confirmed = bool(self._get_confirmed_user_name(session_id))
                response = build_emotional_reply(user_text, llm=self.llm, user_name_confirmed=user_name_confirmed)
            else:

                from .support_chains import build_emotional_reply
                user_name_confirmed = bool(self._get_confirmed_user_name(session_id))
                response = build_emotional_reply(user_text, llm=self.llm, user_name_confirmed=user_name_confirmed)
            

            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            

            self._save_message(session_id, "ai", response_text)
            return response_text
            
        except Exception as e:
            print(f"[ERROR] 정서적 작업 처리 실패: {e}")
            return "죄송해요, 처리 중 오류가 발생했어요."
    
    def _extract_appointment_info(self, user_text: str) -> str:
        try:
            import re
            from datetime import datetime, timedelta
            
            info = {"date": None, "time": None, "place": None}


            place_match = re.search(r"(치과|병원|미용실|약속|회의|미팅|약국|은행|카페|식당)", user_text)
            if place_match:
                info["place"] = place_match.group(1)


            if "내일" in user_text:
                info["date"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            elif "모레" in user_text:
                info["date"] = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
            else:

                dow_match = re.search(r"(다음\s*주\s*)(월|화|수|목|금|토|일)요일?", user_text)
                if dow_match:
                    weekday_map = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
                    target_weekday = weekday_map[dow_match.group(2)]
                    today = datetime.now()
                    days_ahead = (target_weekday - today.weekday() + 7) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    days_ahead += 7
                    target_date = today + timedelta(days=days_ahead)
                    info["date"] = target_date.strftime("%Y-%m-%d")


            time_match = re.search(r"(오전|오후)?\s?(\d{1,2})시", user_text)
            if time_match:
                hour = int(time_match.group(2))
                if time_match.group(1) == "오후" and hour < 12:
                    hour += 12
                info["time"] = f"{hour:02d}:00"


            if any(info.values()):
                parts = []
                if info["date"]:
                    parts.append(f"날짜: {info['date']}")
                if info["time"]:
                    parts.append(f"시간: {info['time']}")
                if info["place"]:
                    parts.append(f"장소: {info['place']}")
                return " | ".join(parts)
            else:
                return "예약"
                
        except Exception as e:
            print(f"[ERROR] 예약 정보 추출 실패: {e}")
            return "예약"

    def _get_entities_by_type(self, session_id: str, entity_types: list) -> list:
        if not hasattr(self, "excel_cache"):
            return []
        sess = self.excel_cache.get(session_id, {})
        entities = []
        for entity_type in entity_types:

            simple_type = entity_type.replace("user.", "").replace("_", "")
            if simple_type in sess:
                for item in sess[simple_type]:
                    entities.append({
                        "entity_key": entity_type,
                        "content": str(item),
                        "이름": item.get("이름", ""),
                        "metadata": {}
                    })
        return entities


    def _get_recent_messages(self, session_id: str, limit: int = 10) -> list:
        try:
            import sqlite3
            conn = sqlite3.connect(self.sqlite_path)
            c = conn.cursor()
            c.execute("""
                SELECT role, message FROM message_store 
                WHERE session_id = ? 
                ORDER BY id DESC 
                LIMIT ?
            """, (session_id, limit))
            messages = c.fetchall()
            conn.close()
            return messages
        except Exception as e:
            print(f"[ERROR] 최근 메시지 조회 실패: {e}")
            return []

    def _save_message(self, session_id: str, role: str, content: str):
        try:
            import sqlite3
            import json
            conn = sqlite3.connect(self.sqlite_path)
            c = conn.cursor()
            

            if not isinstance(content, str):
                if isinstance(content, (list, tuple)):
                    content = str(content)
                elif hasattr(content, '__str__'):
                    content = str(content)
                else:
                    content = repr(content)
            

            message_data = {
                "type": "human" if role == "사용자" else "ai",
                "data": {
                    "content": content,
                    "additional_kwargs": {}
                }
            }
            message_json = json.dumps(message_data, ensure_ascii=False)
            
            c.execute(
                "INSERT INTO message_store (session_id, role, message) VALUES (?, ?, ?)",
                (session_id, role, message_json)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[ERROR] 메시지 저장 실패: {e}")

    def generate(self, session_id: str, user_input: str) -> str:

        response = self.process_user_input(user_input, session_id)
        

        
        return response
    
    def _update_existing_entity(self, session_id: str, entity_key: str, existing_entity: dict, new_entity: dict):
        if False and self.vectorstore:

            all_docs = self.vectorstore.get()
            for i, doc_id in enumerate(all_docs.get("ids", [])):
                if doc_id.startswith(f"{session_id}_{entity_key}"):
                    try:
                        doc_data = json.loads(all_docs["documents"][i])
                        if (doc_data.get("entity_key") == entity_key and 
                            doc_data.get("session_id") == session_id and
                            doc_data.get("이름") == existing_entity.get("이름")):
                            self.vectorstore.delete(ids=[doc_id])
                            break
                    except (json.JSONDecodeError, TypeError):
                        continue
        

        entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
        user_name = self.user_names.get(session_id or "default")
        if user_name:
            self.excel_manager.save_entity_data(user_name, entity_type, new_entity)

            if hasattr(self, "excel_cache"):
                sess = self.excel_cache.setdefault(session_id, {})
                entities = sess.setdefault(entity_type, [])

                for i, ent in enumerate(entities):
                    if ent.get("이름") == existing_entity.get("이름"):
                        entities[i] = new_entity
                        break
                else:
                    entities.append(new_entity)
    
    def _add_new_entity(self, session_id: str, entity_key: str, new_entity: dict):
        self._store_entity_direct(session_id, entity_key, new_entity)
    
    def _cancel_schedule(self, session_id: str, title: str):
        try:
            user_name = self.user_names.get(session_id or "default")
            if not user_name:
                return False
            df = self.excel_manager.load_sheet_data(user_name, "일정")
            if df is None or df.empty:
                return False

            filtered_df = df[df["제목"] != title]
            if len(filtered_df) < len(df):

                excel_path = self.excel_manager.get_user_excel_path(user_name)
                excel_file = self.excel_manager.load_user_excel(user_name)
                excel_data = {}
                if excel_file:
                    for sheet in excel_file.sheet_names:
                        if sheet == "일정":
                            excel_data[sheet] = filtered_df
                        else:
                            excel_data[sheet] = pd.read_excel(excel_file, sheet_name=sheet)
                else:
                    excel_data["일정"] = filtered_df
                with pd.ExcelWriter(excel_path, engine='openpyxl', mode='w') as writer:
                    for sheet_name, df_data in excel_data.items():
                        df_data.to_excel(writer, sheet_name=sheet_name, index=False)

                if hasattr(self, "excel_cache"):
                    sess = self.excel_cache.setdefault(session_id, {})
                    if "일정" in sess:
                        sess["일정"] = [row for row in sess["일정"] if row.get("제목") != title]
                print(f"[DEBUG] 일정 취소 완료: {title}")
                return True
            else:
                print(f"[DEBUG] 취소할 일정을 찾을 수 없음: {title}")
                return False
        except Exception as e:
            print(f"[ERROR] 일정 취소 실패: {e}")
            return False
    
    def _store_entity_direct(self, session_id: str, entity_key: str, entity: dict):
        try:
            entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
            user_name = self.user_names.get(session_id or "default")
            if not user_name:
                return False
            self.excel_manager.save_entity_data(user_name, entity_type, entity)

            if hasattr(self, "excel_cache"):
                sess = self.excel_cache.setdefault(session_id, {})
                entities = sess.setdefault(entity_type, [])
                entities.append(entity)
            print(f"[DEBUG] 엔티티 저장 완료: {entity_key} - {entity.get('이름', 'N/A')}")
            return True
        except Exception as e:
            print(f"[ERROR] 엔티티 저장 실패: {e}")
            return False
    
    def _update_entity_in_vstore(self, session_id: str, entity_key: str, updated_entity: dict):
        try:
            entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
            user_name = self.user_names.get(session_id or "default")
            if not user_name:
                return False

            sheet_mapping = {
                "물건": "물건위치",
                "약": "복약정보",
                "일정": "일정",
                "음식": "음식기록",
                "정서": "감정기록",
                "가족": "가족관계",
            }
            sheet_name = sheet_mapping.get(entity_type, "사용자정보KV")

            df = self.excel_manager.load_sheet_data(user_name, sheet_name)
            if df is not None and not df.empty:

                name = updated_entity.get("이름", "")
                if name and "이름" in df.columns:
                    idx = df[df["이름"] == name].index
                    if len(idx) > 0:

                        for col in updated_entity.keys():
                            if col in df.columns:
                                df.loc[idx[0], col] = updated_entity[col]

                        excel_path = self.excel_manager.get_user_excel_path(user_name)
                        excel_file = self.excel_manager.load_user_excel(user_name)
                        excel_data = {}
                        if excel_file:
                            for sheet in excel_file.sheet_names:
                                if sheet == sheet_name:
                                    excel_data[sheet] = df
                                else:
                                    excel_data[sheet] = pd.read_excel(excel_file, sheet_name=sheet)
                        else:
                            excel_data[sheet_name] = df
                        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                            for sname, sdata in excel_data.items():
                                sdata.to_excel(writer, sheet_name=sname, index=False)

                        if hasattr(self, "excel_cache"):
                            sess = self.excel_cache.setdefault(session_id, {})
                            entities = sess.setdefault(entity_type, [])
                            for i, ent in enumerate(entities):
                                if ent.get("이름") == name:
                                    entities[i] = updated_entity
                                    return True

            self.excel_manager.save_entity_data(user_name, entity_type, updated_entity)

            if hasattr(self, "excel_cache"):
                sess = self.excel_cache.setdefault(session_id, {})
                entities = sess.setdefault(entity_type, [])
                entities.append(updated_entity)
            return True
        except Exception as e:
            print(f"[ERROR] 엔티티 업데이트 실패: {e}")
            return False
    
    def auto_export_conversation(self, session_id: str):
        if not self.cfg.auto_export_enabled:
            return None
        
        try:

            excel_path = self.export_conversation_to_excel(session_id)
            if excel_path:
                print(f"✅ 대화 기록이 자동 추출되었습니다: {excel_path}")
                return excel_path
            else:
                print("❌ 대화 기록 추출 실패")
                return None
        except Exception as e:
            print(f"[ERROR] 자동 추출 실패: {e}")
            return None
    
    def export_conversation_to_excel(self, session_id: str):
        try:
            print(f"[DEBUG] export_conversation_to_excel 시작: {session_id}")
            

            import sqlite3
            import pandas as pd
            from datetime import datetime
            import os
            import json
            
            conn = sqlite3.connect(self.sqlite_path)
            cur = conn.cursor()
            

            try:
                cur.execute(
                    "SELECT id, session_id, role, content, created_at FROM message_store WHERE session_id = ? ORDER BY id",
                    (session_id,)
                )
                rows = cur.fetchall()
            except sqlite3.OperationalError as e:
                if "no such column: created_at" in str(e):

                    cur.execute(
                        "SELECT id, session_id, role, content, id FROM message_store WHERE session_id = ? ORDER BY id",
                        (session_id,)
                    )
                    rows = cur.fetchall()
                else:
                    raise e
            conn.close()
            
            if not rows:
                print("대화 기록이 없습니다.")
                return None
            

            data = []
            print(f"[DEBUG] 메시지 수: {len(rows)}")
            for row in rows:
                msg_id, session_id, role, content, created_at = row
                content = content or ""
                print(f"[DEBUG] 메시지 {msg_id}: role={role}, content={content[:50]}...")
                

                try:
                    if content.startswith('{"type":'):
                        msg_data = json.loads(content)
                        actual_type = msg_data.get("type", "unknown")
                        actual_content = msg_data.get("data", {}).get("content", content)
                        

                        if actual_type == "human":
                            display_role = "사용자"
                        elif actual_type == "ai":
                            display_role = "AI"
                        else:
                            display_role = "unknown"
                    else:
                        actual_content = content

                        if role == "human":
                            display_role = "사용자"
                        elif role == "ai":
                            display_role = "AI"
                        else:
                            display_role = "unknown"
                except Exception:
                    actual_content = content
                    if role == "human":
                        display_role = "사용자"
                    elif role == "ai":
                        display_role = "AI"
                    else:
                        display_role = "unknown"
                
                data.append({
                    "시간": created_at,
                    "발화자": display_role,
                    "내용": actual_content
                })
            

            df = pd.DataFrame(data)
            

            if not os.path.exists(self.cfg.export_dir):
                os.makedirs(self.cfg.export_dir)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            excel_filename = f"session_{session_id}_{timestamp}.xlsx"
            excel_path = os.path.join(self.cfg.export_dir, excel_filename)
            
            df.to_excel(excel_path, index=False, engine='openpyxl')
            
            print(f"📊 총 {len(data)}개 메시지")
            print(f"✅ 엑셀 파일 생성됨: {excel_path}")
            return excel_path
            
        except Exception as e:
            import traceback
            print(f"[ERROR] 엑셀 추출 실패: {e}")
            print(f"[ERROR] 상세 오류: {traceback.format_exc()}")
            return None


    def add_dialog(self, text: str, act_type: str):
        try:

            self.conversation_memory.save_context({"input": text}, {"output": act_type})


            if hasattr(self, "summary_memory"):
                self.summary_memory.save_context({"input": text}, {"output": act_type})

            print(f"[MEMORY] Added dialog: ({act_type}) {text}")

        except Exception as e:
            print(f"[MEMORY ERROR] add_dialog(): {e}")
    



    def flush_memory_to_excel(self, session_id: str):
        try:
            user_name = self.user_names.get(session_id or "default_session", "사용자")
            if user_name and user_name != "사용자":

                try:
                    buffered = getattr(self.excel_manager, "_buffered_changes", {})
                    has_user_buffers = any(k for k in buffered.keys() if k[0] == user_name)
                    if not has_user_buffers:
                        logger.info("[FLUSH] 버퍼 비어있음 - flush 생략")
                        return
                except Exception:
                    pass

                self.excel_manager.request_flush(user_name)
                logger.info(f"[FLUSH] 세션({session_id}) 데이터 엑셀로 동기화 예약 ({user_name})")
            else:
                logger.warning(f"[FLUSH] 사용자 이름이 없어 flush 건너뜀: {session_id}")
        except Exception as e:
            logger.error(f"[ERROR] flush_memory_to_excel 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
