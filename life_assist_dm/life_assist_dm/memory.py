from __future__ import annotations
# 🔧 LangChain DeprecationWarning 억제
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

# LangChain imports
from langchain.schema import HumanMessage, AIMessage
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy import create_engine, text

# Confidence threshold 상수 (task_classifier.py와 일관성 유지)
CONFIDENCE_THRESHOLD = 0.6

# 로깅 설정
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
    """메모리 동작 설정값"""
    sqlite_path: str = "~/.life_assist_dm/history.sqlite"
    chroma_dir: str = "~/.life_assist_dm/chroma"
    summary_enabled: bool = True
    retriever_search_k: int = 5
    buffer_window: int = 3
    auto_export_enabled: bool = True
    export_dir: str = "conversation_extract"


# 최소 normalization dict (동의어 통일용)
NORMALIZE_KEYS = {
    "엄마": "어머니",
    "아빠": "아버지",
    "핸드폰": "휴대폰",
    "핸폰": "휴대폰",
}

# 불용어 (조사는 _normalize_location에서 처리하므로 제외)
STOPWORDS = {
    "한", "번", "쯤", "시쯤", "식후에", "식전에", "반쯤", "시",
    "만", "도", "만큼", "정도", "쯤", "가량", "약", "대략"
}

# 이름 추출 금지 단어 (단어 단위 매칭만 적용)
NAME_BLACKLIST = {
    "오늘", "어제", "내일", "그제", "아침", "점심", "저녁", "밤", "낮",
    "그냥", "그래", "응", "네", "아니", "맞아", "좋아", "싫어",
    "먹었", "먹어", "마셔", "마셨", "갔", "왔", "있", "없",
    "했", "했어", "할", "할게", "할래", "하고", "해서",
    "그", "이", "저", "내", "네", "우리", "너", "나", "저희",
    "뭐라고", "누구게", "뭐게", "뭔가", "뭔지", "뭐지", "뭐야",
    "누구야", "누구지", "누구인지", "누구인가", "누구인가요"
}

# 약 관련 키워드 (중복 제거를 위한 공통 상수)
MEDICINE_KEYWORDS = [
    "알약", "처방", "복용", "복용법", "복용시간", "식후", "식전", "공복",
    "비타민", "영양제", "오메가3", "오메가 3", "철분제", "프로틴", "보충제",
    "유산균", "프로바이오틱스", "마그네슘", "칼슘", "아연", "엽산"
]

# 약 패턴 (정규식)
COMMON_MED_PATTERNS = [
    r"(비타민\s*[A-Z]?\d*)", r"(오메가\s*3)", r"(오메가3)", r"(철분제)", r"(프로틴)",
    r"(보충제)", r"(영양제)", r"(유산균)", r"(유산균제)", r"(프로바이오틱스)",
    r"(마그네슘)", r"(칼슘)", r"(아연)", r"(엽산)", r"([가-힣A-Za-z]+)\s*(?:을|를)?\s*먹"
]

# 복용 방법 패턴 (정규식)
METHOD_PATTERNS = [
    r"식후\s*(\d+)\s*분", r"식후\s*(\d+)분", r"식전", r"공복", r"식사\s*후", r"식사\s*전"
]

# 시간대 키워드 매핑
TIME_OF_DAY_KEYWORDS = {
    "아침": ["아침", "조식", "morning", "breakfast", "기상", "일어나자마자", "일어나자 마자", "기상 후", "기상 시"],
    "점심": ["점심", "중식", "lunch"],
    "저녁": ["저녁", "석식", "dinner", "evening"]
}

# 복용 방법 키워드 매핑
MEDICATION_METHOD_KEYWORDS = {
    "공복": ["공복", "공복에"],
    "식전": ["식전", "식전에"],
    "식후": ["식후", "식후에"]
}

# 한글 숫자 변환 딕셔너리 (문자열 변환용)
KOREAN_NUMBERS_STR = {
    "한": "1", "두": "2", "세": "3", "네": "4", "다섯": "5",
    "여섯": "6", "일곱": "7", "여덟": "8", "아홉": "9", "열": "10"
}

# 한글 숫자 변환 딕셔너리 (정수 변환용)
KOREAN_NUMBERS_INT = {
    "한": 1, "두": 2, "세": 3, "네": 4, "다섯": 5,
    "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9, "열": 10
}


class LifeAssistMemory:
    """생활 지원 메모리 클래스"""

    def __init__(self, cfg: MemoryConfig, session_id: str = "default", debug: bool = False):
        self.cfg = cfg
        self.session_id = session_id
        
        # 로깅 설정
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
        # 시간 컨텍스트 저장 (이전 대화에서 시간 정보 연결용)
        self.time_context = {}  # {session_id: {"last_time": "7시반", "last_meal": "점심"}}
        
        # 감정 상태 추적 (일관성 있는 응답을 위해)
        self.emotional_state = {}  # {session_id: {"mood": "negative", "intensity": 0.8, "last_emotional_turn": 3}}
        
        # 물리적 작업 재질문 상태 관리 (pending_question으로 통합됨)
        
        # 사용자 이름 세션별 추적
        self.user_names = {}  # {session_id: user_name}
        
        # LLM 초기화
        from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm, get_embedding
        self.llm = get_llm()
        self.debug = debug

        # 경로 준비
        self.sqlite_path = str(Path(os.path.expanduser(cfg.sqlite_path)))
        Path(self.sqlite_path).parent.mkdir(parents=True, exist_ok=True)

        # 엑셀 전용으로 변경 - VectorStore 제거
        self.vectorstore = None
        self.vector_store = None
        self.retriever = None
        
        # Excel 관리자 인스턴스 초기화 (버퍼링 구조)
        from .user_excel_manager import UserExcelManager
        self.excel_manager = UserExcelManager()
        
        # 임베딩 기반 캐시 시스템
        self.classification_cache = {}  # {text: ClassificationResult}
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)
        self.cache_embeddings = None
        self.cache_texts = []
        self.similarity_threshold = 0.95  # 유사도 임계값 
        
        # 날짜 정규화 캐시 시스템
        self.date_cache = {}  # {date_str: normalized_date}
        self.max_date_cache_size = 1000  # 최대 캐시 크기

        # 엔티티 추출 체인
        self.entity_chain = self._build_entity_chain()

        # LCEL 체인 초기화 (SQLite 백엔드 사용)
        from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
        from langchain_community.chat_message_histories import SQLChatMessageHistory
        
        # SQLite 백엔드 설정
        engine = create_engine(f"sqlite:///{self.sqlite_path}")
        self._ensure_message_table_exists(engine)
        
        # SQLite 백엔드를 사용하는 메모리 초기화
        self.conversation_memory = ConversationBufferMemory(
            memory_key="history",
            return_messages=True,
            chat_memory=SQLChatMessageHistory(
                session_id="default_session",
                connection=engine
            )
        )
        
        # 요약 메모리 초기화 (ConversationSummaryBufferMemory)
        from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm, get_embedding
        llm = get_llm()
        self.summary_memory = ConversationSummaryBufferMemory(
            llm=llm,
            max_token_limit=1000,
            return_messages=True,
            memory_key="summary_history"
        )

        # 상태 관리
        self.pending_context: Dict[str, str] = {}
        self.asked_pending: Dict[str, str] = {}
        self.pending_question: Dict[str, dict] = {}  # 재질문 상태 관리
        self.current_question: Dict[str, str] = {}  # 현재 재질문 저장
        self._init_sqlite()

    # SQLite 테이블 생성 (기존 데이터 보존)
    def _init_sqlite(self):
        # 제거됨: SQLite 비사용 로그
        conn = sqlite3.connect(self.sqlite_path)
        c = conn.cursor()
        
        # 기존 기록 보존하며 테이블 생성
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
        
        # 대화 요약 전용 테이블 생성
        c.execute(
            "CREATE TABLE IF NOT EXISTS conversation_summaries ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, "
            "summary_text TEXT NOT NULL, "
            "token_count INTEGER, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
            "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        
        # 세션별 대화 히스토리 테이블 생성
        c.execute(
            "CREATE TABLE IF NOT EXISTS conversation_history ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT NOT NULL, "
            "message_type TEXT NOT NULL, "  # 'human' or 'ai'
            "message_content TEXT NOT NULL, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        
        # updated_at 자동 갱신 트리거 추가
        c.execute(
            "CREATE TRIGGER IF NOT EXISTS trg_update_summary "
            "AFTER UPDATE ON conversation_summary "
            "BEGIN "
            "UPDATE conversation_summary SET updated_at = CURRENT_TIMESTAMP "
            "WHERE id = NEW.id; "
            "END;"
        )
        
        # message_store 테이블 생성 (SQLChatMessageHistory 호환)
        c.execute(
            "CREATE TABLE IF NOT EXISTS message_store ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, "
            "session_id TEXT, "
            "role TEXT, "
            "content TEXT, "
            "message TEXT, "
            "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
        )
        
        # created_at 컬럼이 없으면 추가
        try:
            c.execute("ALTER TABLE message_store ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        except sqlite3.OperationalError:
            # 이미 created_at 컬럼이 있으면 무시
            pass
        
        # 세션별 인덱스 생성
        c.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON conversation_summary(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_message_session_id ON message_store(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_summaries_session_id ON conversation_summaries(session_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_history_session_id ON conversation_history(session_id)")
        
        # message 컬럼 추가 (SQLite는 DROP COLUMN을 지원하지 않으므로 content는 유지)
        try:
            c.execute("ALTER TABLE message_store ADD COLUMN message TEXT")
        except sqlite3.OperationalError:
            # 이미 message 컬럼이 있으면 무시
            pass
        
        # role 컬럼 추가 (LangChain SQLChatMessageHistory 호환)
        try:
            c.execute("ALTER TABLE message_store ADD COLUMN role TEXT")
        except sqlite3.OperationalError:
            # 이미 role 컬럼이 있으면 무시
            pass
        
        # 추가 필요한 컬럼들 (LangChain 호환성)
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
        # 제거됨: SQLite 비사용 로그

# _load_session_context 함수 제거됨 - load_previous_session_data로 통합

    def _get_user_entities_from_excel(self) -> List[str]:
        """Excel 캐시 기반으로 사용자 관련 엔티티 정보 가져오기"""
        entities = []
        try:
            session_id = "default_session"
            if hasattr(self, "excel_cache"):
                sess = self.excel_cache.get(session_id, {})
                # 사용자 이름
                user = sess.get("사용자", [])
                if user and user[0].get("이름"):
                    entities.append(f"사용자 이름: {user[0]['이름']}")
                # 물건 위치
                items = sess.get("물건", [])
                for it in items:
                    if it.get("이름") and it.get("위치"):
                        entities.append(f"{it['이름']}: {it['위치']}")
        except Exception as e:
            print(f"[ERROR] 엔티티 조회 실패: {e}")
        return entities

    

    def get_location(self, target: str, return_dict: bool = False) -> Optional[Union[str, dict]]:
        """특정 물건의 저장된 위치 조회 (세션 캐시 → 엑셀). VectorStore 미사용.
        
        Args:
            target: 물건 이름
            return_dict: True면 dict 형태로 반환 (장소, 세부위치 분리), False면 문자열 반환
        """
        try:
            session_id = "default_session"
            # 1) 세션 캐시
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
            # 2) 엑셀
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
                            # nan, None 필터링
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
                            # 위치가 없으면 장소와 세부위치를 조합
                            place = str(row.get("장소", "") or "").strip()
                            sub_location = str(row.get("세부위치", "") or "").strip()
                            # nan, None 필터링
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
        """물건 위치를 엑셀에 저장하고 세션 캐시에 반영."""
        try:
            session_id = "default_session"
            user_name = self.user_names.get(session_id)
            if not user_name:
                return
            self.excel_manager.save_entity_data(user_name, "물건", {"물건이름": item_name, "위치": location})
            # 캐시에 반영
            if not hasattr(self, 'excel_cache'):
                self.excel_cache = {}
            session_cache = self.excel_cache.setdefault(session_id, {})
            items = session_cache.setdefault("물건", [])
            if overwrite:
                # 기존 동일 이름 제거 후 추가
                items = [it for it in items if it.get("이름") != item_name]
                session_cache["물건"] = items
            items.append({"이름": item_name, "위치": location})
        except Exception as e:
            print(f"[ERROR] save_location 실패: {e}")

    def _build_context_for_llm(self, user_input: str, session_id: str) -> str:
        """LLM 응답 생성을 위한 통합 맥락 구성 (엑셀 캐시 + 대화 히스토리)."""
        try:
            context_parts = []
            
            # 1. LCEL Chain에서 현재 세션 대화 히스토리 로딩
            try:
                mem_vars = self.conversation_memory.load_memory_variables({})
                history = mem_vars.get("history", "")
                if history:
                    context_parts.append(f"[현재 세션 대화 히스토리]\n{history}")
            except Exception as e:
                print(f"[WARN] LCEL Chain 히스토리 로딩 실패: {e}")
            
            # 2. 엑셀 캐시에서 최근 저장 정보 구성
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
            
            # 3. 맥락 통합
            if context_parts:
                return "\n\n".join(context_parts) + "\n\n"
            else:
                return ""
                
        except Exception as e:
            print(f"[ERROR] 맥락 구성 실패: {e}")
            return ""

    def end_session(self, session_id: str) -> str:
        """세션 종료 시 전체 대화 요약 생성 및 저장"""
        try:
            print(f"[DEBUG] 세션 종료: {session_id}")
            
            # 현재 대화 메모리에서 메시지 가져오기
            messages = self.conversation_memory.chat_memory.messages
            
            if len(messages) < 2:  # 최소 1번의 대화 (human + ai)
                print(f"[DEBUG] 요약할 대화가 부족함: {len(messages)}개 메시지")
                return ""
            
            # 대화 내용을 텍스트로 변환
            conversation_text = ""
            for msg in messages:
                if hasattr(msg, 'content'):
                    role = "사용자" if msg.__class__.__name__ == "HumanMessage" else "AI"
                    conversation_text += f"{role}: {msg.content}\n"
            
            # 요약 생성
            from life_assist_dm.life_assist_dm.llm.gpt_utils import get_llm, get_embedding
            llm = get_llm()
            
            summary_prompt = f"""
            다음 대화 내용을 간단히 요약해주세요. 주요 정보(이름, 위치, 일정 등)와 중요한 대화 내용만 포함해주세요.
            
            대화 내용:
            {conversation_text}                            
            요약:
            """
            
            summary = llm.invoke(summary_prompt).content.strip()
            
            # SQLite에 요약 저장
            conn = sqlite3.connect(self.sqlite_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO conversation_summaries (session_id, summary_text, token_count) VALUES (?, ?, ?)",
                (session_id, summary, len(summary.split()))
            )
            
            conn.commit()
            conn.close()
            
            print(f"[DEBUG] 세션 요약 저장 완료: {summary[:50]}...")
            
            # 대화 메모리 초기화 (다음 세션을 위해)
            self.conversation_memory.clear()
            
            return summary
            
        except Exception as e:
            print(f"[ERROR] 세션 종료 요약 생성 실패: {e}")
            return ""


    # 제거: SQLite 비사용 정책에 따라 대화 히스토리 저장 함수 삭제

    def load_user_data_from_excel(self, user_name: str, session_id: str):
        """엑셀 파일에서 사용자 데이터 로딩 (SQLite/VectorStore 대신)"""
        try:
            print(f"[DEBUG] 엑셀에서 사용자 데이터 로딩 시작: {user_name}")
            
            if not self.excel_manager.user_exists(user_name):
                print(f"[DEBUG] 사용자 엑셀 파일 없음: {user_name}")
                return
            
            # 엑셀에서 각 시트 데이터 로딩하여 메모리에 임시 저장
            sheets_data = {}
            
            # 물건 위치 로딩
            df_items = self.excel_manager.load_sheet_data(user_name, "물건위치")
            if not df_items.empty:
                items = []
                for _, row in df_items.iterrows():
                    # 과거 데이터 호환: '물건이름' 우선, 없으면 '이름' 사용
                    name_val = row.get("물건이름", None)
                    if name_val is None or name_val == "":
                        name_val = row.get("이름", "")
                    loc_val = row.get("위치", "")
                    try:
                        import pandas as pd
                        if pd.isna(name_val):
                            name_val = ""
                        if pd.isna(loc_val):
                            loc_val = ""
                    except Exception:
                        pass
                    items.append({
                        "이름": name_val,
                        "위치": loc_val
                    })
                sheets_data["물건"] = items
                print(f"[DEBUG] 물건 위치 {len(items)}개 로딩")
            
            # 가족 관계 로딩
            df_family = self.excel_manager.load_sheet_data(user_name, "가족관계")
            if not df_family.empty:
                family = []
                for _, row in df_family.iterrows():
                    family.append({
                        "관계": row.get("관계", ""),
                        "이름": row.get("이름", ""),
                        "정보": row.get("정보", "")
                    })
                sheets_data["가족"] = family
                print(f"[DEBUG] 가족 관계 {len(family)}개 로딩")
            
            # 일정 로딩
            df_schedule = self.excel_manager.load_sheet_data(user_name, "일정")
            if not df_schedule.empty:
                schedules = []
                for _, row in df_schedule.iterrows():
                    schedules.append({
                        "제목": row.get("일정내용", ""),
                        "날짜": row.get("날짜", ""),
                        "시간": row.get("시간", ""),
                        "장소": row.get("장소", "")
                    })
                sheets_data["일정"] = schedules
                print(f"[DEBUG] 일정 {len(schedules)}개 로딩")
            
            # 대화 기록 로딩 (최근 10개) 및 LCEL 메모리에 추가
            df_conversations = self.excel_manager.load_sheet_data(user_name, "대화기록")
            if not df_conversations.empty:
                # 최근 10개 로딩 (더 많은 맥락 유지를 위해)
                recent_conversations = df_conversations.tail(10)
                conversations = []
                
                # LCEL 메모리에 대화 기록 추가
                try:
                    # 기존 세션 메모리가 있는지 확인하고, 없으면 새로 생성
                    chat_memory = self.conversation_memory.chat_memory
                    
                    # 기존 메시지가 없을 때만 엑셀 대화 기록 추가 (중복 방지)
                    existing_messages = chat_memory.messages
                    if not existing_messages or len(existing_messages) == 0:
                        print(f"[DEBUG] 기존 메시지 없음 - 엑셀 대화 기록을 LCEL 메모리에 로딩")
                        
                        for _, row in recent_conversations.iterrows():
                            conv_summary = row.get("대화요약", "")
                            if not conv_summary or str(conv_summary).strip() == "":
                                continue
                            
                            conv_text = str(conv_summary).strip()
                            conversations.append({
                                "날짜": row.get("날짜", ""),
                                "시간": row.get("시간", ""),
                                "대화요약": conv_text
                            })
                            
                            # 대화 요약 파싱: "Q: 질문 | A: 답변" 형식
                            question = None
                            answer = None
                            
                            # 1. "Q: 질문 | A: 답변" 형식 처리
                            if "Q:" in conv_text and "A:" in conv_text:
                                # " | A:" 패턴으로 split
                                if " | A:" in conv_text:
                                    parts = conv_text.split(" | A:", 1)
                                    if len(parts) == 2:
                                        question = parts[0].replace("Q:", "").strip()
                                        answer = parts[1].strip()
                                else:
                                    # "A:" 기준으로 split (대체 방법)
                                    parts = conv_text.split("A:", 1)
                                    if len(parts) == 2:
                                        question = parts[0].replace("Q:", "").strip()
                                        answer = parts[1].strip()
                            
                            # 2. "질문 | 답변" 형식 처리 (Q: A: 없이)
                            elif "|" in conv_text and not question:
                                parts = conv_text.split("|", 1)
                                if len(parts) == 2:
                                    question = parts[0].strip()
                                    answer = parts[1].strip()
                            
                            # 3. 파싱된 질문/답변을 메모리에 추가
                            if question and answer:
                                chat_memory.add_user_message(question)
                                chat_memory.add_ai_message(answer)
                            elif question:
                                # 질문만 있는 경우
                                chat_memory.add_user_message(question)
                            elif answer:
                                # 답변만 있는 경우 (일반적이지 않지만)
                                chat_memory.add_ai_message(answer)
                            else:
                                # 단순 텍스트인 경우 - 사용자 메시지로 간주
                                # (시스템 메시지나 요약 등은 건너뛰기)
                                if conv_text:
                                    # 너무 짧거나 시스템 메시지 같은 것은 건너뛰기
                                    skip_keywords = ["세션", "타임아웃", "종료", "초기화"]
                                    if len(conv_text) > 5 and not any(kw in conv_text for kw in skip_keywords):
                                        chat_memory.add_user_message(conv_text)
                        
                        print(f"[DEBUG] LCEL 메모리에 대화 기록 {len(conversations)}개 추가 완료")
                    else:
                        print(f"[DEBUG] 기존 메시지 존재 ({len(existing_messages)}개) - 엑셀 대화 기록 추가 건너뜀")
                        
                        # 캐시용으로만 저장
                        for _, row in recent_conversations.iterrows():
                            conversations.append({
                                "날짜": row.get("날짜", ""),
                                "시간": row.get("시간", ""),
                                "대화요약": row.get("대화요약", "")
                            })
                except Exception as e:
                    print(f"[ERROR] LCEL 메모리에 대화 기록 추가 실패: {e}")
                    import traceback
                    traceback.print_exc()
                    # 실패해도 캐시는 저장
                    for _, row in recent_conversations.iterrows():
                        conversations.append({
                            "날짜": row.get("날짜", ""),
                            "시간": row.get("시간", ""),
                            "대화요약": row.get("대화요약", "")
                        })
                
                sheets_data["대화"] = conversations
                print(f"[DEBUG] 대화 기록 {len(conversations)}개 로딩 (캐시)")

            # 사용자 개인정보 로딩: 사용자정보KV에서 최신값 계산
            from .dialog_manager.config.config_loader import get_excel_sheets
            sheets = get_excel_sheets()
            df_userkv = self.excel_manager.load_sheet_data(user_name, sheets.get("user_info_kv", "사용자정보KV"))
            if not df_userkv.empty:
                latest_map = {}
                try:
                    # 최근 행이 가장 아래라고 가정하고 역순으로 스캔하여 최초 매칭을 최신으로 사용
                    for _, row in df_userkv.iloc[::-1].iterrows():
                        key = str(row.get("키", "")).strip()
                        if not key:
                            continue
                        val = row.get("값", "")
                        sval = str(val).strip()
                        if sval.lower() in ("nan", "none"):
                            continue
                        if key and key not in latest_map and sval != "":
                            latest_map[key] = sval
                    user_row = {
                        "이름": latest_map.get("이름", ""),
                        "나이": latest_map.get("나이", ""),
                        "학교": latest_map.get("학교", ""),
                        "직업": latest_map.get("직업", ""),
                        "취미": latest_map.get("취미", ""),
                        "날짜": df_userkv.iloc[-1].get("날짜", "")
                    }
                    sheets_data["사용자"] = [user_row]
                    print(f"[DEBUG] 사용자정보(KV) 로딩 완료: keys={list(k for k,v in latest_map.items() if v)}")
                except Exception as e:
                    print(f"[ERROR] 사용자정보KV 처리 실패: {e}")
            
            # 세션별 임시 저장소에 저장
            if not hasattr(self, 'excel_cache'):
                self.excel_cache = {}
            self.excel_cache[session_id] = sheets_data
            
            print(f"[DEBUG] 엑셀 데이터 로딩 완료: {user_name}")
            
        except Exception as e:
            print(f"[ERROR] 엑셀 데이터 로딩 실패: {e}")
    
    def get_excel_data(self, session_id: str, data_type: str):
        """엑셀에서 로딩된 데이터 조회"""
        if not hasattr(self, 'excel_cache'):
            return []
        
        session_data = self.excel_cache.get(session_id, {})
        return session_data.get(data_type, [])

    def handle_duplicate_answer(self, user_input: str, pending_data: dict) -> dict:
        """
        중복 엔티티 재질문에 대한 사용자 응답 처리 (엑셀 전용 단순화)
        """
        text = (user_input or "").strip().lower()
        
        # 긍정 키워드 (저장)
        positive = ["응", "어", "그래", "맞아", "바꿔", "업데이트", "덮어", "새로", "다시", "저장해", "좋아", "네", "예", "추가", "함께", "같이", "둘다", "또", "새로운", "더", "그리고", "또한"]
        # 부정 키워드 (취소)
        negative = ["아니", "아냐", "아닌", "그냥", "놔둬", "유지", "그대로", "안돼", "싫어", "아니요", "아니야", "취소"]
        
        # 부정 응답 → 취소
        if any(k in text for k in negative):
            return {
                "success": True,
                "duplicate": False,
                "message": "알겠어요. 저장하지 않을게요."
            }
        
        # 긍정 응답 → 엑셀에 저장
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
        
        # 모호한 응답 → 재질문
        return {
            "success": False,
            "duplicate": True,
            "message": "저장할까요, 아니면 취소할까요? (예/아니오)"
        }

    def _check_duplicate_entity(self, entity_type: str, new_data: dict, session_id: str = None) -> dict:
        """엑셀 기반 중복 엔티티 확인."""
        try:
            user_name = self.user_names.get((session_id or "default_session"))
            if not user_name:
                return {"is_duplicate": False}

            # 사용자 이름
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

            # 물건
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
        """엑셀 기반 엔티티 삭제."""
        try:
            user_name = self.user_names.get("default_session")
            if not user_name:
                return
            if entity_type == "물건":
                df = self.excel_manager.load_sheet_data(user_name, "물건위치")
                if df is not None and not df.empty:
                    # 삭제된 데이터를 버퍼에 저장 (기존 행을 제외하고 새로 저장)
                    filtered_records = []
                    for _, row in df.iterrows():
                        if str(row.get("물건이름", "")) != str(existing_value):
                            filtered_records.append(row.to_dict())
                    # 버퍼에 저장 후 flush 필요 (즉시 반영)
                    if filtered_records:
                        for record in filtered_records:
                            self.excel_manager.save_entity_data(user_name, "물건", record)
                        # ✅ request_flush() 사용하여 지연 병합 처리 (race condition 방지)
                        self.excel_manager.request_flush(user_name)
        except Exception as e:
            print(f"[WARN] 엑셀 엔티티 삭제 실패: {e}")

    def _check_missing_fields(self, entity_type: str, data: dict) -> dict:
        """
        필수 필드가 누락되었는지 확인하고, 누락 시 재질문 메시지 반환
        """
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
        """
        약명 + 용량 + 단위 추출 (다중 약도 지원)
        예) "아스피린 2알, 비타민 1알" → [{"이름":"아스피린","용량":"2","단위":"알"}, ...]
        예) "나는 아침마다 비타민 먹어" → [{"이름":"비타민","용량":"","단위":""}]
        ✅ 개선: "~약" 패턴(비염약, 혈압약 등)을 우선적으로 추출
        """
        medicines = []
        
        
        med_match = re.search(r"([가-힣A-Za-z]+약)", text)
        if med_match:
            med_name = med_match.group(1)
            dose_match = re.search(rf"{re.escape(med_name)}.*?(\d+)\s*(알|정|캡슐|포|mg|ml|병)(?:씩)?(?!\s*(?:분|시간|시))", text)
            
            if dose_match:
                # ✅ 매칭된 숫자 앞의 텍스트 확인
                match_pos = dose_match.start()
                number_pos = text.find(dose_match.group(1), match_pos)
                before_number = text[:number_pos]
                
                # ✅ "식후/식전/공복 + 숫자 + 분/시간" 패턴이 앞에 있으면 용량이 아님
                # 예: "근육통약 식후 30분" → "30"은 용량이 아님
                if re.search(r"(식후|식전|공복|식사\s*(?:전|후))\s*\d+\s*(?:분|시간|시)", before_number + dose_match.group(0)):
                    # "식후 30분" 같은 패턴이 있으면 용량이 아님
                    medicines.append({
                        "이름": med_name,
                        "용량": "",
                        "단위": ""
                    })
                else:
                    # 실제 용량인 경우
                    medicines.append({
                        "이름": med_name,
                        "용량": dose_match.group(1).strip(),
                        "단위": dose_match.group(2)
                    })
            else:
                # ✅ 한글 숫자 패턴: "한 알", "두 알", "세 알" 등
                # "번", "회"는 빈도 단위이므로 용량 단위가 아님
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
            # "약명 + 약" 패턴 (마그네슘 약, 칼슘 약 등) - 공백 포함
            med_match = re.search(r"([가-힣A-Za-z]+)\s+약", text)
            if med_match:
                med_name = med_match.group(1).strip()
                medicines.append({
                    "이름": med_name,
                    "용량": "",
                    "단위": ""
                })
            else:
                # ✅ 우선순위 2: "약명 + 숫자 + 단위" 패턴
                # ✅ "번", "회"는 빈도 단위이므로 용량 단위가 아님
                pattern = r"([가-힣A-Za-z]+)\s*(\d+)\s*(알|정|캡슐|포|mg|ml|병)?"
                matches = re.findall(pattern, text)

                for match in matches:
                    name, dose, unit = match
                    # 숫자 단위(하루, 식후 등)가 아닌 실제 약명인지 확인
                    if name not in ["하루", "식후", "식전", "아침", "점심", "저녁"]:
                        medicines.append({
                            "이름": name.strip(),
                            "용량": dose.strip(),
                            "단위": unit if unit else "알"   # 기본 단위
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
        """
        끼니/날짜/메뉴 추출
        예) "오늘 아침에 김치찌개 먹었어" → {"날짜":"2025-10-01","끼니":"아침","메뉴":["김치찌개"]}
        """
        # import re는 전역에서 이미 import되어 있으므로 제거
        from datetime import datetime, timedelta
        
        meal = {"날짜": None, "끼니": None, "메뉴": []}

        
        try:
            has_med_keyword = any(keyword in text for keyword in MEDICINE_KEYWORDS)
            # "약" 키워드 체크 (단, "약속"은 제외 - 약속은 일정)
            if not has_med_keyword and "약" in text and "약속" not in text:
                # "~약" 패턴 (혈압약, 감기약 등) 또는 "약 먹" 패턴 체크
                if re.search(r"[가-힣A-Za-z]+약|약\s*[먹드]", text):
                    has_med_keyword = True
            # "~약" 패턴 체크 (안전하게 - 길이 제한, "약속" 제외)
            if not has_med_keyword and len(text) < 200:
                if re.search(r"[가-힣A-Za-z]+약(?!속)", text):
                    has_med_keyword = True
            if has_med_keyword:
                return meal  # 빈 식사 엔티티 반환
        except Exception as e:
            print(f"[WARN] 약 키워드 체크 중 오류: {e}")

        # 끼니 패턴
        for time_key, keywords in TIME_OF_DAY_KEYWORDS.items():
            if any(k in text for k in keywords):
                meal["끼니"] = time_key
                break

        # 날짜 추출 (예: 오늘/내일/어제/요일)
        if "어제" in text:
            meal["날짜"] = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        elif "오늘" in text:
            meal["날짜"] = datetime.now().strftime("%Y-%m-%d")
        elif "내일" in text:
            meal["날짜"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
        elif "모레" in text:
            meal["날짜"] = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        # TODO: "목요일" 같은 요일 파싱도 추가 가능

        # 음식 추출 (단순히 "먹었어/먹음" 앞뒤 명사 추출)
        food_pattern = r"(?:\s|^)([가-힣A-Za-z]+)\s*(먹었|먹음|먹다|먹을)"
        matches = re.findall(food_pattern, text)
        if matches:
            meal["메뉴"] = [m[0] for m in matches]

        return meal

    def extract_with_fallback(self, text: str, entity_type: str):
        """
        규칙 기반 → 실패 시 LLM 보완 추출
        """
        if entity_type == "약":
            meds = self._extract_medicine_entities(text)
            if meds:
                return meds
        elif entity_type == "식사":
            meal = self._extract_meal_entity(text)
            if meal.get("끼니") or meal.get("메뉴"):
                return meal

        # 규칙 기반 실패 → LLM 추출기로 fallback
        try:
            # 기존 LLM 체인을 사용하여 엔티티 추출
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
        """SQLite에서 최근 대화 요약 불러오기"""
        try:
            conn = sqlite3.connect(self.sqlite_path)
            c = conn.cursor()
            
            # 최근 3개의 대화 요약 불러오기
            c.execute("""
                SELECT summary FROM conversation_summary 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT 3
            """, (session_id,))
            
            summaries = c.fetchall()
            conn.close()
            
            if summaries:
                # 요약들을 시간순으로 결합
                recent_summaries = [summary[0] for summary in reversed(summaries)]
                return "\n".join(recent_summaries)
            else:
                return ""
                
        except Exception as e:
            print(f"[ERROR] 대화 요약 불러오기 실패: {e}")
            return ""

    def _convert_conversation_history_to_string(self, conversation_history) -> str:
        """LCEL conversation_history를 문자열로 변환 (길이 제한 적용)"""
        if isinstance(conversation_history, str):
            # 문자열인 경우 길이 제한 (최대 2000자)
            if len(conversation_history) > 2000:
                return conversation_history[-2000:] + "..."
            return conversation_history
        elif isinstance(conversation_history, list):
            # 메시지 객체 리스트인 경우 문자열로 변환 (최대 10개 메시지)
            history_text = ""
            recent_messages = conversation_history[-10:]  # 최근 10개 메시지만 사용
            for msg in recent_messages:
                if hasattr(msg, 'content'):
                    history_text += f"{msg.content}\n"
                else:
                    history_text += f"{str(msg)}\n"
            
            # 전체 길이 제한 (최대 2000자)
            if len(history_text) > 2000:
                return history_text[-2000:] + "..."
            return history_text.strip()
        else:
            return str(conversation_history) if conversation_history else ""

    def _get_current_timestamp(self) -> str:
        """현재 타임스탬프 반환"""
        from datetime import datetime
        return datetime.now().isoformat()

# _get_similar_cached_result 함수 제거됨 - _get_cached_classification으로 통합

# _update_classification_cache 함수 제거됨 - _add_to_cache로 통합

    def save_entity_to_vectorstore(self, entity_type: str, data: dict, session_id: str = None) -> dict:
        """
        엔티티를 VectorStore와 SQLite에 저장 (중복 검증 포함)
        - 중복이 있으면 저장하지 않고, 재질문 메시지를 반환
        - 성공적으로 저장되면 {"success": True, "message": "..."} 반환
        """
        import json
        from datetime import datetime
        
        try:
            # ---------- 1. 중복 체크 ----------
            print(f"[DEBUG] 중복 확인 시작: entity_type={entity_type}, data={data}")
            duplicate_info = self._check_duplicate_entity(entity_type, data, session_id)
            print(f"[DEBUG] 중복 확인 결과: {duplicate_info}")

            if duplicate_info.get("is_duplicate"):
                # 중복이 있으면 저장 중단 → 상위 체인에서 재질문 발화
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

            # ---------- 2. 실제 저장 ----------
            # 통일된 포맷으로 변환
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
            
            # 엑셀 파일에만 저장 (VectorStore/SQLite 제거)
            try:
                user_name = self.user_names.get(session_id or "default")
                
                # 사용자 이름 유효성 검증
                if user_name and user_name != "사용자" and len(user_name.strip()) > 0:
                    # 저장 전 추가 검증
                    import re
                    if not re.match(r'^[가-힣A-Za-z0-9\s]+$', user_name) or len(user_name) > 20:
                        print(f"[WARNING] 유효하지 않은 사용자 이름: {user_name}")
                        user_name = None
                
                if user_name and user_name != "사용자":
                    # 버퍼링 저장 (즉시 저장하지 않음)
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

    # 대화 히스토리 (SQLChatMessageHistory 사용)
    def _history(self, session_id: str) -> SQLChatMessageHistory:
        # Deprecation warning 방지를 위해 connection 사용
        engine = create_engine(f"sqlite:///{self.sqlite_path}")
        
        self._ensure_message_table_exists(engine)
        
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=engine
        )
    
    def _ensure_message_table_exists(self, engine):
        """message_store 테이블 스키마 보장"""
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

    # 최근 요약 가져오기 (1개만)
    def _get_recent_summaries(self, session_id: str, limit: int = 3) -> List[str]:
        """최근 n개의 요약 가져오기"""
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

    # 안전한 JSON 파싱
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

    # 날짜/시간 정규화 (추측 금지 → 시간 None 처리)
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

        # 오전/오후 시각 (숫자로 명시된 경우만)
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

    # 엔티티 추출 LLM 체인
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
        """물건 위치 추출 (Rule-based 우선 처리)"""
        out: Dict[str, List[Dict[str, Any]]] = {}
        t = user_input.strip()
        
        # 문장을 쉼표로 분리하여 각각 처리
        sentences = [s.strip() for s in t.split(',') if s.strip()]
        
        for sentence in sentences:
            # 물건 위치 패턴들 (다양한 표현 지원)
            location_patterns = [
                # 기본 패턴: "내 물컵은 주방 찬장 안에 있어"
                r"(?:내|네|이|그)\s*(.+?)\s*(?:은|는)\s*(.+?)\s*(?:안에|위에|밖에|옆에|앞에|뒤에|아래에|에)\s*(?:있어|있고|있어요|있습니다|둬|놔|두|놓|보관)",
                # 일반 패턴: "물건은 위치에 있어"
                r"(.+?)\s*(?:은|는)\s*(.+?에|위에|안에|밖에|옆에|앞에|뒤에|아래에)\s*(?:있|둬|놔|두|놓|보관)",
                # 위치+물건 순서: "주방 찬장에 물컵이 있어"
                r"(.+?)\s*(?:에|위에|안에|밖에|옆에|앞에|뒤에|아래에)\s*(.+?)\s*(?:이|가)?\s*(?:있어|있고|있어요|있습니다)",
                # "물건 위치에 있어" 패턴
                r"(.+?)\s*(.+?에|위에|안에|밖에|옆에|앞에|뒤에|아래에)\s*(?:에|에서)\s*(?:있|둬|놔|두|놓|보관)",
                # "물건을 위치에 두었어" 패턴
                r"(.+?)\s*(?:을|를)\s*(.+?에|위에|안에|밖에|옆에|앞에|뒤에|아래에)\s*(?:두었|놓았|보관했)",
                # 추가 패턴: "물건은 위치에 있고"
                r"(.+?)\s*(?:은|는)\s*(.+?에|위에|안에|밖에|옆에|앞에|뒤에|아래에)\s*(?:있고|있어)",
                # 추가 패턴: "물건은 위치에 있고" (더 간단한 버전)
                r"(.+?)\s*(?:은|는)\s*(.+?에)\s*(?:있고|있어)",
            ]
            
            for pattern in location_patterns:
                m = re.search(pattern, sentence)
                if m:
                    # 패턴에 따라 그룹 순서가 다를 수 있음
                    groups = m.groups()
                    if len(groups) >= 2:
                        # 첫 번째 그룹이 물건인지 위치인지 판단
                        g1, g2 = groups[0].strip(), groups[1].strip()
                        
                        # 위치 키워드가 포함된 그룹을 찾기
                        location_keywords = ["에", "위에", "안에", "밖에", "옆에", "앞에", "뒤에", "아래에", "주방", "거실", "침실", "찬장", "서랍", "책상", "방"]
                        
                        if any(kw in g2 for kw in location_keywords):
                            # g1이 물건, g2가 위치
                            item = re.sub(r"^(내|네|이|그)\s*", "", g1)
                            location = g2
                        elif any(kw in g1 for kw in location_keywords):
                            # g1이 위치, g2가 물건 (역순)
                            location = g1
                            item = re.sub(r"^(내|네|이|그)\s*", "", g2)
                        else:
                            # 기본적으로 첫 번째 그룹이 물건
                            item = re.sub(r"^(내|네|이|그)\s*", "", g1)
                            location = g2
                        
                        # 유효한 물건명과 위치인지 확인
                        if (item and location and 
                            len(item) >= 1 and len(location) >= 2 and
                            item not in ["것", "거", "이것", "그것", "저것"]):
                            
                            # 장소와 세부위치 분리
                            # ✅ "내방 책상" 같은 복합 장소는 하나의 장소로 보고, "위에", "안에" 같은 방향만 세부위치로 분리
                            direction_keywords = ["위에", "옆에", "앞에", "뒤에", "아래에", "안에", "밖에", "에"]
                            
                            place = None
                            sub_location = ""
                            
                            # 1단계: 방향 키워드 추출 (세부위치)
                            # "위에", "옆에" 등 명확한 방향 키워드를 우선 체크
                            direction_found = None
                            for direction in ["위에", "옆에", "앞에", "뒤에", "아래에", "안에", "밖에"]:
                                if location.endswith(direction):
                                    direction_found = direction
                                    location_without_direction = location[:-len(direction)].strip()
                                    break
                            
                            # "안에"가 없지만 "안"으로 끝나는 경우 (예: "내 방 안")
                            if not direction_found and location.endswith("안"):
                                direction_found = "안"
                                location_without_direction = location[:-1].strip()
                            
                            # 2단계: 장소 추출
                            if direction_found:
                                # ✅ 방향 키워드가 있으면 나머지 부분 전체를 장소로, 방향을 세부위치로
                                # 예: "내 방 안" → place="내 방", sub_location="안"
                                # 예: "내방 책상 위에" → place="내방 책상", sub_location="위에"
                                place = location_without_direction
                                sub_location = direction_found
                            elif location.endswith("에"):
                                # ✅ "에"만 있는 경우도 처리 (예: "내방 책상에")
                                place = location[:-1].strip()  # "에" 제거
                                sub_location = "에"
                            else:
                                # ✅ 방향 키워드가 없으면 전체를 장소로, 세부위치는 빈 문자열
                                # 예: "내방 책상" → place="내방 책상", sub_location=""
                                place = location
                                sub_location = ""
                            
                            out.setdefault("user.물건", []).append({
                                "이름": item,
                                "위치": location,  # 하위 호환성을 위해 전체 위치도 유지
                                "장소": place,
                                "세부위치": sub_location,
                                "추출방법": "rule-based"
                            })
                            break  # 첫 번째 매치만 사용
        
        return out

    def _extract_item_command_rule(self, user_input: str) -> Dict[str, List[Dict[str, Any]]]:
        """물건 명령 추출 (Rule-based) - 꺼내와, 가져와, 찾아줘 등"""
        out: Dict[str, List[Dict[str, Any]]] = {}
        t = user_input.strip()
        
        # 물건 명령 패턴들
        command_patterns = [
            # "위치에서 물건 꺼내와/가져와"
            r"(.+?에서|에)\s*(.+?)\s*(?:꺼내와|가져와|꺼내다|가져다|꺼내줘|가져다줘)",
            # "물건 꺼내와/가져와"
            r"(.+?)\s*(?:꺼내와|가져와|꺼내다|가져다|꺼내줘|가져다줘)",
            # "물건 찾아줘"
            r"(.+?)\s*(?:찾아줘|찾아다|찾아)",
            # "물건 어디 있어?"
            r"(.+?)\s*(?:어디|위치).*?(?:있어|있나)",
        ]
        
        for pattern in command_patterns:
            matches = re.findall(pattern, t)
            for match in matches:
                if isinstance(match, tuple):
                    if len(match) == 2:
                        location, item = match
                        # 위치에서 물건 추출
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
                        # 물건만 추출
                        item = match[0] if match else ""
                        if item and len(item) >= 1 and item not in ["것", "거", "이것", "그것", "저것"]:
                            out.setdefault("user.물건", []).append({
                                "이름": item,
                                "위치": None,
                                "추출방법": "command-rule"
                            })
                else:
                    # 단일 매치
                    item = match.strip()
                    if item and len(item) >= 1 and item not in ["것", "거", "이것", "그것", "저것"]:
                        out.setdefault("user.물건", []).append({
                            "이름": item,
                            "위치": None,
                            "추출방법": "command-rule"
                        })
        
        return out

    # 규칙 기반 추출 (LLM 보완용, 추측 금지)
    def _rule_based_extract(self, text: str, session_id: str = None) -> Dict[str, Any]:
        # print(f"[DEBUG] _rule_based_extract 호출: '{text}'")
        out: Dict[str, Any] = {}
        groups = []  # ✅ 반드시 초기화 (일정 패턴 매칭에서 사용)
        try:
            t = text.strip() if text else ""
            if not t:
                return out

            # 질문성 문장은 스킵
            if re.search(r"\?$", t):
                return out

            # 약 복용 패턴 체크: 약 키워드 + "먹" 또는 "복용" 키워드
            # ✅ "약속"은 일정이므로 약물로 처리하지 않음
            has_medicine_keyword = False
            try:
                has_medicine_keyword = any(keyword in t for keyword in MEDICINE_KEYWORDS)
                # "약" 키워드 체크 (단, "약속"은 제외 - 약속은 일정)
                if not has_medicine_keyword and "약" in t and "약속" not in t:
                    # "~약" 패턴 (혈압약, 감기약 등) 또는 "약 먹" 패턴 체크
                    if re.search(r"[가-힣A-Za-z]+약|약\s*[먹드]", t):
                        has_medicine_keyword = True
                # "~약" 패턴 체크 (안전하게 - 길이 제한, "약속" 제외)
                if not has_medicine_keyword and len(t) < 200:
                    if re.search(r"[가-힣A-Za-z]{1,20}약(?!속)", t):
                        has_medicine_keyword = True
            except Exception as e:
                print(f"[WARN] 약 키워드 체크 중 오류: {e}")
            
            # "약" + "먹" 또는 "복용" 패턴 체크 (단, "약속"은 제외 - 약속은 일정)
            has_medicine_pattern = False
            try:
                # "약속"이 아닌 경우에만 약 복용 패턴 체크
                if "약속" not in t:
                    # 기본 패턴: "약.*?먹|약.*?복용|복용"
                    has_medicine_pattern = bool(re.search(r"약.*?먹|약.*?복용|복용", t)) or (has_medicine_keyword and "먹" in t)
                    
                    # 추가 패턴: "알" 단위 + "먹" 또는 복용 방법 키워드 (공복, 식전, 식후 등)
                    if not has_medicine_pattern:
                        # "X알" + "먹" 또는 복용 방법 패턴
                        if re.search(r"\d+\s*알.*?먹|\d+\s*알.*?(공복|식전|식후)|[한두세네다섯]\s*알.*?먹", t):
                            has_medicine_pattern = True
                        # 복용 방법 키워드가 있으면 약으로 인식 (공복, 식전, 식후 등)
                        elif re.search(r"(공복|식전|식후|복용)", t) and re.search(r"\d+\s*알|[한두세네다섯]\s*알", t):
                            has_medicine_pattern = True
                else:
                    # "약속"이 있으면 약 복용 패턴이 아닌 것으로 처리
                    has_medicine_pattern = False
            except Exception as e:
                print(f"[WARN] 약 패턴 체크 중 오류: {e}")
            
            # 식사 관련 키워드 (약과 구분)
            has_food_keyword = False
            try:
                food_keywords = r"(밥|식사|음식|요리|메뉴|김치|찌개|국|탕|면|라면|치킨|피자|햄버거|떡볶이|삼겹살|갈비)"
                has_food_keyword = bool(re.search(food_keywords, t))
            except Exception as e:
                print(f"[WARN] 식사 키워드 체크 중 오류: {e}")
            
            if has_medicine_pattern and not has_food_keyword:
                # 새로운 약 엔티티 추출 함수 사용
                try:
                    print(f"[DEBUG] 약 복용 패턴 매칭: {t}")
                    medicines = self._extract_medicine_entities(t)
                    print(f"[DEBUG] _extract_medicine_entities 결과: {medicines}")
                    
                    # ✅ 시간대 추출 (여러 시간대 지원: "아침 점심 저녁" → "아침/점심/저녁")
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
                    
                    # ✅ "하루 X번" 같은 빈도 표현을 시간대로 변환
                    # "하루 3번" → "아침/점심/저녁", "하루 2번" → "아침/저녁" 등
                    if not time_of_day_list:
                        # "하루 X번" 패턴 추출
                        frequency_match = re.search(r"하루\s*(?:에\s*)?(\d+|[한두세네다섯])\s*번", t)
                        if frequency_match:
                            freq_str = frequency_match.group(1)
                            # 숫자 또는 한글 숫자를 정수로 변환
                            if freq_str.isdigit():
                                frequency = int(freq_str)
                            else:
                                frequency = KOREAN_NUMBERS_INT.get(freq_str, 0)
                            
                            # 빈도에 따라 시간대 설정
                            # ✅ "하루 1번"은 시간대를 설정하지 않음 (아침일 수도 저녁일 수도 있음)
                            if frequency == 1:
                                # 하루 1번은 특정 시간대로 변환하지 않음
                                time_of_day_list = []
                            elif frequency == 2:
                                time_of_day_list = ["아침", "저녁"]
                            elif frequency == 3:
                                time_of_day_list = ["아침", "점심", "저녁"]
                            elif frequency >= 4:
                                # 4번 이상이면 기본적으로 아침/점심/저녁/밤
                                time_of_day_list = ["아침", "점심", "저녁", "밤"]
                    
                    # 여러 시간대가 있으면 "/"로 구분하여 저장
                    if time_of_day_list:
                        time_of_day = "/".join(time_of_day_list)
                    else:
                        time_of_day = None
                    
                    # 날짜 추출 (오늘/내일/모레 등) - 중복 import 제거
                    date_str = None
                    if "오늘" in t or "지금" in t:
                        date_str = datetime.now().strftime("%Y-%m-%d")
                    elif "내일" in t:
                        date_str = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
                    elif "모레" in t:
                        date_str = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
                    else:
                        # 날짜 언급이 없으면 기본값으로 오늘
                        date_str = datetime.now().strftime("%Y-%m-%d")
                    
                    if medicines:
                        # ✅ 복용방법 추출 (식후 30분, 공복에 등)
                        # import re는 전역에서 이미 import되어 있으므로 제거
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
                        
                        # ✅ 복용기간 추출 (일주일 동안, 한 달 동안, 복용 기간은 일주일 등)
                        복용기간_값 = ""
                        # 우선순위: "복용 기간" 키워드가 있는 경우 먼저 처리
                        # "복용 기간은 일주일이야", "복용 기간은 일주일" 등 다양한 표현 지원
                        # "은/는"이 포함될 수 있고, 다양한 종결어미도 허용
                        period_with_keyword_patterns = [
                            r"복용\s*기간\s*(?:은|는|이|가)?\s*일주일(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                            r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*일(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                            r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*주(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                            r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*개월(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                            r"복용\s*기간.*?일주일",  # 더 넓은 패턴 (fallback)
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
                        
                        # "복용 기간" 키워드가 없으면 일반 패턴 검사
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
                            # 약 복용 엔티티 생성 (개선된 버전)
                            # ✅ 시간대를 실제 추출된 값으로 설정 (기본값 "미정" 사용 최소화)
                            actual_time = time_of_day or ""
                            # 시간대가 비어있으면 "미정" 대신 빈 문자열로 설정 (더 정확한 중복 감지)
                            medication_entity = {
                                "약명": medicine.get("이름", ""),
                                "용량": medicine.get("용량", ""),
                                "단위": medicine.get("단위", ""),
                                "시간대": actual_time,  # ✅ 빈 문자열 또는 실제 시간대 사용
                                "복용": "예정" if "먹을" in t else "완료",
                                "날짜": date_str,
                                "복용방법": 복용방법_값,  # ✅ 추가
                                "복용기간": 복용기간_값   # ✅ 추가
                            }
                            
                            out.setdefault("user.약", []).append(medication_entity)
                            print(f"[DEBUG] 약 복용 엔티티 추출: {medication_entity}")
                        return out
                    else:
                        # 약 이름이 없는 경우 fallback
                        # ✅ fallback에서도 복용방법과 복용기간 추출 시도
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
                            # 복용 기간 패턴 재검사
                            period_with_keyword_patterns = [
                                r"복용\s*기간\s*(?:은|는|이|가)?\s*일주일(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                                r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*일(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                                r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*주(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                                r"복용\s*기간\s*(?:은|는|이|가)?\s*(\d+)\s*개월(?:\s*(?:이야|이에요|입니다|동안|치|이다|다|어|아))?",
                                r"복용\s*기간.*?일주일",  # 더 넓은 패턴 (fallback)
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
                            "시간대": actual_time,  # ✅ 빈 문자열 또는 실제 시간대 사용
                            "복용": "예정" if "먹을" in t else "완료",
                            "날짜": date_str,
                            "복용방법": 복용방법_값,  # ✅ 추가
                            "복용기간": 복용기간_값   # ✅ 추가
                        }
                        
                        out.setdefault("user.약", []).append(medication_entity)
                        print(f"[DEBUG] 약 복용 엔티티 추출 (fallback, 약 이름 없음): {medication_entity}")
                        return out
                except Exception as e:
                    print(f"[ERROR] 약 복용 엔티티 추출 중 오류: {e}")
                    import traceback
                    traceback.print_exc()
                    # 오류 발생 시 계속 진행

            # ✅ 일정 관련 키워드가 포함된 경우, 일정 추출을 먼저 시도 (물건 추출보다 우선)
            schedule_keywords = ["약속", "모임", "회식", "미팅", "미팅이", "일정", "예약", "만남", "데이트", "모임이", "약속이"]
            has_schedule_keyword = any(k in t for k in schedule_keywords)
            
            # ✅ 일정 추출을 먼저 시도 (물건 추출보다 우선)
            # 일정 추출은 아래에서 처리되므로 여기서는 플래그만 설정
            # 일정 키워드가 있으면 물건 추출을 완전히 건너뛰기
            if has_schedule_keyword:
                print(f"[DEBUG] [RULE] 일정 키워드 감지 → 물건 추출 완전 제외: {t}")
            else:
                # ✅ 물건 위치 추출: "내 물컵은 주방 찬장 안에 있어" 등
                # 일정 키워드가 없을 때만 물건 위치 추출 시도
                item_location_keywords = ["있어", "위치", "서랍", "찬장", "방", "주방", "책상", "물건", "보관", "놓았", "두었", "안에", "위에"]
                has_item_location_keywords = any(k in t for k in item_location_keywords)
                
                # 물건 위치 엔티티가 아직 없고, 약/식사 패턴이 아닐 때만 물건 위치 추출
                if has_item_location_keywords and "user.물건" not in out:
                    try:
                        item_extracted = self._extract_item_location_rule(t)
                        if item_extracted and item_extracted.get("user.물건"):
                            out.setdefault("user.물건", []).extend(item_extracted["user.물건"])
                            print(f"[DEBUG] _rule_based_extract에서 물건 위치 추출: {item_extracted['user.물건']}")
                    except Exception as e:
                        print(f"[DEBUG] 물건 위치 추출 중 오류: {e}")
                        pass

            # ✅ 사용자 개인정보 추출 (이름, 나이, 학교 등)
            try:
                print(f"[DEBUG] 사용자 개인정보 추출 시작: '{t}'")
            except Exception:
                pass
            
            # 나이 추출
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
            
            # 학교 추출
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
        
            # 직업 추출
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
                    # 직업 관련 키워드가 있는 경우만
                    if any(keyword in job_candidate for keyword in ["학생", "회사원", "선생님", "의사", "간호사", "엔지니어", "개발자", "디자이너"]):
                        job = job_candidate
                        break
            
            # 개인정보가 있으면 사용자 엔티티 생성
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
        
            # 1차: 질문 패턴 확인 (새로운 엔티티 생성하지 않음)
            if self._is_name_question(t):
                print(f"[DEBUG] 이름 질문 패턴으로 인해 스킵")
                return out  # 질문은 새로운 엔티티 생성하지 않음
            
            # 2차: LLM 기반 이름 및 별칭 추출 (문맥 이해)
            llm_result = self._extract_name_llm(t)
            if llm_result and llm_result.get("name"):
                user_entity = {"이름": llm_result["name"], "확인됨": True}
                if llm_result.get("alias"):
                    user_entity["별칭"] = llm_result["alias"]
                out.setdefault("user.사용자", []).append(user_entity)
            else:
                # 3차: 규칙 기반 fallback (본명과 별칭 구분)
                clean_text = re.sub(r'[\.,!?]+$', '', t.strip())
                
                # 가족 이름 패턴이 포함된 경우 사용자 이름 추출 스킵
                family_patterns = [
                    r"우리\s*(동생|엄마|아빠|형|누나|언니|오빠|할머니|할아버지)",
                    r"(동생|엄마|아빠|형|누나|언니|오빠|할머니|할아버지)\s*이름",
                    r"가족\s*이름"
                ]
                
                is_family_context = any(re.search(pattern, clean_text) for pattern in family_patterns)
                
                # 본명 추출 패턴 (가족 컨텍스트와 관계없이 정의)
                name_patterns = [
                    r"내\s*이름(?:은|이)?\s*([가-힣A-Za-z\s]{2,10})(?:이야|이에요|입니다|예요|야|다|어|아)?",
                    r"나는\s*([가-힣A-Za-z\s]{2,10})(?:야|이다|입니다|이에요|예요)",
                    r"저는\s*([가-힣A-Za-z\s]{2,10})(?:입니다|이에요|예요|야)",
                    r"난\s*([가-힣A-Za-z\s]{2,10})(?:야|이다|이야)"
                ]
                
                # 가족 컨텍스트가 아닌 경우에만 사용자 이름 추출 시도
                if not is_family_context:
                    # 본명 추출 시도 (별명은 LLM이 처리하므로 여기서는 본명만)
                    for pattern in name_patterns:
                        m = re.search(pattern, clean_text)
                        if m:
                            name = self._normalize_name(m.group(1))
                            if self._is_valid_name(name):
                                out.setdefault("user.사용자", []).append({"이름": name, "확인됨": True})
                                break
                
                # 가족 맥락이어도 '내 이름' 패턴은 허용
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

            # ✅ 가족 (원문 substring 검증 + 이름 추출)
            # ✅ FAMILY_RELATION_KEYWORDS와 일치하도록 확장: 며느리, 사위 등 추가
            family_patterns = [
                r"(남편|아내|엄마|어머니|아빠|아버지|아들|딸|형|누나|동생|언니|오빠|할머니|할아버지|손자|손녀|손주|며느리|사위|부모)"
            ]
            for pattern in family_patterns:
                m = re.search(pattern, t)
                if m:
                    rel = NORMALIZE_KEYS.get(m.group(1), m.group(1))
                    # 원문에 실제로 존재하는지 검증
                    if m.group(1) in t:
                        family_info = {"관계": rel}
                    
                    # 가족 이름 추출 (예: "아빠 이름은 홍길동", "우리 동생 이름은 권서율이야")
                    # **주의: 가족 이름만 추출하고 사용자 이름과 절대 혼동하지 않음**
                    name_patterns = [
                        f"{m.group(1)}\\s*이름(?:은|이)?\\s*([가-힣A-Za-z]{{2,}})(?:이야|이에요|입니다|야|다|어|아|이고|이고요)?",
                        f"{m.group(1)}\\s*([가-힣A-Za-z]{{2,}})(?:이야|이에요|입니다|야|다|어|아|이고|이고요)(?!\\s*(?:이름|은|이|이다|입니다))"  # "이름", "은", "이", "이다", "입니다" 뒤에 오는 것은 제외
                    ]
                    
                    name_found = False
                    for name_pattern in name_patterns:
                        name_match = re.search(name_pattern, t)
                        if name_match:
                            name = self._normalize_name(name_match.group(1))
                            # 관계명 자체는 이름에 들어가지 않도록 제외 + "이름은" 같은 불완전한 표현 제외
                            if (name 
                                and name not in NAME_BLACKLIST 
                                and name != rel 
                                and name not in ["이름은", "이름이", "이름", "은", "이", "이다", "입니다"]):
                                family_info["이름"] = name
                                name_found = True
                            break
                    
                    # 추가 패턴: "권서율이라고" 같은 경우 처리
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
                    
                    # 이름이 있을 때만 가족 정보 추가 (중복 방지)
                    if name_found:
                        # 이미 동일한 가족 정보가 있는지 확인
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

            # ✅ 약 추출 (개선된 약별 복용 정보 분리)
            if (re.search(r"\b약\b", t) or 
                re.search(r"[가-힣A-Za-z]+약", t) or 
                any(drug in t for drug in ["아스피린", "타이레놀", "이부프로펜", "아세트아미노펜"])):
                drugs = self._extract_drugs_with_info(t)
                if drugs:
                    out.setdefault("user.약", []).extend(drugs)

            # ✅ 일정 추출 (단순화된 패턴)
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
            
            # 일정 취소 처리
            for pattern in cancel_patterns:
                m = re.search(pattern, t)
                if m:
                    title = m.group(1).strip()
                    print(f"[DEBUG] 일정 취소 감지: '{title}'")
                    # 취소된 일정을 VectorStore에서 삭제
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
                    ampm_part = ""  # ✅ 초기화 추가
                    
                    
                    is_appointment_pattern = (
                        (i == 4) or  # 패턴 인덱스 4 (약속 패턴 - "이번 주 금요일 저녁 7시에 친구랑 약속 있어")
                        (("이번\\s*주" in pattern or "다음\\s*주" in pattern or "이번주" in pattern or "다음주" in pattern) and
                         ("약속|모임|만남" in pattern or "약속" in pattern or "모임" in pattern or "만남" in pattern))
                    )
                    
                    print(f"[DEBUG] 패턴 {i+1} 검사: is_appointment_pattern={is_appointment_pattern}, len(groups)={len(groups)}")
                    
                    if is_appointment_pattern and len(groups) >= 6:
                        # groups[0] = 이번주/다음주, groups[1] = 금요일, groups[2] = 저녁, groups[3] = 7시, groups[4] = 친구, groups[5] = 약속
                        print(f"[DEBUG] 약속 패턴 감지됨 (인덱스 {i}) - 일정으로 처리")
                        week_part = groups[0] if len(groups) > 0 and groups[0] else ""
                        day_part = groups[1] if len(groups) > 1 and groups[1] else ""
                        ampm_part = groups[2] if len(groups) > 2 and groups[2] else ""
                        time_hour_part = groups[3] if len(groups) > 3 and groups[3] else ""
                        person_part = groups[4] if len(groups) > 4 and groups[4] else ""
                        keyword_part = groups[5] if len(groups) > 5 and groups[5] else ""
                        
                        # 날짜 문자열 정규화
                        if week_part and day_part:
                            date_part = f"{week_part.strip()} {day_part.strip()}".strip()
                        elif week_part:
                            date_part = week_part.strip()
                        elif day_part:
                            date_part = day_part.strip()
                        else:
                            date_part = ""
                        
                        # 시간 문자열 정규화
                        if ampm_part and time_hour_part:
                            time_part = f"{ampm_part.strip()} {time_hour_part.strip()}".strip()
                        elif ampm_part:
                            time_part = ampm_part.strip()
                        elif time_hour_part:
                            time_part = time_hour_part.strip()
                        else:
                            time_part = ""
                        
                        # 제목 문자열 정규화
                        if person_part and keyword_part:
                            title_part = f"{person_part.strip()} {keyword_part.strip()}".strip()
                        elif person_part:
                            title_part = person_part.strip()
                        elif keyword_part:
                            title_part = keyword_part.strip()
                        else:
                            title_part = "약속"  # 기본 제목
                        
                        # 일정 정보가 있으면 바로 저장하고 break
                        if title_part or date_part:
                            schedule_info = {
                                "제목": title_part,
                                "날짜": date_part,
                                "시간": time_part
                            }
                            out.setdefault("user.일정", []).append(schedule_info)
                            print(f"[DEBUG] 일정 엔티티 추출 (약속 패턴): {schedule_info}")
                            # ✅ 일정이 저장되었으면 물건 엔티티 제거 (이미 추가되었다면)
                            if "user.물건" in out:
                                print(f"[DEBUG] 일정 엔티티 저장됨 → 물건 엔티티 제거: {out.get('user.물건')}")
                                out.pop("user.물건", None)
                            # ✅ 일정이 저장되었으면 함수 종료 (물건 추출 방지)
                            return out
                    
                    # ✅ 패턴 4 (인덱스 3): (이번주/다음주) (요일) (제목) - 일정 처리
                    # 패턴: (다음\s*주|이번\s*주|다음주|이번주)\s*([가-힣]+요일)\s*(?:에|에는)?\s*([가-힣A-Za-z\s]+?)\s*(?:가야|해야|있어|있어요)
                    if i == 3 and len(groups) >= 3 and ("이번" in pattern or "다음" in pattern) and "주" in pattern and "요일" in pattern:
                        # groups[0] = 이번주/다음주, groups[1] = 요일, groups[2] = 제목
                        week_part = groups[0] if len(groups) > 0 and groups[0] else ""
                        day_part = groups[1] if len(groups) > 1 and groups[1] else ""
                        title_part = groups[2] if len(groups) > 2 and groups[2] else ""
                        
                        # 날짜 문자열 정규화
                        if week_part and day_part:
                            date_part = f"{week_part.strip()} {day_part.strip()}".strip()
                        elif week_part:
                            date_part = week_part.strip()
                        elif day_part:
                            date_part = day_part.strip()
                        else:
                            date_part = ""
                        
                        # 제목 정리
                        if title_part:
                            title_clean = re.sub(r"^(는|은|이|가|을|를)\s*", "", title_part.strip())
                            title_clean = re.sub(r"(가|이|을|를)\s*$", "", title_clean.strip())
                        else:
                            title_clean = ""
                        
                        # 일정 정보가 있으면 바로 저장하고 return
                        if title_clean or date_part:
                            schedule_info = {
                                "제목": title_clean,
                                "날짜": date_part,
                                "시간": ""
                            }
                            out.setdefault("user.일정", []).append(schedule_info)
                            print(f"[DEBUG] 일정 엔티티 추출 (패턴 4): {schedule_info}")
                            # ✅ 일정이 저장되었으면 물건 엔티티 제거
                            if "user.물건" in out:
                                print(f"[DEBUG] 일정 엔티티 저장됨 → 물건 엔티티 제거: {out.get('user.물건')}")
                                out.pop("user.물건", None)
                            # ✅ 일정이 저장되었으면 함수 종료 (물건 추출 방지)
                            return out
                    
                    # 패턴 1 처리: "내일 4시에 미팅있어" - (날짜) (시간) (제목) (있어)
                    if i == 0 and len(groups) >= 3 and "있어" in pattern:
                        date_part = groups[0] if groups[0] else "오늘"
                        time_part = groups[1] if groups[1] else ""
                        title_part = groups[2] if groups[2] else ""
                        ampm_part = ""
                        # 제목 정리 (공백 제거, 조사 제거)
                        if title_part:
                            title_part = re.sub(r"(가|이|을|를|은|는)\s*$", "", title_part.strip())
                        # 일정 저장
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
                    # 패턴 2 처리: "내일 오후 3시에 병원 예약이 있어" - (날짜) (오후/오전?) (시간) (제목) (있어)
                    elif i == 1 and len(groups) >= 4 and "있어" in pattern:
                        date_part = groups[0] if groups[0] else "오늘"
                        ampm_part = groups[1] if groups[1] else ""
                        time_part = groups[2] if groups[2] else ""
                        title_part = groups[3] if groups[3] else ""
                        # 오후/오전 정보가 있으면 시간에 포함
                        if ampm_part:
                            time_part = f"{ampm_part} {time_part}"
                        # 제목 정리
                        if title_part:
                            title_part = re.sub(r"(가|이|을|를|은|는)\s*$", "", title_part.strip())
                        # 일정 저장
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
                    # 첫 번째 패턴: (날짜) (오후/오전?) (시간) (제목)
                    elif pattern.startswith(r"(내일|오늘|어제)") and len(groups) >= 4:
                        date_part = groups[0] if groups[0] else "오늘"
                        ampm_part = groups[1] if groups[1] else ""
                        time_part = groups[2] if groups[2] else ""
                        title_part = groups[3] if groups[3] else ""
                        
                        # 제목에서 시간 정보 추출 (예: "일본 여행, 아침" → "일본 여행"과 "아침")
                        if title_part and "," in title_part:
                            parts = title_part.split(",")
                            if len(parts) >= 2:
                                title_part = parts[0].strip()
                                time_info = parts[1].strip()
                                # 시간 정보가 있으면 기존 시간과 결합
                                if time_info:
                                    time_part = f"{time_info} {time_part}" if time_part else time_info
                        
                        # 오후/오전 정보가 있으면 시간에 포함
                        if ampm_part:
                            time_part = f"{ampm_part} {time_part}"
                    
                    # ✅ 새로운 패턴: (날짜/시간 표현) (저녁|오후?) (시간?) (제목) (약속|모임|만남) (있어|있어요)
                    elif "약속|모임|만남" in pattern and "있어|있어요" in pattern and "이번.*주|다음.*주" not in pattern:
                        # groups[0] = 날짜/시간 표현, groups[1] = 저녁/오후 등, groups[2] = 시간, groups[3] = 제목, groups[4] = 약속/모임/만남
                        if len(groups) >= 5:
                            # 날짜 정보 추출
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
                            # 그룹이 부족하면 기본값 사용
                            date_match = re.search(r"(이번\s*주|다음\s*주|이번주|다음주|오늘|내일|어제)", t)
                            if date_match:
                                date_part = date_match.group(1)
                            else:
                                date_part = "오늘"
                            time_part = ""
                            title_part = groups[0] if len(groups) > 0 else ""
                    
                    # 네 번째 패턴: (다음 주|이번 주) (요일) (제목) - 약속/모임 없음
                    elif "다음" in pattern and "주" in pattern and "약속|모임|만남" not in pattern:
                        date_part = f"{groups[0]} {groups[1]}" if len(groups) > 1 else groups[0] if len(groups) > 0 else ""
                        title_part = groups[2] if len(groups) > 2 else ""
                        time_part = ""  # 시간 정보 초기화
                # 두 번째 패턴: (제목) (가야/해야) (날짜?) (시간?)
                elif "가야" in pattern and "해야" in pattern:
                    title_part = groups[0] if len(groups) > 0 else ""
                    date_part = groups[1] if len(groups) > 1 and groups[1] else "오늘"
                    time_part = groups[2] if len(groups) > 2 and groups[2] else ""
                # 세 번째 패턴: (병원|회의|약속|미팅|데이트|일정|스케줄|예약)
                elif "병원|회의|약속|미팅|데이트|일정|스케줄|예약" in pattern:
                    title_part = groups[0] if len(groups) > 0 else ""
                    time_part = ""  # 시간 정보 초기화
                    # 날짜 정보 추출
                    date_match = re.search(r"(오늘|내일|어제|다음주|이번주)", t)
                    if date_match:
                        date_part = date_match.group(1)
                # 네 번째 패턴: (다음 주|이번 주) (요일) (제목)
                elif "다음.*주|이번.*주" in pattern:
                    date_part = f"{groups[0]} {groups[1]}" if len(groups) > 1 else groups[0] if len(groups) > 0 else ""
                    title_part = groups[2] if len(groups) > 2 else ""
                    time_part = ""  # 시간 정보 초기화
                # 다섯 번째 패턴: (월 일) (제목)
                elif r"\d{1,2}\s*월\s*\d{1,2}\s*일" in pattern:
                    date_part = groups[0] if len(groups) > 0 else ""
                    title_part = groups[1] if len(groups) > 1 else ""
                    time_part = ""  # 시간 정보 초기화
                # 여섯 번째 패턴: (오늘|내일|어제) (저녁|오후|오전) (시) (제목)
                elif "로\s*했어|로\s*했어요|만나기로\s*했어|만나기로\s*했어요" in pattern:
                    date_part = groups[0] if len(groups) > 0 else ""
                    time_part = f"{groups[1]} {groups[2]}" if len(groups) > 2 and groups[1] and groups[2] else ""
                    title_part = groups[3] if len(groups) > 3 else ""
                # 새로운 패턴: (내일|오늘|어제) (제목), (아침|오후|오전|저녁) (시간) (내용)
                elif r"내일|오늘|어제.*,\s*아침|오후|오전|저녁" in pattern:
                    date_part = groups[0] if len(groups) > 0 else ""
                    title_part = groups[1] if len(groups) > 1 else ""
                    time_part = f"{groups[2]} {groups[3]}" if len(groups) > 3 and groups[2] and groups[3] else ""
                    # 제목과 시간이 바뀌어서 파싱된 경우 수정
                    if title_part and time_part and title_part.isdigit() and not time_part.isdigit():
                        # 제목과 시간을 바꿔서 저장
                        temp = title_part
                        title_part = time_part
                        time_part = temp
                    # 추가 수정: "일본 여행 아침"을 "일본 여행"과 "아침"으로 분리
                    if title_part and "여행" in title_part and "아침" in title_part:
                        # "일본 여행 아침" -> "일본 여행"과 "아침"으로 분리
                        if "아침" in title_part:
                            title_part = title_part.replace(" 아침", "").replace("아침", "")
                            time_part = f"아침 {time_part}" if time_part else "아침"
                # 새로운 패턴: (내일|오늘|어제) (제목) (있어|있어요|있습니다)
                elif r"내일|오늘|어제.*있어|있어요|있습니다" in pattern and len(groups) == 2:
                    date_part = groups[0] if len(groups) > 0 else ""
                    title_part = groups[1] if len(groups) > 1 else ""
                    time_part = ""
                # 새로운 패턴: (제목), (내일|오늘|어제) (아침|오후|오전|저녁) (시간)
                elif r",\s*내일|오늘|어제.*아침|오후|오전|저녁" in pattern:
                    title_part = groups[0] if len(groups) > 0 else ""
                    date_part = groups[1] if len(groups) > 1 else ""
                    time_part = f"{groups[2]} {groups[3]}" if len(groups) > 3 and groups[2] and groups[3] else ""
                # 패턴 7: (내일|오늘|어제) (제목), (아침|오후|오전|저녁) (시간) (추가정보)
                elif "야|이야|예요|에요" in pattern:
                    date_part = groups[0] if len(groups) > 0 else ""
                    title_part = groups[1] if len(groups) > 1 else ""
                    time_part = f"{groups[2]} {groups[3]}" if len(groups) > 3 and groups[2] and groups[3] else ""
                # 패턴 8: (내일|오늘|어제) (제목) (있어|있어요|있습니다)
                elif len(groups) == 2 and "있어" in t:
                    date_part = groups[0] if len(groups) > 0 else ""
                    title_part = groups[1] if len(groups) > 1 else ""
                    time_part = ""
                
                    # ✅ 일정 정보 정리 및 저장
                    if title_part or date_part:
                        # 제목에서 불필요한 조사 제거
                        if title_part:
                            title_clean = re.sub(r"^(는|은|이|가|을|를)\s*", "", title_part.strip())
                            title_clean = re.sub(r"(가|이|을|를)\s*$", "", title_clean.strip())
                        else:
                            # 제목이 없으면 날짜 정보를 제목으로 사용
                            title_clean = date_part if date_part else "일정"
                        
                        # 날짜가 없으면 기본값
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
                        # ✅ 일정이 저장되었으면 물건 엔티티 제거 (이미 추가되었다면)
                        if "user.물건" in out:
                            print(f"[DEBUG] 일정 엔티티 저장됨 → 물건 엔티티 제거: {out.get('user.물건')}")
                            out.pop("user.물건", None)
                        # ✅ 일정이 저장되었으면 함수 종료 (물건 추출 방지)
                        return out

            # ✅ 생일 추출 보강
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
            
            # 다른 기념일들
            m = re.search(r"(제사|기일|결혼기념일).*?(\d{1,2}\s*월\s*\d{1,2}\s*일)", t)
            if m:
                out.setdefault("user.기념일", []).append({
                    "관계": "",
                    "제목": m.group(1),
                    "날짜": m.group(2)
                })

            # 취미
            m = re.search(r"취미(?:는|가)?\s*([가-힣A-Za-z0-9 ]+)", t)
            if m:
                hobby = re.sub(r"(이야|야|입니다|예요|에요)$", "", m.group(1)).strip()
                if hobby:
                    out.setdefault("user.취미", []).append({"이름": hobby})

            # ✅ 취향 (패턴 확장)
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
                    # 조사 제거 (를, 을, 가, 이)
                    val = re.sub(r'(를|을|가|이)$', '', val).strip()
                    # "나는", "난", "전", "저는" 제거
                    val = re.sub(r'^(나는|난|전|저는)\s*', '', val).strip()
                    if val and val not in STOPWORDS:
                        out.setdefault("user.취향", []).append({"종류": "", "값": val})
                    break

            # 약 복용 정보 추출
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
                        # 여러 약물
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
                        # 복용 주기
                        frequency = m.group(1)
                        # 이전 약 정보에 복용 주기 추가
                        if "user.약" in out and out["user.약"]:
                            for med in out["user.약"]:
                                med["복용주기"] = f"하루 {frequency}번"
                    elif "일" in pattern or "주" in pattern or "개월" in pattern or "년" in pattern:
                        # 복용 기간
                        period = m.group(1) + ("일" if "일" in pattern else "주" if "주" in pattern else "개월" if "개월" in pattern else "년")
                        # 이전 약 정보에 복용 기간 추가
                        if "user.약" in out and out["user.약"]:
                            for med in out["user.약"]:
                                med["복용기간"] = period
                    else:
                        # 단일 약물
                        med_name = m.group(1)
                        out.setdefault("user.약", []).append({
                            "약명": med_name,
                            "복용량": None,
                            "복용주기": None,
                            "복용기간": None
                        })
                    break

            # 건강 상태
            m = re.search(r"(두통|머리\s*아픔|기침|재채기|콧물|피곤|어지럼|열|발열|복통|몸살)", t)
            if m:
                out.setdefault("user.건강상태", []).append({
                    "증상": m.group(1),
                    "정도": None,
                    "기간": None,
                    "기타": None
                })

            # ✅ 약물 패턴 (식사보다 우선)
            drug_patterns = [
                r"([가-힣A-Za-z]+약)(?:을|를)?\s*(?:먹었어|먹었어요|먹었습니다|먹음|드셨어|드셨어요|드셨습니다|드심|복용했어|복용했어요|복용했습니다|복용함)",
                r"(?:먹었어|먹었어요|먹었습니다|먹음|드셨어|드셨어요|드셨습니다|드심|복용했어|복용했어요|복용했습니다|복용함)\s*([가-힣A-Za-z]+약)",
                r"([가-힣A-Za-z]+약)(?:을|를)?\s*(?:먹어|먹어요|먹습니다|먹어야|먹어야해|먹어야해요|먹어야합니다|먹어야함|복용해|복용해요|복용합니다|복용해야|복용해야해|복용해야해요|복용해야합니다|복용해야함)",
                r"(?:먹어|먹어요|먹습니다|먹어야|먹어야해|먹어야해요|먹어야합니다|먹어야함|복용해|복용해요|복용합니다|복용해야|복용해야해|복용해야해요|복용해야합니다|복용해야함)\s*([가-힣A-Za-z]+약)"
            ]
            
            # 약물 패턴 먼저 체크 (LLM보다 우선)
            for pattern in drug_patterns:
                m = re.search(pattern, t)
                if m:
                    drug_name = m.group(1).strip()
                    print(f"[DEBUG] 약물 패턴 감지: '{drug_name}'")
                    return {"user.약물": [{"약명": drug_name, "복용일": "오늘"}]}
        
            # ✅ 식사 (시간 포함 + 자동 파싱 + 밥 처리) - 개선된 버전
            # 약 복용 관련 키워드가 있으면 식사로 처리하지 않음 (약 복용 체크를 먼저 해야 함)
            # ✅ "약속"은 일정이므로 약물로 처리하지 않음
            has_medicine_keyword = any(keyword in t for keyword in MEDICINE_KEYWORDS)
            # "약" 키워드 체크 (단, "약속"은 제외 - 약속은 일정)
            if not has_medicine_keyword and "약" in t and "약속" not in t:
                # "~약" 패턴 (혈압약, 감기약 등) 또는 "약 먹" 패턴 체크
                if re.search(r"[가-힣A-Za-z]+약|약\s*[먹드]", t):
                    has_medicine_keyword = True
            # "~약" 패턴 체크 (단, "약속"은 제외)
            if not has_medicine_keyword:
                if re.search(r"[가-힣A-Za-z]+약(?!속)", t):
                    has_medicine_keyword = True
        
            if "먹" in t and not has_medicine_keyword:
                # "안 먹었어", "굶었어" 패턴 체크 (skip 처리)
                skip_patterns = [
                    r"(아무것도|아무것)\s*안\s*먹",
                    r"안\s*먹었어",
                    r"굶었어",
                    r"먹지\s*않았어",
                    r"식사\s*안\s*했어"
                ]
                
                for skip_pattern in skip_patterns:
                    if re.search(skip_pattern, t):
                        # 식사 skip - 엔티티 생성하지 않음
                        break
                else:
                    # 새로운 식사 엔티티 추출 함수 사용
                    meal_data = self._extract_meal_entity(t)
                    if meal_data.get("끼니") or meal_data.get("메뉴"):
                        # 기존 패턴과 호환되도록 변환
                        # 날짜가 None이면 "오늘"로 기본값 설정
                        meal_date = meal_data.get("날짜") or "오늘"
                        meal_entity = {
                            "끼니": meal_data.get("끼니"),
                            "메뉴": meal_data.get("메뉴", []),
                            "날짜": meal_date,
                            "시간": None  # 시간은 별도 추출
                        }
                        
                        # 시간 자동 추출
                        extracted_time = self._extract_time_from_text(t)
                        if extracted_time:
                            meal_entity["시간"] = extracted_time
                        
                        out.setdefault("user.식사", []).append(meal_entity)
                        print(f"[DEBUG] 식사 엔티티 추출 (개선): {meal_entity}")
                        return out
                    else:
                        # 기존 방식으로 fallback
                        # 패턴 1: "오늘 점심 햄버거 먹었어" (메뉴 포함) - 조사 제외
                        m1 = re.search(r"(오늘|어제|내일)?\s*(아침|점심|저녁).*?([가-힣]+?)\s*(?:먹|먹어|먹었|먹었어|먹었어요|먹었습니다)", t)
                    # 패턴 1-1: "점심에는 그냥 초코 파이 먹었어" (에는 포함) - 더 유연한 패턴
                    m1_1 = re.search(r"(오늘|어제|내일)?\s*(아침|점심|저녁)에\s*([가-힣]+?)\s*(?:먹|먹어|먹었|먹었어|먹었어요|먹었습니다)", t)
                    # 패턴 1-2: "아침에는 떡만둣국먹어썽" 같은 경우를 위한 추가 패턴
                    m1_2 = re.search(r"(오늘|어제|내일)?\s*(아침|점심|저녁)에\s*([가-힣]+?)(?:먹어|먹었|먹었어|먹었어요|먹었습니다|먹어썽)", t)
                    # 패턴 2: "저녁 먹었어" (메뉴 없음) - 메뉴가 없는 경우만 매치
                    m2 = re.search(r"(오늘|어제|내일)?\s*(아침|점심|저녁)\s*먹", t)
                    # 패턴 2-1: "점심에는 먹었어" (에는 포함, 메뉴 없음)
                    m2_1 = re.search(r"(오늘|어제|내일)?\s*(아침|점심|저녁)에\s*먹", t)
                    # 패턴 3: "밥 먹었어" (끼니 없음, 메뉴만)
                    m3 = re.search(r"([가-힣]+)\s*먹", t)
                    # 패턴 3-1: "떡만둣국 먹었고, 7시반에 먹었어" (메뉴 + 시간)
                    m3_1 = re.search(r"([가-힣]+)\s*먹었고,?\s*(\d{1,2}시\s*반?)\s*에?\s*먹", t)
            
                    if m3_1:
                        # 패턴 3-1: "떡만둣국 먹었고, 7시반에 먹었어"
                        # print(f"[DEBUG] 패턴 3-1 매칭: 메뉴={m3_1.group(1)}, 시간={m3_1.group(2)}")
                        menu_item = m3_1.group(1).strip()
                        extracted_time = m3_1.group(2).strip()
                        
                        # "약" 등은 식사 메뉴에서 제외 (밥은 유효한 메뉴)
                        # "식후에", "식전에" 같은 약 복용 관련 표현도 제외
                        if menu_item not in {"음식", "뭔가", "뭐", "약", "약물", "약품", "식후에", "식전에", "식후", "식전"}: 
                            # 시간대 기반으로 끼니 추론
                            inferred_meal = None
                            if extracted_time:
                                hour = self._extract_hour_from_time(extracted_time)
                                if hour and 6 <= hour < 11:
                                    inferred_meal = "아침"
                                elif hour and 11 <= hour < 15:
                                    inferred_meal = "점심"
                                elif hour and 15 <= hour < 22:
                                    inferred_meal = "저녁"
                            
                            # 시간으로 추론 안되면 끼니를 null로 설정 (추측하지 않음)
                            if not inferred_meal:
                                inferred_meal = None
                        
                            meal_entity = {
                                "끼니": inferred_meal, "메뉴": [menu_item], "날짜": "오늘", "시간": extracted_time
                            }
                            # print(f"[DEBUG] 패턴 3-1 식사 엔티티 생성: {meal_entity}")
                            out.setdefault("user.식사", []).append(meal_entity)
                    elif m2 or m2_1:
                        # 패턴 2: "저녁 먹었어" (메뉴 없음) - 시간 정보만 있는 경우
                        if m2_1:
                            rel_date, meal = m2_1.group(1), m2_1.group(2)
                        else:
                            rel_date, meal = m2.group(1), m2.group(2)
                        
                        # 시간 자동 추출
                        extracted_time = self._extract_time_from_text(t)
                        
                        # 이전 대화에서 시간 정보가 있었는지 확인
                        if not extracted_time and session_id in self.time_context:
                            context = self.time_context[session_id]
                            if context.get("last_meal") == meal and context.get("last_time"):
                                extracted_time = context["last_time"]
                                logger.debug(f"이전 컨텍스트에서 시간 정보 연결: {meal} {extracted_time}")
                        
                        # 시간 컨텍스트 업데이트
                        if extracted_time:
                            if session_id not in self.time_context:
                                self.time_context[session_id] = {}
                            self.time_context[session_id]["last_time"] = extracted_time
                            self.time_context[session_id]["last_meal"] = meal
                            self.time_context[session_id]["last_menu"] = None  # 메뉴는 비어있음
                        
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
                        
                        # 메뉴 추출 시 불용어 필터링 강화 (복합 메뉴명 보존)
                        # 먼저 복합 메뉴명을 보존하기 위해 특별한 패턴 처리
                        menu_raw_processed = menu_raw
                        
                        # 복합 메뉴명 패턴 보존 (예: "샤인 머스켓", "치킨 버거", "피자 슬라이스" 등)
                        complex_menu_patterns = [
                            r"샤인\s*머스켓", r"치킨\s*버거", r"피자\s*슬라이스", r"햄\s*버거", r"치즈\s*버거",
                            r"김치\s*찌개", r"된장\s*찌개", r"미역\s*국", r"계란\s*말이", r"김치\s*전"
                        ]
                        
                        for pattern in complex_menu_patterns:
                            menu_raw_processed = re.sub(pattern, lambda m: m.group(0).replace(" ", "_"), menu_raw_processed)
                        
                        # 이제 분할 (공백은 제외하고 구분자만 사용)
                        menu_candidates = [x.strip().replace("_", " ") for x in re.split(r"[,와과랑및]", menu_raw_processed) if x.strip()]
                        
                        # 메뉴 불용어 확장
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
                                    not re.match(r'^[가-힣]{1}$', x))]  # 한 글자 단어 제외
                        
                        # 시간 자동 추출 (이전 컨텍스트 활용)
                        extracted_time = self._extract_time_from_text(t)
                        
                        # 이전 대화에서 시간 정보가 있었는지 확인
                        if not extracted_time and session_id in self.time_context:
                            context = self.time_context[session_id]
                            if context.get("last_meal") == meal and context.get("last_time"):
                                extracted_time = context["last_time"]
                                logger.debug(f"이전 컨텍스트에서 시간 정보 연결: {meal} {extracted_time}")
                        
                        # 시간 컨텍스트 업데이트
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
                        
                        # 시간 정보가 포함된 경우 메뉴로 처리하지 않음
                        extracted_time = self._extract_time_from_text(t)
                        if extracted_time and menu_item in {"시에", "시", "분", "오전", "오후", "새벽", "밤"}:
                            # 시간 정보만 있는 경우 - 메뉴 없음으로 처리
                            # 이전 컨텍스트에서 끼니 정보 가져오기
                            if session_id in self.time_context:
                                context = self.time_context[session_id]
                                if context.get("last_meal"):
                                    meal = context["last_meal"]
                                    out.setdefault("user.식사", []).append({
                                        "끼니": meal, "메뉴": [], "날짜": "오늘", "시간": extracted_time
                                    })
                                    # 시간 컨텍스트 업데이트
                                    self.time_context[session_id]["last_time"] = extracted_time
                                    return out
                        
                        # "약" 등은 식사 메뉴에서 제외 (밥은 유효한 메뉴)
                        # "식후에", "식전에" 같은 약 복용 관련 표현도 제외
                        if menu_item not in {"음식", "뭔가", "뭐", "약", "약물", "약품", "시에", "시", "분", "오전", "오후", "새벽", "밤", "식후에", "식전에", "식후", "식전"}: 
                            extracted_time = self._extract_time_from_text(t)
                            
                            # 시간대 기반으로 끼니 추론
                            inferred_meal = None
                            if extracted_time:
                                hour = self._extract_hour_from_time(extracted_time)
                                if hour and 6 <= hour < 11:
                                    inferred_meal = "아침"
                                elif hour and 11 <= hour < 15:
                                    inferred_meal = "점심"
                                elif hour and 15 <= hour < 22:
                                    inferred_meal = "저녁"
                            
                            # 시간으로 추론 안되면 끼니를 null로 설정 (추측하지 않음)
                            if not inferred_meal:
                                inferred_meal = None
                            
                            out.setdefault("user.식사", []).append({
                                "끼니": inferred_meal, "메뉴": [menu_item], "날짜": "오늘", "시간": extracted_time
                            })

            # ✅ 물건 추출 (명령 패턴 포함)
            # ✅ 일정 엔티티가 이미 있으면 물건 추출 건너뛰기 (일정 우선)
            if "user.물건" not in out and "user.일정" not in out:
                try:
                    # 1. 위치 포함 케이스 (기존)
                    item_location_result = self._extract_item_location_rule(t)
                    if item_location_result:
                        out.update(item_location_result)
                    else:
                        # 2. 명령 패턴 케이스 (새로 추가)
                        item_command_result = self._extract_item_command_rule(t)
                        if item_command_result:
                            out.update(item_command_result)
                except Exception as e:
                    print(f"[WARN] 물건 추출 중 오류: {e}")

        except Exception as e:
            print(f"[ERROR] _rule_based_extract 전체 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            return out  # 오류 발생 시 빈 딕셔너리 반환

        # ✅ 일정 엔티티가 저장되었으면 물건 엔티티 제거 (최종 확인)
        if "user.일정" in out and "user.물건" in out:
            print(f"[DEBUG] [FINAL CHECK] 일정 엔티티 존재 → 물건 엔티티 제거: {out.get('user.물건')}")
            out.pop("user.물건", None)
        
        return out

    def _extract_entities_with_llm(self, user_input: str, session_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """LLM 기반 엔티티 추출 (문맥 이해 기반)"""
        try:
            # LLM에게 문맥을 이해해서 엔티티를 추출하도록 요청
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
            
            # JSON 파싱 (```json 마크다운 제거)
            import json
            import re
            
            # ```json과 ``` 제거
            json_text = re.sub(r'```json\s*', '', result_text)
            json_text = re.sub(r'```\s*$', '', json_text).strip()
            
            try:
                entities = json.loads(json_text)
                print(f"[DEBUG] LLM 엔티티 추출 결과: {entities}")
                
                # user. 접두사 추가하여 기존 형식과 호환
                formatted_entities = {}
                for entity_type, entity_list in entities.items():
                    if entity_list:  # 빈 배열이 아닌 경우만
                        formatted_entities[f"user.{entity_type}"] = entity_list
                
                return formatted_entities
                
            except json.JSONDecodeError as e:
                print(f"[DEBUG] LLM JSON 파싱 실패: {e}")
                return {}
                
        except Exception as e:
            print(f"[DEBUG] LLM 엔티티 추출 실패: {e}")
            return {}

    # (사전) 엔티티 추출: LLM → Rule → Merge 순서
    def _pre_extract_entities(self, user_input: str, session_id: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """엔티티 추출: LLM 우선 → Rule 보완 + Slot-filling 체크 (문맥 이해 기반)"""
        merged: Dict[str, List[Dict[str, Any]]] = {}

        # 1️⃣ 구조화된 엔티티 추출 LLM 체인 사용 (약, 일정, 식사 등 모든 엔티티)
        print(f"[DEBUG] LLM 엔티티 추출 시작 (entity_chain): '{user_input}'")
        structured_llm_out = {}
        try:
            if hasattr(self, 'entity_chain') and self.entity_chain:
                result = self.entity_chain.invoke({"utterance": user_input})
                print(f"[DEBUG] entity_chain 원본 응답: {result}")
                
                # 결과를 기존 형식으로 변환 (user. 접두사 추가)
                if isinstance(result, dict):
                    for key, value in result.items():
                        if key.startswith("user."):
                            structured_llm_out[key] = value if isinstance(value, list) else [value]
                        else:
                            # 약, 일정 등은 user. 접두사 추가
                            entity_key = f"user.{key}" if key in ["약", "일정", "식사", "가족", "물건", "기념일", "취미", "취향", "건강상태"] else f"user.{key}"
                            structured_llm_out[entity_key] = value if isinstance(value, list) else [value]
                    
                    print(f"[DEBUG] entity_chain 변환된 결과: {structured_llm_out}")
        except Exception as e:
            print(f"[WARN] entity_chain 추출 실패: {e}")
            import traceback
            traceback.print_exc()
        
        # 2️⃣ 사용자 정보 전용 LLM 추출 (사용자 정보만 필요할 때)
        user_info_llm_out = {}
        try:
            user_info_llm_out = self._extract_entities_with_llm(user_input, session_id)
            print(f"[DEBUG] 사용자 정보 LLM 추출 결과: {user_info_llm_out}")
        except Exception as e:
            print(f"[WARN] 사용자 정보 LLM 추출 실패: {e}")
        
        # 3️⃣ 결과 병합: 구조화된 엔티티 우선, 사용자 정보는 보완
        if structured_llm_out:
            merged = structured_llm_out.copy()
            # 사용자 정보가 있으면 추가
            if user_info_llm_out.get("user.사용자"):
                merged["user.사용자"] = user_info_llm_out["user.사용자"]
        elif user_info_llm_out:
            merged = user_info_llm_out.copy()
        
        # 4️⃣ LLM 결과가 없거나 부족하면 rule-based 보완
        if not merged or (not structured_llm_out and not user_info_llm_out):
            print(f"[DEBUG] LLM 결과 없음 또는 부족 → rule-based fallback 시도")
            rule_out = self._rule_based_extract(user_input, session_id)
            print(f"[DEBUG] _pre_extract_entities rule_out (fallback): {rule_out}")
            
            # LLM 결과와 rule-based 결과 병합 (LLM 결과가 있으면 우선, 없으면 rule-based만)
            if merged:
                # LLM 결과와 rule-based 결과 병합 (중복 제거)
                for key, value in rule_out.items():
                    if key not in merged:
                        merged[key] = value
                    else:
                        # 중복 제거: 기존 엔티티와 새 엔티티 비교
                        existing = merged[key]
                        new_items = [item for item in value if item not in existing]
                        if new_items:
                            merged[key].extend(new_items)
            else:
                merged = rule_out.copy()  # rule-based만 사용
        
        # 재질문 체크
        for entity_key, entity_list in merged.items():
            for entity in entity_list:
                if isinstance(entity, dict) and "질문" in entity:
                    print(f"[DEBUG] _pre_extract_entities에서 재질문 발견: {entity['질문']}")
                    return {entity_key: [entity]}
        
        # 1.5️⃣ Slot-filling 체크 (엔티티 추출 단계에서)
        for entity_key, entity_list in merged.items():
            if entity_list:  # 엔티티가 있는 경우만 체크
                for entity in entity_list:
                    if isinstance(entity, dict):
                        # 엔티티 타입 추출 (user.약 → 약)
                        entity_type = entity_key.replace("user.", "")
                        
                        # 필수 필드 확인
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
        
        # 약물 패턴 우선 처리 (LLM 결과보다 우선)
        if "user.약물" in merged:
            print(f"[DEBUG] 약물 패턴 우선 처리: {merged['user.약물']}")
            return merged
        
        # LLM 결과 반환
        return merged

    def _dedup_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """엔티티 중복 제거"""
        seen = set()
        unique = []
        for e in entities:
            # 딕셔너리를 정렬된 튜플로 변환하여 중복 체크 (리스트는 문자열로 변환)
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
        """약 엔티티 중복 제거 및 병합 (약명 기준, 복용 정보 포함)"""
        drug_dict = {}
        for e in entities:
            drug_name = e.get("약명", "")
            if not drug_name:
                continue
                
            if drug_name not in drug_dict:
                drug_dict[drug_name] = e
            else:
                # 기존 엔티티와 새 엔티티 병합
                existing = drug_dict[drug_name]
                merged = {"약명": drug_name}
                
                # 복용 정보 병합
                if existing.get("복용") or e.get("복용"):
                    merged["복용"] = existing.get("복용", []) + e.get("복용", [])
                
                # 식사와의 관계 병합
                if existing.get("식사와의 관계") or e.get("식사와의 관계"):
                    merged["식사와의 관계"] = e.get("식사와의 관계") or existing.get("식사와의 관계")
                
                # 복용 기간 병합
                if existing.get("복용 기간") or e.get("복용 기간"):
                    merged["복용 기간"] = e.get("복용 기간") or existing.get("복용 기간")
                
                # 기타 필드 병합
                for key, value in existing.items():
                    if key not in merged and value:
                        merged[key] = value
                for key, value in e.items():
                    if key not in merged and value:
                        merged[key] = value
                
                drug_dict[drug_name] = merged
        
        return list(drug_dict.values())

    def _is_complete_entity(self, entity_key: str, entity: dict) -> bool:
        """엔티티가 완전한지 확인 (필수 필드가 모두 있는지)"""
        required_fields = {
            "user.사용자": ["이름"],
            "user.약": ["약명"],
            "user.일정": ["제목", "날짜"],
            "user.기념일": ["제목", "날짜"],
            "user.가족": ["관계"],
            "user.건강상태": ["증상"],
            "user.물건": ["이름"],
            "user.식사": ["끼니"],  # 끼니만 필수로 변경
            "user.취미": ["이름"],
            "user.취향": ["값"]
        }
        
        required = required_fields.get(entity_key, [])
        for field in required:
            if not entity.get(field) or entity.get(field) == "" or entity.get(field) is None:
                return False
        return True


    def _generate_followup_questions(self, entity_key: str, missing_fields: List[str], value: dict = None) -> List[str]:
        """누락된 필드에 대한 재질문 생성 (중복 방지 강화)"""
        questions = []
        
        for field in missing_fields:
            # 이미 값이 있으면 질문 스킵
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
        """Follow-up 질문들을 자연스럽게 통합"""
        if not questions:
            return ""
        
        if len(questions) == 1:
            return questions[0]
        
        # 엔티티별로 질문 그룹화
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
        
        # 각 엔티티별로 자연스러운 문장으로 통합
        consolidated = []
        for entity, qs in entity_questions.items():
            if entity == "약":
                if len(qs) == 1:
                    consolidated.append(qs[0])
                else:
                    # 여러 질문이 있는 경우에만 통합 (추가 정보 요청 제거)
                    if qs:
                        consolidated.extend(qs)  # 개별 질문들을 그대로 추가
            elif entity == "식사":
                if len(qs) == 1:
                    consolidated.append(qs[0])
                else:
                    # 끼니, 메뉴, 시간이 모두 빠진 경우
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
        """확정된 사용자 이름 가져오기 (별칭 지원) - 모든 세션에서 조회"""
        try:
            # VectorStore 전체에서 user.사용자 필터링 (모든 세션에서 조회)
            docs = self.vectorstore.get()
            for i, doc_id in enumerate(docs.get("ids", [])):
                # user.사용자 패턴이 포함된 doc_id만 선택 (모든 세션)
                if "_user.사용자_" in doc_id:
                    data = json.loads(docs["documents"][i])
                    # 확인된 사용자 이름이 있으면 반환
                    if (data.get("이름") and data.get("확인됨")):
                        # 별칭이 있으면 별칭 반환, 없으면 이름 반환
                        if data.get("별칭"):
                            return data["별칭"]
                        return data["이름"]
        except Exception as e:
            print(f"[WARN] 사용자 이름 조회 실패: {e}")
        
        return "사용자"

    def _analyze_entity_context(self, user_input: str, existing_entity: dict, new_entity: dict, entity_key: str) -> dict:
        """LLM을 활용한 문맥 분석으로 같은 대상인지 판단"""
        try:
            # 문맥 분석을 위한 프롬프트
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
            # 분석 실패 시 기본적으로 같은 대상으로 간주
            return {"is_same_entity": True, "reason": "분석 실패로 인한 기본값"}

    def _find_item_location(self, user_input: str, session_id: str) -> dict:
        """물건 위치 검색 (명령/질문 처리용) - dict 반환"""
        try:
            # 사용자 입력에서 물건명 추출 (더 유연한 방식)
            item_keywords = []
            
            # 물건 관련 질문 패턴에서 물건명 추출
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
            
            # VectorStore에서 물건 검색 (필터 없이)
            docs = self.vectorstore.similarity_search(" ".join(item_keywords), k=20)
            
            if not docs:
                return None
            
            # 위치 정보가 있는 물건 찾기 (세션 필터링)
            for doc in docs:
                try:
                    data = json.loads(doc.page_content)
                    
                    # 엔티티 키 확인 (모든 세션에서 검색)
                    if data.get("entity_key") == "user.물건":
                        item_name = data.get("이름", "")
                        location = data.get("위치", "")
                        
                        if location and any(keyword in item_name for keyword in item_keywords):
                            # 안전성을 위해 normalize 재적용 (이전 데이터 대비)
                            normalized_location = self._normalize_location(location)
                            return {"이름": item_name, "위치": normalized_location}
                except Exception:
                    continue
            
            return None
            
        except Exception as e:
            print(f"[WARN] 물건 위치 검색 실패: {e}")
            return None

    def _handle_location_query(self, user_input: str, session_id: str) -> str:
        """위치 질문 처리 (cognitive 카테고리용)"""
        location_info = self._find_item_location(user_input, session_id)
        if location_info:
            return f"{location_info['이름']}은 {location_info['위치']}에 있어요."
        else:
            return "죄송해요, 그 물건의 위치를 모르겠어요. 어디에 두었는지 알려주시면 기억해둘게요!"


    def _build_personalized_emotional_reply(self, user_input: str, session_id: str) -> str:
        """개인화된 감정 응답 생성 (엔티티 정보 활용)"""
        try:
            # 사용자 이름과 VectorStore 기반 사실 정보 가져오기
            user_name = self._get_confirmed_user_name(session_id)
            facts_text = self._get_facts_text(session_id)
            
            # 개인화된 프롬프트 생성
            context_info = []
            if user_name:
                context_info.append(f"사용자의 이름은 {user_name}입니다.")
            if facts_text:
                context_info.append(f"저장된 정보: {facts_text}")
            
            context_text = "\n".join(context_info) if context_info else ""
            
            # 개인화된 감정 응답 생성
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
                # 기본 감정 응답 - 사용자 이름 확인 상태 전달
                user_name_confirmed = bool(self._get_confirmed_user_name(session_id))
                return build_emotional_reply(user_input, llm=self.llm, user_name_confirmed=user_name_confirmed)
            
            response = self.llm.invoke(prompt)
            return response
            
        except Exception as e:
            print(f"[WARN] 개인화된 감정 응답 생성 실패: {e}")
            # Fallback: 기본 감정 응답 - 사용자 이름 확인 상태 전달
            user_name_confirmed = bool(self._get_confirmed_user_name(session_id))
            return build_emotional_reply(user_input, llm=self.llm, user_name_confirmed=user_name_confirmed)

    def _prevent_name_family_conflict(self, entities: Dict[str, List[Dict[str, Any]]]) -> None:
        """이름/가족 정보 충돌 방지"""
        # 가족 정보에서 이름 추출
        family_names = set()
        for family_entity in entities.get("user.가족", []):
            if "이름" in family_entity:
                family_names.add(family_entity["이름"])
        
        # 사용자 이름이 가족 이름과 충돌하는지 확인
        for user_entity in entities.get("user.사용자", []):
            if "이름" in user_entity:
                user_name = user_entity["이름"]
                if user_name in family_names:
                    # 충돌 시 사용자 이름 제거 (가족 이름이 우선)
                    # print(f"[INFO] 이름 충돌 방지: '{user_name}'은 가족 이름으로 사용됨")  # 테스트 시 로그 제거
                    user_entity.pop("이름", None)

    def _normalize_name(self, name: str) -> str:
        """이름 후보 정규화 (공백 제거, 조사 제거)"""
        if not name: 
            return name
        
        # 공백 제거 (김 철수 → 김철수)
        name = name.strip().replace(" ", "")
        
        # 인용 패턴 제거 (이라고 해, 라고 해, 라 해 등)
        name = re.sub(r"(이라고\s*해|라고\s*해|라\s*해|이라고\s*불러|라고\s*불러|라\s*불러)$", "", name).strip()
        
        # 종결어 제거 (이야, 이에요, 입니다, 예요, 이고, 이고요 등)
        name = re.sub(r"(이야|이에요|입니다|예요|이고|이고요|야|다|어|아)$", "", name).strip()
        
        # 호칭 제거 (님, 씨)
        name = re.sub(r"(님|씨)$", "", name).strip()
        
        return name

    def _normalize_location(self, location: str) -> str:
        """위치에서 조사 제거하고 변형어 정규화하여 순수 명사구만 반환 (상대 위치 보존)"""
        if not location:
            return location
        
        # 조사 패턴 제거 (에, 에서, 으로, 쪽에, 안에 등) - + 붙여서 중복 제거
        location = re.sub(r"(에|에서|으로|쪽에|안에|밖에|옆에|앞에|뒤에|아래에)+$", "", location).strip()
        
        # 상대 위치 패턴 추출 (위/아래/옆 등)
        relative_position_pattern = r"(.+?)(위|아래|옆|앞|뒤|왼쪽|오른쪽|가운데|중앙|중간)$"
        m = re.match(relative_position_pattern, location)
        
        if m:
            base_location = m.group(1).strip()
            relative_pos = m.group(2)
            
            # 기본 위치 정규화
            base_normalized = self._normalize_base_location(base_location)
            
            # 상대 위치 보존하여 반환
            return f"{base_normalized}({relative_pos})"
        
        # 상대 위치가 없는 경우 일반 정규화
        return self._normalize_base_location(location)
    
    def _normalize_base_location(self, location: str) -> str:
        """기본 위치 정규화 (상대 위치 제외)"""
        # 변형어 정규화 사전
        location_normalize_map = {
            # 방/공간 관련
            "거실": "거실", "방": "거실", "응접실": "거실", "라운지": "거실",
            "침실": "침실", "자기방": "침실", "개인방": "침실",
            "부엌": "부엌", "주방": "부엌", "요리실": "부엌",
            "화장실": "화장실", "욕실": "화장실", "세면실": "화장실",
            "다용도실": "다용도실", "서재": "다용도실", "작업실": "다용도실",
            "베란다": "베란다", "발코니": "베란다", "테라스": "베란다",
            "지하": "지하", "지하실": "지하", "지하층": "지하",
            "옥상": "옥상", "루프탑": "옥상",
            
            # 가구/물건 관련
            "화장지": "화장지", "휴지": "화장지", "두루마리": "화장지",
            "냉장고": "냉장고", "냉동고": "냉장고",
            "책상": "책상", "데스크": "책상", "작업대": "책상",
            "침대": "침대", "베드": "침대", "매트리스": "침대",
            "소파": "소파", "쇼파": "소파", "의자": "소파",
            "테이블": "테이블", "탁자": "테이블", "식탁": "테이블",
            "옷장": "옷장", "장롱": "옷장", "드레스룸": "옷장",
            "서랍": "서랍", "서랍장": "서랍", "수납함": "서랍",
            "선반": "선반", "책장": "선반", "수납장": "선반",
            
            # 방향/위치 관련 (상대 위치가 아닌 경우)
            "앞": "앞", "앞쪽": "앞", "정면": "앞",
            "뒤": "뒤", "뒤쪽": "뒤", "후면": "뒤",
            "왼쪽": "왼쪽", "좌측": "왼쪽", "왼편": "왼쪽",
            "오른쪽": "오른쪽", "우측": "오른쪽", "오른편": "오른쪽",
            "위": "위", "위쪽": "위", "상단": "위",
            "아래": "아래", "아래쪽": "아래", "하단": "아래",
            "가운데": "가운데", "중앙": "가운데", "중간": "가운데",
            "옆": "옆", "옆쪽": "옆", "측면": "옆",
            
            # 일반적인 위치
            "여기": "여기", "이곳": "여기", "현재위치": "여기",
            "저기": "저기", "그곳": "저기", "저쪽": "저기",
            "어디": "어디", "어디선가": "어디", "어딘가": "어디",
        }
        
        # 변형어 정규화 적용
        return location_normalize_map.get(location, location)

    def _merge_entity_values(self, old_value: dict, new_value: dict, entity_key: str) -> dict:
        """중복 엔티티 머지 로직 개선 (기존 값 유지 + 새 값 추가)"""
        if not old_value:
            return new_value
        if not new_value:
            return old_value
        
        # 기본 머지: 기존 값 유지 + 새 값 추가
        merged = {**old_value, **new_value}
        
        # 엔티티별 특수 머지 로직
        if entity_key.endswith("사용자"):
            # 사용자: 이름은 새 값 우선, 별칭은 누적
            if "별칭" in new_value and "별칭" in old_value:
                # 별칭이 다르면 둘 다 유지 (리스트로)
                if old_value["별칭"] != new_value["별칭"]:
                    merged["별칭"] = [old_value["별칭"], new_value["별칭"]]
                else:
                    merged["별칭"] = new_value["별칭"]
            elif "별칭" in old_value and "별칭" not in new_value:
                merged["별칭"] = old_value["별칭"]
            elif "별칭" in new_value and "별칭" not in old_value:
                merged["별칭"] = new_value["별칭"]
        
        elif entity_key.endswith("약"):
            # 약: 복용 정보는 누적, 기간은 새 값 우선
            if "복용" in old_value and "복용" in new_value:
                merged["복용"] = (old_value["복용"] or []) + (new_value["복용"] or [])
            elif "복용" in old_value:
                merged["복용"] = old_value["복용"]
            elif "복용" in new_value:
                merged["복용"] = new_value["복용"]
        
        elif entity_key.endswith("일정"):
            # 일정: 시간 정보는 누적
            if "시간" in old_value and "시간" in new_value:
                old_time = self._normalize_time_field(old_value["시간"])
                new_time = self._normalize_time_field(new_value["시간"])
                merged["시간"] = sorted(list(set(old_time + new_time))) if (old_time or new_time) else None
            elif "시간" in old_value:
                merged["시간"] = old_value["시간"]
            elif "시간" in new_value:
                merged["시간"] = new_value["시간"]
        
        elif entity_key.endswith("식사"):
            # 식사: 메뉴는 누적
            if "메뉴" in old_value and "메뉴" in new_value:
                old_menus = old_value["메뉴"] if isinstance(old_value["메뉴"], list) else [old_value["메뉴"]]
                new_menus = new_value["메뉴"] if isinstance(new_value["메뉴"], list) else [new_value["메뉴"]]
                merged["메뉴"] = list(set(old_menus + new_menus))
            elif "메뉴" in old_value:
                merged["메뉴"] = old_value["메뉴"]
            elif "메뉴" in new_value:
                merged["메뉴"] = new_value["메뉴"]
        
        elif entity_key.endswith("물건"):
            # 물건: 위치는 새 값 우선, 설명은 누적
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
        """복용 기간 추출 통합 함수"""
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
        """약 복용 시 식전/식후 정보 추출"""
        if "식후" in text:
            return "식후"
        if "식전" in text:
            return "식전"
        return None

    def _extract_drugs_with_info(self, text: str) -> List[Dict[str, Any]]:
        """여러 약 복용 정보를 약별로 구분하여 추출"""
        results = []
        seen_drugs = set()  # 중복 방지
        
        # 약이 아닌 단어들 (약으로 끝나지만 의약품이 아닌 것들)
        non_drug_words = {
            "예약", "약속", "약속시간", "약속장소", "약속일",
            "치약", "세정약", "세정제", "세정액", "세정용품",
            "약속", "약속시간", "약속장소", "약속일", "약속시간", "약속장소"
        }
        
        # 먼저 모든 약명을 찾기
        drug_patterns = [
            r"([가-힣A-Za-z]+약)",  # 기존 패턴
            r"(아스피린|타이레놀|이부프로펜|아세트아미노펜)",  # 일반 약명
        ]
        
        all_drugs = []
        for pattern in drug_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                if match not in non_drug_words:
                    all_drugs.append(match)
        
        # 각 약에 대해 개별적으로 복용 정보 추출
        for drug in all_drugs:
            # 중복 방지
            if drug in seen_drugs:
                continue
            seen_drugs.add(drug)
            
            drug_info = {"약명": drug}
            
            # 해당 약명이 포함된 문장을 찾기 (마침표, 쉼표로 분리)
            sentences = re.split(r'[.,]', text)
            
            for sentence in sentences:
                if drug in sentence:
                    sentence = sentence.strip()
                    if not sentence:
                        continue
                    
                    # 복용 주기 추출
                    dosages = self._extract_dosage(sentence)
                    if dosages:
                        drug_info["복용"] = dosages

                    # 식전/식후 정보 추출
                    meal_relation = self._extract_meal_relation(sentence)
                    if meal_relation:
                        drug_info["식사와의 관계"] = meal_relation

                    # 복용 기간 추출
                    period = self._extract_period(sentence)
                    if period:
                        drug_info["복용 기간"] = period
                    
                    break  # 해당 약의 문장을 찾았으면 중단

            results.append(drug_info)

        return results

    def _extract_dosage(self, text: str) -> List[Dict[str, str]]:
        """복용 횟수/방법 추출 통합 함수 (실제 언급된 정보만 추출)"""
        dosage_patterns = [
            r"(하루\s*에?\s*\d+번|\d+번\s*복용)",
            r"(아침|점심|저녁)",
            r"(\d+시\s*\d+분?)",
        ]
        
        dosages = []
        seen_dosages = set()  # 중복 방지를 위한 set
        
        for pattern in dosage_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                # 정규화된 텍스트로 중복 체크
                normalized_match = re.sub(r'\s+', ' ', match.strip())
                if normalized_match not in seen_dosages:
                    seen_dosages.add(normalized_match)
                    dosages.append({"원문": normalized_match})
        
        # 실제로 언급된 복용 정보만 반환 (추측하지 않음)
        return dosages if dosages else None

    def _add_to_date_cache(self, date_str: str, normalized_date: str) -> None:
        """날짜 캐시에 추가 (크기 제한 적용)"""
        self.date_cache[date_str] = normalized_date
        
        # 캐시 크기 제한 적용
        if len(self.date_cache) > self.max_date_cache_size:
            # 가장 오래된 항목 제거 (FIFO)
            oldest_key = next(iter(self.date_cache))
            del self.date_cache[oldest_key]
            logger.debug(f"날짜 캐시 크기 제한으로 오래된 항목 제거: '{oldest_key}'")

    def _normalize_date(self, date_str: str, session_id: str = None) -> str:
        """하이브리드 날짜 정규화: Rule-based + 캐시 + dateparser + LLM fallback"""
        if not date_str:
            return date_str
        
        # 0차: 캐시 확인 (가장 빠른 처리)
        if date_str in self.date_cache:
            logger.debug(f"날짜 캐시 hit: '{date_str}' → '{self.date_cache[date_str]}'")
            return self.date_cache[date_str]
        
        now = datetime.now()
        
        # 1차: Rule-based 빠른 매핑 (성능 최적화)
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
        
        # 복잡한 날짜 표현 처리
        if "다음 주" in date_str or "다음주" in date_str:
            # "다음 주 금요일" 처리
            if "금요일" in date_str:
                # 다음 주 금요일 계산
                days_ahead = 4 - now.weekday()  # 금요일은 4
                if days_ahead <= 0:  # 금요일이 지났으면
                    days_ahead += 7
                days_ahead += 7  # 다음 주
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "월요일" in date_str:
                days_ahead = 0 - now.weekday()  # 월요일은 0
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "화요일" in date_str:
                days_ahead = 1 - now.weekday()  # 화요일은 1
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "수요일" in date_str:
                days_ahead = 2 - now.weekday()  # 수요일은 2
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "목요일" in date_str:
                days_ahead = 3 - now.weekday()  # 목요일은 3
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "토요일" in date_str:
                days_ahead = 5 - now.weekday()  # 토요일은 5
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
            elif "일요일" in date_str:
                days_ahead = 6 - now.weekday()  # 일요일은 6
                if days_ahead <= 0:
                    days_ahead += 7
                days_ahead += 7
                return (now + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
        
        # 기본 상대 날짜 매핑
        if date_str in relative_dates:
            result = relative_dates[date_str].strftime('%Y-%m-%d')
            self._add_to_date_cache(date_str, result)  # 캐시에 저장
            return result
        
        # 1.5차: 요일 패턴 직접 처리 (성능 최적화)
        weekday_map = {
            "월요일": 0, "화요일": 1, "수요일": 2, "목요일": 3, 
            "금요일": 4, "토요일": 5, "일요일": 6
        }
        
        # 이번주/다음주/다다음주 + 요일 패턴
        weekday_pattern = r"(이번주|다음주|다다음주)\s*(월요일|화요일|수요일|목요일|금요일|토요일|일요일)"
        m = re.match(weekday_pattern, date_str.strip())
        if m:
            week_type = m.group(1)
            weekday_name = m.group(2)
            
            # 기준 날짜 설정
            if week_type == "이번주":
                base = now
            elif week_type == "다음주":
                base = now + timedelta(weeks=1)
            else:  # 다다음주
                base = now + timedelta(weeks=2)
            
            target_weekday = weekday_map[weekday_name]
            
            # 목표 요일까지의 일수 계산
            days_ahead = (target_weekday - base.weekday()) % 7
            target_date = base + timedelta(days=days_ahead)
            
            result = target_date.strftime('%Y-%m-%d')
            self._add_to_date_cache(date_str, result)  # 캐시에 저장
            return result
        
        # 1.5.1차: 단일 요일 패턴: "금요일" (현재 주 기준)
        single_weekday_pattern = r"(월요일|화요일|수요일|목요일|금요일|토요일|일요일)$"
        m = re.match(single_weekday_pattern, date_str.strip())
        if m:
            weekday_name = m.group(1)
            target_weekday = weekday_map[weekday_name]
            
            # 현재 주의 해당 요일 계산
            days_ahead = (target_weekday - now.weekday()) % 7
            target_date = now + timedelta(days=days_ahead)
            
            result = target_date.strftime('%Y-%m-%d')
            self._add_to_date_cache(date_str, result)  # 캐시에 저장
            return result
        
        # 1.6차: 월/일 패턴 직접 처리 (LLM 호출 최적화)
        # 기본 월/일 패턴: "10월 3일"
        month_day_pattern = r"(\d{1,2})월\s*(\d{1,2})일"
        m = re.match(month_day_pattern, date_str.strip())
        if m:
            month = int(m.group(1))
            day = int(m.group(2))
            
            # 올해 기준으로 날짜 생성
            try:
                target_date = datetime(now.year, month, day)
                # 이미 지난 날짜면 내년으로
                if target_date < now:
                    target_date = datetime(now.year + 1, month, day)
                result = target_date.strftime('%Y-%m-%d')
                self._add_to_date_cache(date_str, result)  # 캐시에 저장
                return result
            except ValueError:
                pass  # 잘못된 날짜면 다음 단계로
        
        # 1.6.1차: 연도 포함 월/일 패턴: "올해 12월 25일", "내년 5월 10일"
        year_month_day_pattern = r"(올해|내년)?\s*(\d{1,2})월\s*(\d{1,2})일"
        m = re.match(year_month_day_pattern, date_str.strip())
        if m:
            year_type = m.group(1)
            month = int(m.group(2))
            day = int(m.group(3))
            
            try:
                if year_type == "내년":
                    target_date = datetime(now.year + 1, month, day)
                else:  # "올해" 또는 생략 (올해 기준)
                    target_date = datetime(now.year, month, day)
                    # 이미 지난 날짜면 내년으로
                    if target_date < now:
                        target_date = datetime(now.year + 1, month, day)
                
                result = target_date.strftime('%Y-%m-%d')
                self._add_to_date_cache(date_str, result)  # 캐시에 저장
                return result
            except ValueError:
                pass  # 잘못된 날짜면 다음 단계로
        
        # 1.6.2차: 상대 표현 패턴: "3일 뒤", "2주 후", "1개월 후"
        relative_patterns = [
            # 일 단위: "3일 뒤", "5일 후", "일주일 후"
            (r"(\d+)\s*일\s*(뒤|후)", lambda m: now + timedelta(days=int(m.group(1)))),
            (r"일주일\s*(뒤|후)", lambda m: now + timedelta(days=7)),
            (r"한주\s*(뒤|후)", lambda m: now + timedelta(days=7)),
            # 주 단위: "2주 후", "3주 뒤"
            (r"(\d+)\s*주\s*(뒤|후)", lambda m: now + timedelta(weeks=int(m.group(1)))),
            # 개월 단위: "1개월 후", "2개월 뒤"
            (r"(\d+)\s*개월\s*(뒤|후)", lambda m: now + timedelta(days=int(m.group(1)) * 30)),  # 근사치
            # 년 단위: "1년 후", "2년 뒤"
            (r"(\d+)\s*년\s*(뒤|후)", lambda m: now + timedelta(days=int(m.group(1)) * 365)),  # 근사치
        ]
        
        for pattern, handler in relative_patterns:
            m = re.match(pattern, date_str.strip())
            if m:
                try:
                    target_date = handler(m)
                    result = target_date.strftime('%Y-%m-%d')
                    self._add_to_date_cache(date_str, result)  # 캐시에 저장
                    return result
                except (ValueError, OverflowError):
                    pass  # 잘못된 날짜면 다음 패턴 시도
        
        # 1.7차: 자연어 날짜 패턴 직접 처리 (성능 최적화)
        natural_date_patterns = [
            # "이번 달 15일", "다음 달 20일" 패턴
            (r"(이번\s*달|다음\s*달|다다음\s*달)\s*(\d{1,2})일", self._parse_month_day_natural),
            # "이번 주 금요일", "다음 주 월요일" 패턴 (이미 위에서 처리됨)
            # "이번 달 마지막 날" 패턴
            (r"(이번\s*달|다음\s*달|다다음\s*달)\s*마지막\s*날", self._parse_month_last_day),
            # "이번 달 첫째 주 금요일" 패턴
            (r"(이번\s*달|다음\s*달|다다음\s*달)\s*첫째\s*주\s*(월요일|화요일|수요일|목요일|금요일|토요일|일요일)", self._parse_month_first_week),
        ]
        
        for pattern, handler in natural_date_patterns:
            m = re.search(pattern, date_str.strip())
            if m:
                try:
                    result = handler(m, now)
                    if result:
                        logger.debug(f"자연어 날짜 처리 성공: '{date_str}' → '{result}'")
                        self._add_to_date_cache(date_str, result)  # 캐시에 저장
                        return result
                except Exception as e:
                    logger.debug(f"자연어 날짜 처리 실패: {str(e)[:50]}...")
                    pass  # 실패하면 다음 패턴 시도
        
        # 2차: dateparser로 자연어 날짜 파싱 시도
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
                self._add_to_date_cache(date_str, result)  # 캐시에 저장
                return result
        except Exception:
            pass
        
        # 3차: LLM fallback (특수한 경우나 문화적 날짜) - confidence 기반 검증
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
                # 안전한 JSON 파싱 (response가 string일 경우 대비)
                try:
                    result = json.loads(str(response.content))
                    # confidence가 high이고 date가 있을 때만 반환
                    if result.get("confidence") == "high" and result.get("date"):
                        self._add_to_date_cache(date_str, result["date"])  # 캐시에 저장
                        return result["date"]
                    # confidence가 low이면 None 반환 (사용자 재질문 필요)
                    elif result.get("confidence") == "low":
                        return None
                except (json.JSONDecodeError, TypeError):
                    pass  # JSON 파싱 실패 시 다음 단계로
        except Exception:
            pass
        
        # 모든 방법이 실패하면 원본 반환
        return date_str

    def _parse_month_day_natural(self, match, now: datetime) -> str:
        """자연어 월/일 패턴 파싱 (이번 달 15일, 다음 달 20일)"""
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
        else:  # 다다음 달
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
        """월 마지막 날 패턴 파싱 (이번 달 마지막 날)"""
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
        else:  # 다다음 달
            if now.month >= 11:
                target_month = now.month + 2 - 12
                target_year = now.year + 1
            else:
                target_month = now.month + 2
                target_year = now.year
        
        # 해당 월의 마지막 날 계산
        if target_month == 12:
            next_month = 1
            next_year = target_year + 1
        else:
            next_month = target_month + 1
            next_year = target_year
        
        last_day = datetime(next_year, next_month, 1) - timedelta(days=1)
        return last_day.strftime('%Y-%m-%d')

    def _parse_month_first_week(self, match, now: datetime) -> str:
        """월 첫째 주 요일 패턴 파싱 (이번 달 첫째 주 금요일)"""
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
        else:  # 다다음 달
            if now.month >= 11:
                target_month = now.month + 2 - 12
                target_year = now.year + 1
            else:
                target_month = now.month + 2
                target_year = now.year
        
        # 해당 월의 첫째 주 해당 요일 찾기
        first_day = datetime(target_year, target_month, 1)
        days_ahead = (target_weekday - first_day.weekday()) % 7
        target_date = first_day + timedelta(days=days_ahead)
        
        return target_date.strftime('%Y-%m-%d')

    def _check_date_normalization_failure(self, entities: Dict[str, List[Dict[str, Any]]]) -> Optional[str]:
        """날짜 정규화 실패 시 사용자에게 재질문하는 메시지 생성 (모든 날짜 엔티티 검사)"""
        if not entities:
            return None
            
        # 모든 날짜 관련 엔티티에서 날짜 정규화 실패 확인
        date_entity_types = ["user.일정", "user.약", "user.식사", "user.기념일", "user.건강상태"]
        
        for entity_type, entity_list in entities.items():
            if entity_type in date_entity_types and isinstance(entity_list, list):
                for entity in entity_list:
                    if isinstance(entity, dict) and "날짜" in entity:
                        date_value = entity["날짜"]
                        # 날짜가 정규화되지 않은 원본 문자열인 경우
                        if date_value and not self._is_normalized_date(date_value):
                            # 먼저 날짜 정규화 시도
                            normalized_date = self._normalize_date(date_value)
                            # 정규화가 실패했거나 원본과 동일하면 재질문
                            if normalized_date == date_value or not self._is_normalized_date(normalized_date):
                                # 일정 엔티티에서 제목이 날짜 필드에 잘못 들어간 경우 스킵
                                if entity_type == "user.일정" and "제목" in entity and entity["제목"] == date_value:
                                    continue
                                # 월/일 패턴은 정규화가 가능하므로 스킵
                                if re.match(r'\d{1,2}월\s*\d{1,2}일', date_value):
                                    continue
                                # 일정 엔티티에서 제목 패턴이 날짜 필드에 들어간 경우 스킵
                                if entity_type == "user.일정" and any(re.search(pattern, date_value) for pattern in [r'(여행|파티|회의|약속|미팅|데이트|일정|스케줄|예약)', r'(병원|치과|약국|은행|우체국)']):
                                    continue
                                return f"'{date_value}'라는 날짜 표현이 명확하지 않네요. 구체적인 날짜로 말씀해주실래요? (예: '다음주 금요일', '12월 25일' 등)"
        
        return None

    def _is_normalized_date(self, date_str: str) -> bool:
        """날짜 문자열이 정규화된 형식인지 확인 (YYYY-MM-DD)"""
        if not date_str:
            return False
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False

    def _extract_name_llm(self, text: str) -> Optional[Dict[str, str]]:
        """LLM 기반 이름 및 별칭 추출 (문맥 이해)"""
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
                # 안전한 JSON 파싱 (response가 string일 경우 대비)
                try:
                    result = json.loads(str(response.content))
                    name = result.get('name')
                    alias = result.get('alias')
                    confidence = result.get('confidence', 0.0)
                    
                    # 확신도가 0.7 이상일 때만 LLM 결과 사용
                    if confidence >= 0.7 and name and self._is_valid_name(name):
                        # 이름 정규화 적용 (LLM 결과도 정규화)
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
        """텍스트에서 감정 상태 분석"""
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
        """감정적 맥락을 유지해야 하는지 판단"""
        if session_id not in self.emotional_state:
            return False
        
        emotional_info = self.emotional_state[session_id]
        # 최근 3턴 이내에 강한 감정이 있었고, 현재가 기능적 요청인 경우
        if (emotional_info.get("intensity", 0) > 0.6 and 
            emotional_info.get("last_emotional_turn", 0) <= 3 and
            current_category in ["cognitive", "physical"]):
            return True
        return False

    def _is_valid_name(self, name: str) -> bool:
        """이름 유효성 검증 (유연한 길이 제한)"""
        if not name or len(name) < 2:
            return False
        
        # 부사/형용사/일반 단어 제외
        invalid_words = {
            "사실", "편하게", "그냥", "정말", "진짜", "완전", "너무", "정말로",
            "그러면", "그래서", "그런데", "하지만", "그리고", "그러나",
            "좋아", "싫어", "좋다", "싫다", "맞다", "틀리다",
            "이름", "나", "내", "저", "제", "우리", "너", "당신"
        }
        
        if name in invalid_words:
            return False
        
        # 한글 이름 패턴 (2-5글자) - 현수민, 김철수민 등 긴 이름 지원
        if re.match(r'^[가-힣]{2,5}$', name):
            return True
        
        # 영문 이름 패턴 (2-15글자) - Alexander, Christopher 등 긴 이름 지원
        if re.match(r'^[a-zA-Z]{2,15}$', name):
            return True
        
        return False

    def _extract_nickname(self, text: str) -> Optional[str]:
        """사용자가 제안한 별칭 추출 (deprecated - LLM 기반 추출 사용)"""
        # 하드코딩된 패턴 매칭 제거됨. 별명 추출은 entity_chain의 LLM이 처리합니다.
        return None

    def _is_name_question(self, text: str) -> bool:
        """이름 확인 질문 패턴 확인"""
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
        """사용자 엔티티 병합 로직 개선"""
        if not existing_entities:
            return [new_entity]
        
        # 기존 엔티티와 새 엔티티 비교
        existing = existing_entities[0]
        new_name = new_entity.get("이름", "")
        new_aliases = new_entity.get("별칭", [])
        
        # 이름이 같은 경우
        if existing.get("이름") == new_name:
            # 별칭 병합
            if new_aliases:
                existing_aliases = existing.get("별칭", [])
                for alias in new_aliases:
                    if alias not in existing_aliases:
                        existing_aliases.append(alias)
                existing["별칭"] = existing_aliases
            return existing_entities
        
        # 별칭과 본명 매칭 확인
        existing_name = existing.get("이름", "")
        existing_aliases = existing.get("별칭", [])
        
        # 새 이름이 기존 별칭과 같은 경우
        if new_name in existing_aliases:
            # 새 이름을 별칭에서 제거하고 본명으로 설정
            existing_aliases.remove(new_name)
            existing["이름"] = new_name
            existing["별칭"] = existing_aliases
            return existing_entities
        
        # 기존 이름이 새 별칭과 같은 경우
        if existing_name in new_aliases:
            # 기존 이름을 별칭으로 이동
            existing_aliases = existing.get("별칭", [])
            if existing_name not in existing_aliases:
                existing_aliases.append(existing_name)
            existing["이름"] = new_name
            existing["별칭"] = existing_aliases
            return existing_entities
        
        # 부분 매칭 확인 (권서연 vs 서연)
        if self._is_name_variant(existing_name, new_name):
            # 더 긴 이름을 본명으로, 짧은 이름을 별칭으로
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
        
        # 매칭되지 않는 경우 새 엔티티 추가
        return existing_entities + [new_entity]

    def _is_name_variant(self, name1: str, name2: str) -> bool:
        """두 이름이 변형 관계인지 확인 (권서연 vs 서연)"""
        if not name1 or not name2:
            return False
        
        # 공백 제거 후 비교
        clean1 = name1.replace(" ", "")
        clean2 = name2.replace(" ", "")
        
        # 한 이름이 다른 이름의 끝부분과 일치하는지 확인
        return clean1.endswith(clean2) or clean2.endswith(clean1)

    def _get_existing_user_entities(self, session_id: str) -> List[Dict]:
        """기존 사용자 엔티티 조회"""
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
        """단일 엔티티 처리 (기존 로직을 메서드로 분리)"""
        try:
            # 일정 저장 확인
            if entity_key.endswith("일정"):
                has_schedule = True
            
            # 식사 엔티티에서 시간이 없으면 강제로 질문 추가 (필수 필드 체크에서 처리됨)
            
            # 약을 식사로 착각한 경우 제거
            if entity_key.endswith("식사") and "메뉴" in filtered_value:
                menus = filtered_value["메뉴"]
                if isinstance(menus, list):
                    # 약명이 포함된 메뉴 제거
                    filtered_menus = [menu for menu in menus if not menu.endswith("약")]
                    if not filtered_menus:
                        # 모든 메뉴가 약이면 이 식사 엔티티 제거
                        print(f"[INFO] 약을 식사로 착각한 엔티티 제거: {entity_key} - {filtered_value}")
                        return has_schedule
                    filtered_value["메뉴"] = filtered_menus
            
            # 필수 필드 체크
            missing_fields = self._check_missing_fields(entity_key, filtered_value)

            if missing_fields:
                # 3️⃣ 필수 필드가 비면 follow-up 질문 생성 (저장은 보류)
                logger.debug(f"누락된 필드 감지: {entity_key} - {missing_fields}, 값: {filtered_value}")
                followup_questions = self._generate_followup_questions(entity_key, missing_fields, filtered_value)
                questions.extend(followup_questions)
                
                # 식사 엔티티는 메뉴나 시간이 없어도 저장 (점진적 정보 수집)
                if entity_key.endswith("식사") and (
                    ("메뉴" in missing_fields and filtered_value.get("메뉴") == []) or
                    ("시간" in missing_fields and not filtered_value.get("시간"))
                ):
                    # 메뉴가 빈 리스트이거나 시간이 없는 경우에도 저장
                    final_value = self._add_to_vstore(
                        entity_key, filtered_value,
                        {"session_id": session_id, "entity_key": entity_key, "type": "entity", "created_at": datetime.now().isoformat()},
                        strategy="merge",
                        session_id=session_id
                    )
                    logger.debug(f"식사 엔티티 임시 저장 (메뉴/시간 누락): {filtered_value}")
                    
                    # 시간이 누락된 경우 재질문 상태 설정
                    if "시간" in missing_fields and not filtered_value.get("시간"):
                        self.pending_question[session_id] = {
                            "기존_엔티티": final_value,
                            "새_엔티티": final_value,
                            "entity_key": entity_key
                        }
                        print(f"[DEBUG] 재질문 상태 설정: {entity_key} - 시간 누락")
                else:
                    return has_schedule

            # 2️⃣ 모든 필수 필드가 있으면 저장 (merge 정책 적용)
            final_value = self._add_to_vstore(
                entity_key, filtered_value,
                {"session_id": session_id, "entity_key": entity_key, "type": "entity", "created_at": datetime.now().isoformat()},
                strategy="merge"
            )

            # 약은 복용 정보 세부 필드 확인 후 추가 질문
            if entity_key.endswith(".약"):
                if final_value.get("복용"):
                    enriched = [self._enrich_dose_dict(d) for d in final_value["복용"]]
                    final_value["복용"] = enriched
                    
                    # enrich된 정보를 VectorStore에 업데이트
                    try:
                        self._add_to_vstore(
                            entity_key=entity_key,
                            value=final_value,
                            metadata={"session_id": session_id, "type": "entity"},
                            strategy="merge",
                            identity=final_value.get("약명"),
                            session_id=session_id
                        )
                        # print(f"[DEBUG] 복용 정보 enrich 후 VectorStore 업데이트 완료: {final_value.get('약명')}")
                    except Exception as e:
                        print(f"[WARN] 복용 정보 enrich 후 업데이트 실패: {e}")
                    
                    # 복용 정보가 이미 있는 경우 추가 질문하지 않음
                    # "복용" 필드가 있으면 충분한 정보로 간주
                    pass
            
            return has_schedule
            
        except Exception as e:
            print(f"[ERROR] 엔티티 처리 실패: {e}")
            return has_schedule

    def _get_cached_classification(self, text: str, similarity_threshold: float = 0.85) -> Optional[Dict]:
        """임베딩 기반 캐시에서 유사한 분류 결과 찾기"""
        if not self.cache_texts or self.cache_embeddings is None:
            return None
        
        try:
            # 입력 텍스트를 임베딩으로 변환
            input_embedding = self.vectorizer.transform([text])
            
            # 기존 캐시와 유사도 계산
            similarities = cosine_similarity(input_embedding, self.cache_embeddings)[0]
            
            # 가장 유사한 결과 찾기
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
        """분류 결과를 캐시에 추가"""
        try:
            # 캐시에 결과 저장
            self.classification_cache[text] = result
            
            # 텍스트 리스트에 추가
            self.cache_texts.append(text)
            
            # 임베딩 업데이트 (최대 100개까지만 유지)
            if len(self.cache_texts) > 100:
                # 오래된 것 제거
                old_text = self.cache_texts.pop(0)
                self.classification_cache.pop(old_text, None)
            
            # 임베딩 재계산
            if self.cache_texts:
                self.cache_embeddings = self.vectorizer.fit_transform(self.cache_texts)
            
        except Exception as e:
            logger.warning(f"캐시 추가 실패: {e}")

    def _sync_to_exact_cache(self, user_input: str, result_dict: Dict, pre_entities: Dict = None) -> None:
        """정확 캐시에 동기화 (task_classifier.py의 LRU 캐시)"""
        try:
            # task_classifier의 캐시 함수들을 동적으로 import
            from .task_classifier import _add_to_cache as add_exact_cache, ClassificationResult
            
            # 입력 정규화 (task_classifier와 동일한 방식)
            import re
            norm_text = re.sub(r"\s+", " ", user_input.strip().lower())
            
            # ClassificationResult 객체 생성
            category = result_dict.get("category", "query")
            confidence = result_dict.get("confidence", 0.5)
            probabilities = result_dict.get("probabilities", {category: 0.5})
            
            # Confidence threshold 적용 (task_classifier.py와 일관성 유지)
            if confidence < CONFIDENCE_THRESHOLD:
                logger.warning(f"낮은 confidence로 인한 fallback: {confidence:.2f} < {CONFIDENCE_THRESHOLD}")
                # 낮은 confidence일 때는 기본 분류 사용
                from .task_classifier import classify
                category, _ = classify_hybrid(user_input, pre_entities)
                confidence = 0.5
                probabilities = {category: 0.5}
            
            classification_result = ClassificationResult(category, confidence, probabilities)
            
            # 정확 캐시에 추가
            add_exact_cache(norm_text, classification_result)
            logger.debug(f"정확 캐시 동기화: '{norm_text[:30]}...'")
            
        except Exception as e:
            logger.error(f"정확 캐시 동기화 실패: {str(e)[:50]}...")

    def _classify_with_cache(self, user_input: str, pre_entities: Dict[str, List[Dict[str, Any]]] = None) -> Dict:
        """다층 캐시 기반 하이브리드 분류 (캐시 동기화)"""
        # 1차: 정확 캐싱 (task_classifier.py에서 처리됨)
        # 2차: 유사 캐싱 (임베딩 기반)
        cached_result = self._get_cached_classification(user_input)
        if cached_result:
            logger.debug(f"유사 캐시 hit: '{user_input[:30]}...'")
            # 유사 캐시 hit 시 정확 캐시에도 동기화
            self._sync_to_exact_cache(user_input, cached_result, pre_entities)
            return cached_result
        
        # 3차: 하이브리드 분류 실행 (정확 캐싱 포함)
        try:
            classification_result = classify_hybrid(user_input, pre_entities)
            result_dict = classification_result.to_dict()
            
            # 양쪽 캐시에 동기화하여 저장
            self._add_to_cache(user_input, classification_result)  # 유사 캐시
            self._sync_to_exact_cache(user_input, result_dict, pre_entities)      # 정확 캐시
            
            return result_dict
            
        except Exception as e:
            logger.error(f"하이브리드 분류 실패, 기본 분류 사용: {e}")
            # 4차: 기본 분류 fallback
            category, _ = classify_hybrid(user_input, pre_entities)
            fallback_result = {
                "category": category,
                "confidence": 0.5,
                "probabilities": {category: 0.5}
            }
            
            # fallback 결과도 캐시에 저장
            self._sync_to_exact_cache(user_input, fallback_result)
            return fallback_result

    def _normalize_duration(self, duration_str: str) -> str:
        """복용 기간을 정규화된 형태로 변환"""
        if not duration_str:
            return duration_str
        
        # 정규화 매핑
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
        
        # 정확한 매칭
        if duration_str in duration_mapping:
            return duration_mapping[duration_str]
        
        # 패턴 매칭 (숫자 + 단위)
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
        
        # 매칭되지 않으면 원본 반환
        return duration_str

    def _extract_duration_from_dosage(self, dosage_list: List[dict]) -> Tuple[List[dict], str]:
        """복용 배열에서 기간 정보를 분리하여 복용 기간으로 추출"""
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
                        if not extracted_period:  # 첫 번째로 발견된 기간만 사용
                            extracted_period = self._normalize_duration(text)
                        is_period = True
                        break
                
                if not is_period:
                    filtered_dosage.append(dosage)
            else:
                filtered_dosage.append(dosage)
        
        return filtered_dosage, extracted_period

    def _is_valid_entity(self, entity_key: str, value: dict) -> bool:
        """엔티티 유효성 검사 (잘못된 데이터 저장 방지)"""
        if entity_key.endswith("사용자") and "이름" in value:
            name = value["이름"]
            # 이름이 블랙리스트에 있거나 너무 짧으면 무효
            if name in NAME_BLACKLIST or len(name) < 2:
                return False
            # 한 글자 이름도 무효 (예: "화")
            if len(name) == 1:
                return False
        
        if entity_key.endswith("물건") and "이름" in value:
            item_name = value["이름"]
            # 물건 이름이 너무 일반적이거나 짧으면 무효
            if item_name in {"물건", "거", "것", "뭐", "뭔가", "화", "알고", "다시"} or len(item_name) < 1:
                return False
        
        if entity_key.endswith("식사") and "메뉴" in value:
            menus = value["메뉴"]
            if isinstance(menus, list):
                # 메뉴가 비어있거나 의미없는 단어만 있으면 무효 (밥은 유효한 메뉴)
                valid_menus = [m for m in menus if m and m not in STOPWORDS and (len(m) > 1 or m == "밥")]
                if not valid_menus:
                    return False
        
        return True

    def maintenance_dedup_user(self, session_id: str):
        """잘못된 사용자 엔티티 정리 (1회성 마이그레이션)"""
        try:
            # 현재 세션의 사용자 엔티티 조회 (필터 없이)
            docs = self.vectorstore.similarity_search("사용자 이름", k=100)
            
            valid_docs = []
            invalid_ids = []
            
            for doc in docs:
                try:
                    data = json.loads(doc.page_content)
                    # 세션 ID와 엔티티 키 확인
                    if (data.get("session_id") == session_id and 
                        data.get("entity_key") == "user.사용자" and
                        self._is_valid_entity("user.사용자", data)):
                        valid_docs.append(doc)
                    elif (data.get("session_id") == session_id and 
                          data.get("entity_key") == "user.사용자"):
                        invalid_ids.append(doc.metadata.get("id"))
                except Exception:
                    invalid_ids.append(doc.metadata.get("id"))
            
            # 잘못된 엔티티 삭제
            if invalid_ids:
                self.vectorstore.delete(ids=invalid_ids)
                print(f"[MAINT] 잘못된 사용자 엔티티 {len(invalid_ids)}개 삭제")
            
            print(f"[MAINT] 유효한 사용자 엔티티 {len(valid_docs)}개 유지")
            
        except Exception as e:
            print(f"[WARN] 사용자 엔티티 정리 실패: {e}")

    def _extract_time_from_text(self, text: str) -> str:
        """텍스트에서 시간 추출 (_normalize_datetime과 통일)"""
        time_patterns = [
            r"(\d{1,2}시\s*반)",
            r"(\d{1,2}시\s*\d{1,2}분)",
            r"(\d{1,2}:\d{2})",  # 7:30 같은 형식 우선
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
        """텍스트에서 메뉴 정보 추출 (단순한 메뉴명만)"""
        # 메뉴 패턴들 (음식명 추출)
        menu_patterns = [
            r"([가-힣]{2,10})",  # 한글 2-10글자 (음식명)
        ]
        
        # 불용어 제외
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
        """시간 문자열에서 시간(시) 추출"""
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
        """시간 형식 정규화 (7:30 → 7시 30분)"""
        if not time_str:
            return time_str
        
        # 7:30 → 7시 30분 변환
        if re.match(r"\d{1,2}:\d{2}", time_str):
            hour, minute = time_str.split(":")
            return f"{hour}시 {minute}분"
        
        return time_str

    def _normalize_time_field(self, time_value) -> List[str]:
        """시간 필드 정규화 (항상 list로 변환)"""
        if not time_value:
            return []
        if isinstance(time_value, list):
            return [str(t) for t in time_value if t]
        return [str(time_value)]

    def _entity_identity(self, entity_key: str, v: dict) -> str:
        """엔티티 identity 생성"""
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
        """엔티티 스쿼시 (중복/불완전 엔티티 정리)"""
        out = {}
        for k, vals in entities.items():
            by_id: Dict[str, dict] = {}
            for v in vals:
                # N/A 값 필터링
                filtered_v = self._filter_meaningful_data(v, user_input)
                if not filtered_v:
                    continue
                    
                id_ = self._entity_identity(k, filtered_v)
                base = by_id.get(id_, {})
                # 필드 채우기(새 값이 더 구체적이면 교체)
                for fk, fv in filtered_v.items():
                    if fv in (None, "", []): 
                        continue
                    if isinstance(fv, list) and isinstance(base.get(fk), list):
                        # 딕셔너리가 포함된 리스트는 중복 제거 후 합치기
                        combined = base[fk] + fv
                        base[fk] = self._dedup_entities(combined) if all(isinstance(item, dict) for item in combined) else list({*base[fk], *fv})
                    else:
                        # 길이가 더 긴 문자열/더 많은 정보 우선
                        if not base.get(fk) or (isinstance(fv, str) and len(str(fv)) > len(str(base.get(fk)))):
                            base[fk] = fv
                by_id[id_] = base
            
            # 식사는 불완전해도 임시 저장 (점진적 merge 지원)
            if k.endswith("식사"):
                # 식사 엔티티 중복 제거 - 같은 끼니와 날짜가 있으면 하나만 유지
                unique_meals = []
                seen_combinations = set()
                for v in by_id.values():
                    meal_key = f"{v.get('끼니', '')}_{v.get('날짜', '')}_{v.get('시간', '')}"
                    if meal_key not in seen_combinations or not meal_key.strip('_'):
                        unique_meals.append(v)
                        seen_combinations.add(meal_key)
                out[k] = unique_meals
            else:
                # 완성된 것만 남김(필수필드 채워진 것 위주)
                filtered = []
                for v in by_id.values():
                    missing = self._check_missing_fields(k, v)
                    if not missing:
                        filtered.append(v)
                    else:
                        # 전부 빠지면 질문 유도용으로 하나는 남길 수도 있지만,
                        # 이번 이슈(불필요 재질문) 방지를 위해 완성된 것만 우선 저장
                        pass
                out[k] = filtered if filtered else list(by_id.values())
        
        # 약 엔티티에 특별히 중복 제거 적용
        if "user.약" in out:
            out["user.약"] = self._dedup_drug_entities(out["user.약"])
        
        return out


    def _enrich_dose_dict(self, d: dict) -> dict:
        """약 복용 정보 파서 강화"""
        txt = d.get("원문","") or ""
        # 횟수: 숫자/한글 수사
        m = re.search(r"하루\s*(\d+)\s*번", txt)
        if m: d["횟수"] = int(m.group(1))
        kor = {"한":1,"두":2,"세":3,"네":4,"다섯":5}
        m = re.search(r"하루\s*([한두세네다섯])\s*번", txt)
        if m: d["횟수"] = kor[m.group(1)]
        # 식전/식후
        if "식후" in txt: d["식전후"] = "식후"
        elif "식전" in txt: d["식전후"] = "식전"
        # 시간대 힌트(아침/점심/저녁)
        if any(k in txt for k in ["아침","점심","저녁"]):
            d.setdefault("시간대", [])  # 자유 필드
            for k in ["아침","점심","저녁"]:
                if k in txt and k not in d["시간대"]:
                    d["시간대"].append(k)
        return d

    # VectorStore 저장 (merge 포함)
    def _filter_meaningful_data(self, value: dict, user_input: str = None) -> dict:
        """N/A 값이나 의미없는 데이터를 필터링"""
        if not value:
            return {}
        
        # 타입 안전성 체크
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
        
        # 약 엔티티의 경우 복용 정보가 있으면 유효한 엔티티로 인정
        if "약명" in filtered and ("복용" in filtered or "식사와의 관계" in filtered):
            return filtered
        
        # 약 복용 엔티티의 경우 시간대, 복용, 날짜 중 하나라도 있으면 유효한 엔티티로 인정
        if any(key in filtered for key in ["시간대", "복용", "날짜"]) and any(key in filtered for key in ["시간대", "복용", "날짜"]):
            return filtered
        
        # 모든 필드가 필터링되면 빈 딕셔너리 반환
        if not filtered:
            return {}
        
        return filtered

    def _merge_meal_entity(self, existing: dict, new: dict) -> dict:
        """식사 엔티티 병합 - 같은 날짜/끼니/시간이면 메뉴 리스트만 갱신"""
        # ✅ merge 보호 로직: existing이 None이거나 dict가 아니면 새 값 반환
        if existing is None or not isinstance(existing, dict):
            return new
        merged = existing.copy()
        
        # 메뉴 병합
        if "메뉴" in new and new["메뉴"]:
            existing_menus = existing.get("메뉴", [])
            new_menus = new["메뉴"] if isinstance(new["메뉴"], list) else [new["메뉴"]]
            
            # 기존 메뉴와 새 메뉴 합치기 (중복 제거)
            all_menus = list(set(existing_menus + new_menus))
            merged["메뉴"] = all_menus
        
        # 다른 필드들도 업데이트 (새 값이 있으면)
        for key, val in new.items():
            if key != "메뉴" and val and val not in ["", "N/A", "null"]:
                merged[key] = val
        
        return merged

    def _add_to_vstore(self, entity_key: str, value: dict, metadata: dict, strategy: str = "merge", identity: Optional[str] = None, user_input: str = None, session_id: Optional[str] = None) -> dict:
        try:
            # N/A 값 필터링 - 의미있는 데이터만 저장
            filtered_value = self._filter_meaningful_data(value, user_input)
            if not filtered_value:
                logger.debug(f"의미없는 데이터 필터링: {entity_key} - {value}")
                return value
        except Exception as e:
            if str(e).startswith("QUESTION:"):
                question = str(e)[9:]  # "QUESTION:" 제거
                print(f"[DEBUG] _add_to_vstore에서 재질문 예외 처리: {question}")
                return {"질문": question}
            raise e
        
        # 재질문이 반환된 경우
        if isinstance(filtered_value, dict) and "질문" in filtered_value:
            print(f"[DEBUG] _add_to_vstore에서 재질문 반환: {filtered_value['질문']}")
            # 재질문을 전역 상태에 저장하고 즉시 반환
            if session_id:
                self.current_question[session_id] = filtered_value["질문"]
            return filtered_value
        
        # 완전성 검사 - 필수 필드가 없는 경우 저장하지 않음 (식사 제외)
        if not entity_key.endswith("식사") and not self._is_complete_entity(entity_key, filtered_value):
            logger.debug(f"불완전한 엔티티 저장 거부: {entity_key} - {filtered_value}")
            return filtered_value
        
        base_key = f"{metadata.get('session_id', '')}_{entity_key}"
        if identity is None:
            # 1차 목표: Identity 정책 단순화
            if entity_key.endswith("사용자"):
                identity = "user_name"  # 사용자는 고정
            elif entity_key.endswith("일정"):
                identity = f"{filtered_value.get('제목')}|{filtered_value.get('날짜')}"  # 제목+날짜만
            elif entity_key.endswith("물건"):
                identity = filtered_value.get("이름")  # 이름만
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

        # 날짜 정규화 강제 적용 (모든 엔티티)
        session_id = metadata.get('session_id', 'default')
        if "날짜" in filtered_value and filtered_value["날짜"]:
            filtered_value["날짜"] = self._normalize_date(filtered_value["날짜"], session_id)
        elif entity_key.endswith(("일정", "식사", "기념일", "약", "건강상태", "취미", "취향")):
            # 날짜 필드가 없으면 오늘로 자동 삽입
            filtered_value["날짜"] = self._normalize_date("오늘", session_id)
        
        # 물건 위치 정규화 적용 (저장 시점에서 강제 정규화)
        if entity_key.endswith("물건") and "위치" in filtered_value:
            filtered_value["위치"] = self._normalize_location(filtered_value["위치"])

        try:
            # 사용자 정보 중복 방지 - 같은 세션 내에서만 중복 체크
            if entity_key.endswith("사용자") and "이름" in filtered_value:
                filtered_value["이름"] = self._normalize_name(filtered_value["이름"])
                
                # 엑셀 캐시에서 중복 체크 (VectorStore 비활성화 대응)
                if hasattr(self, 'excel_cache') and session_id in self.excel_cache:
                    cache_entities = self.excel_cache[session_id].get("사용자", [])
                    for cached_entity in cache_entities:
                        if isinstance(cached_entity, dict) and cached_entity.get("이름") == filtered_value.get("이름"):
                            logger.debug(f"사용자 정보 중복 방지: '{filtered_value.get('이름')}' 이미 존재 (세션: {session_id})")
                            return self._merge_entity_values(cached_entity, filtered_value, "user.사용자")
                
                # VectorStore 비활성화 - 엑셀 캐시 기반 중복 체크로 대체됨 (위에서 이미 처리)
            
            # 가족 정보 중복 방지 - 같은 세션 내에서만 중복 체크
            if entity_key.endswith("가족") and "이름" in filtered_value and "관계" in filtered_value:
                filtered_value["이름"] = self._normalize_name(filtered_value["이름"])
                
                # 엑셀 캐시에서 중복 체크 (VectorStore 비활성화 대응)
                if hasattr(self, 'excel_cache') and session_id in self.excel_cache:
                    cache_entities = self.excel_cache[session_id].get("가족", [])
                    for cached_entity in cache_entities:
                        if (isinstance(cached_entity, dict) and 
                            cached_entity.get("이름") == filtered_value.get("이름") and
                            cached_entity.get("관계") == filtered_value.get("관계")):
                            logger.debug(f"가족 정보 중복 방지: '{filtered_value.get('관계')} {filtered_value.get('이름')}' 이미 존재 (세션: {session_id})")
                            return self._merge_entity_values(cached_entity, filtered_value, "user.가족")
                
                # VectorStore 비활성화 - 엑셀 캐시 기반 중복 체크로 대체됨
            
            # 동적 중복 검사 - 물건은 전역, 나머지는 세션별
            if "이름" in filtered_value and filtered_value.get("이름"):
                print(f"[DEBUG] 동적 중복 검사 시작: entity_key={entity_key}, session_id={session_id}, 이름={filtered_value.get('이름')}")
                
                entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
                existing_entities = []
                
                # 엑셀 캐시에서 중복 체크 (VectorStore 비활성화 대응)
                if hasattr(self, 'excel_cache'):
                    # 물건은 전역 체크 (모든 세션), 나머지는 세션별 체크
                    if entity_key.endswith("물건"):
                        # 모든 세션의 캐시 확인
                        for sess_id, cache_data in self.excel_cache.items():
                            cache_entities = cache_data.get(entity_type, [])
                            for cached_entity in cache_entities:
                                if isinstance(cached_entity, dict) and cached_entity.get("이름") == filtered_value.get("이름"):
                                    existing_entities.append(cached_entity)
                                    print(f"[DEBUG] 기존 엔티티 발견: {cached_entity}")
                    else:
                        # 세션별 체크
                        if session_id in self.excel_cache:
                            cache_entities = self.excel_cache[session_id].get(entity_type, [])
                            for cached_entity in cache_entities:
                                if isinstance(cached_entity, dict):
                                    existing_entities.append(cached_entity)
                                    print(f"[DEBUG] 기존 엔티티 발견: {cached_entity}")
                
                # 동일한 엔티티 타입이 이미 존재하는 경우 최신 정보로 자동 업데이트 (재질문 제거)
                if existing_entities:
                    existing_name = existing_entities[0].get("이름", "")
                    new_name = filtered_value.get("이름", "")
                    entity_type = entity_key.replace("user.", "")
                    
                    print(f"[DEBUG] 중복 엔티티 발견: 기존='{existing_name}', 새='{new_name}' - 최신 정보로 자동 업데이트")
                    
                    # 같은 이름이면 최신 정보로 자동 병합 (재질문 없이)
                    if existing_name == new_name:
                        # 기존 엔티티와 새 엔티티 병합하여 최신 정보 반영
                        merged = self._merge_entity_values(existing_entities[0], filtered_value, entity_key)
                        print(f"[DEBUG] 중복 엔티티 자동 병합 완료: {merged}")
                        # 병합된 엔티티로 filtered_value 업데이트 (아래 엑셀 저장 로직 실행되도록)
                        filtered_value = merged
                        # 병합된 엔티티를 캐시에서 업데이트
                        if session_id in self.excel_cache:
                            cache_entities = self.excel_cache[session_id].get(entity_type, [])
                            for i, cached_entity in enumerate(cache_entities):
                                if isinstance(cached_entity, dict) and cached_entity.get("이름") == existing_name:
                                    cache_entities[i] = merged
                                    break
                        # return 하지 않고 계속 진행하여 엑셀에도 저장되도록 함
                    else:
                        # 다른 이름이면 새 엔티티로 저장 (기존 엔티티는 유지)
                        print(f"[DEBUG] 다른 이름의 엔티티 - 새 엔티티로 저장: {new_name}")
                        # 계속 진행하여 새 엔티티 저장
            
             # 약 정보 중복 방지 - 최신 정보로 자동 업데이트 (재질문 제거)
            if entity_key.endswith("약") and "약명" in filtered_value:
                # 엑셀 캐시에서 동일한 약 정보 확인 (약명 기준)
                if hasattr(self, 'excel_cache') and session_id in self.excel_cache:
                    cache_entities = self.excel_cache[session_id].get("약", [])
                    for cached_entity in cache_entities:
                        if isinstance(cached_entity, dict) and cached_entity.get("약명") == filtered_value.get("약명"):
                            logger.debug(f"약 정보 중복 발견: '{filtered_value.get('약명')}' 이미 존재 - 최신 정보로 자동 병합 (세션: {session_id})")
                            # 최신 정보로 자동 병합
                            merged = self._merge_entity_values(cached_entity, filtered_value, "user.약")
                            # 병합된 엔티티로 filtered_value 업데이트 (아래 엑셀 저장 로직 실행되도록)
                            filtered_value = merged
                            # 병합된 엔티티를 캐시에서 업데이트
                            for i, entity in enumerate(cache_entities):
                                if isinstance(entity, dict) and entity.get("약명") == filtered_value.get("약명"):
                                    cache_entities[i] = merged
                                    break
                            # return 하지 않고 계속 진행하여 엑셀에도 저장되도록 함
                            break
            
            # 식사 정보 중복 방지 - 엑셀 캐시 기반 (VectorStore 비활성화 대응)
            if entity_key.endswith("식사"):
                # 엑셀 캐시에서 동일한 식사 정보 확인
                if hasattr(self, 'excel_cache') and session_id in self.excel_cache:
                    cache_entities = self.excel_cache[session_id].get("식사", [])
                    for cached_entity in cache_entities:
                        if not isinstance(cached_entity, dict):
                            continue
                        
                        # 중복 검사 기준: 날짜+끼니가 모두 있으면 그것으로, 없으면 메뉴로
                        # 날짜+끼니가 모두 있는 경우
                        if (filtered_value.get("날짜") and filtered_value.get("끼니") and 
                            cached_entity.get("날짜") == filtered_value.get("날짜") and
                            cached_entity.get("끼니") == filtered_value.get("끼니")):
                            logger.debug(f"식사 정보 중복 방지: '{filtered_value.get('날짜')} {filtered_value.get('끼니')}' 이미 존재 (세션: {session_id})")
                            merged_meal = self._merge_meal_entity(cached_entity, filtered_value)
                            # 캐시 업데이트
                            cache_entities.remove(cached_entity)
                            cache_entities.append(merged_meal)
                            return merged_meal
                        
                        # 메뉴만 있는 경우 (끼니/날짜가 없는 경우)
                        elif (filtered_value.get("메뉴") and cached_entity.get("메뉴") and
                              not filtered_value.get("끼니") and not cached_entity.get("끼니") and
                              filtered_value.get("메뉴") == cached_entity.get("메뉴")):
                            logger.debug(f"식사 정보 중복 방지: 메뉴 '{filtered_value.get('메뉴')}' 이미 존재 (세션: {session_id})")
                            merged_meal = self._merge_meal_entity(cached_entity, filtered_value)
                            # 캐시 업데이트
                            cache_entities.remove(cached_entity)
                            cache_entities.append(merged_meal)
                            return merged_meal
                        
                        # 시간만 있는 경우 - 기존 식사에 시간 추가
                        elif (filtered_value.get("시간") and not filtered_value.get("메뉴") and 
                              not filtered_value.get("끼니") and cached_entity.get("끼니")):
                            logger.debug(f"식사 시간 업데이트: 기존 식사에 시간 '{filtered_value.get('시간')}' 추가 (세션: {session_id})")
                            merged_meal = self._merge_meal_entity(cached_entity, filtered_value)
                            # 캐시 업데이트
                            cache_entities.remove(cached_entity)
                            cache_entities.append(merged_meal)
                            return merged_meal
            
            # strategy == "merge" - 엑셀 캐시 기반 (VectorStore 비활성화 대응)
            if strategy == "merge":
                # 엑셀 캐시에서 기존 엔티티 확인
                entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
                old_val = None
                
                if hasattr(self, 'excel_cache') and session_id in self.excel_cache:
                    cache_entities = self.excel_cache[session_id].get(entity_type, [])
                    # identity 기반으로 기존 엔티티 찾기
                    if identity and cache_entities:
                        for cached_entity in cache_entities:
                            if isinstance(cached_entity, dict) and str(identity) in str(cached_entity):
                                old_val = cached_entity
                                break
                    
                    if old_val:
                        # 사용자 정보 중복 방지 강화
                        if entity_key.endswith("사용자"):
                            # 동일한 이름이 이미 존재하는지 확인
                            if old_val.get("이름") and old_val.get("이름") == value.get("이름"):
                                # 동일한 이름이면 기존 정보 업데이트만 하고 새로 저장하지 않음
                                logger.debug(f"사용자 정보 중복 방지: '{value.get('이름')}' 이미 존재")
                                return self._merge_entity_values(old_val, value, "user.사용자")
                            
                            if old_val.get("이름") and old_val.get("이름") != value.get("이름"):
                                # 사용자 이름 충돌 - 재질문으로 처리
                                print(f"[WARN] 사용자 이름 충돌: 기존 '{old_val.get('이름')}' vs 새 '{filtered_value.get('이름')}' - 재질문으로 처리")
                                return old_val
                            # 기존 이름이 없거나 같으면 새 정보로 업데이트
                            filtered_value = self._merge_entity_values(old_val, filtered_value, entity_key)
                        
                        # 약 - 복용 정보 누적, 기간은 새로운 값 우선
                        elif entity_key.endswith("약"):
                            # 복용 정보 병합
                            combined_dosage = (old_val.get("복용") or []) + (filtered_value.get("복용") or [])
                        # 복용 정보에서 기간 정보 분리
                        filtered_dosage, extracted_period = self._extract_duration_from_dosage(combined_dosage)
                        filtered_value["복용"] = filtered_dosage
                        
                        # 복용 기간 우선순위: 추출된 기간 > 새로운 값 > 기존 값 (정규화 적용)
                        if extracted_period:
                            filtered_value["복용 기간"] = extracted_period
                        elif filtered_value.get("복용 기간"):
                            filtered_value["복용 기간"] = self._normalize_duration(filtered_value["복용 기간"])
                        elif old_val.get("복용 기간"):
                            filtered_value["복용 기간"] = self._normalize_duration(old_val["복용 기간"])
                        
                        # 새로운 머지 로직 적용
                        filtered_value = self._merge_entity_values(old_val, filtered_value, entity_key)
                    
                    # 일정 - 제목+날짜 기준으로 merge, 시간 필드 정규화
                    elif entity_key.endswith("일정"):
                        if old_val.get("제목") == filtered_value.get("제목") and old_val.get("날짜") == filtered_value.get("날짜"):
                            # 새로운 머지 로직 적용
                            filtered_value = self._merge_entity_values(old_val, filtered_value, entity_key)
                    
                    # 기념일 - 관계+제목+날짜 기준으로 merge
                    elif entity_key.endswith("기념일"):
                        if (old_val.get("관계") == filtered_value.get("관계") and 
                            old_val.get("제목") == filtered_value.get("제목") and 
                            old_val.get("날짜") == filtered_value.get("날짜")):
                            filtered_value = {**old_val, **filtered_value}
                    
                    # 식사 - 날짜+끼니 기준으로 merge, 메뉴는 누적, 시간 필드 정규화
                    elif entity_key.endswith("식사"):
                        if (old_val.get("날짜") == filtered_value.get("날짜") and 
                            old_val.get("끼니") == filtered_value.get("끼니")):
                            # 메뉴 누적 (중복 제거)
                            old_menus = old_val.get("메뉴", [])
                            new_menus = filtered_value.get("메뉴", [])
                            filtered_value["메뉴"] = list(set(old_menus + new_menus))
                            
                            # 시간 필드 정규화 및 업데이트 (새로운 시간이 있으면 우선)
                            if filtered_value.get("시간"):
                                # 새로운 시간 정보가 있으면 그것을 사용
                                filtered_value["시간"] = self._normalize_time_field(filtered_value.get("시간"))
                            elif old_val.get("시간"):
                                # 기존 시간 정보 유지
                                filtered_value["시간"] = self._normalize_time_field(old_val.get("시간"))
                            
                            filtered_value = {**old_val, **filtered_value}
                    
                    # 물건 - 이름 기준으로 merge, 위치는 최신/더 구체적으로 업데이트
                    elif entity_key.endswith("물건"):
                        if old_val.get("이름") == filtered_value.get("이름"):
                            # 위치가 더 구체적이면 업데이트
                            if filtered_value.get("위치") and (
                                not old_val.get("위치") or 
                                len(str(filtered_value.get("위치"))) > len(str(old_val.get("위치")))
                            ):
                                old_val["위치"] = filtered_value.get("위치")
                            filtered_value = {**old_val, **filtered_value}
                    
                    # 건강상태 - 증상 기준으로 merge, 정도는 더 심한 것으로 선택
                    elif entity_key.endswith("건강상태"):
                        if old_val.get("증상") == filtered_value.get("증상"):
                            # 정도가 더 심하면 업데이트
                            severity_order = {"경미": 1, "보통": 2, "심함": 3, "매우심함": 4}
                            old_sev = severity_order.get(old_val.get("정도"), 0)
                            new_sev = severity_order.get(filtered_value.get("정도"), 0)
                            if new_sev > old_sev:
                                old_val["정도"] = filtered_value.get("정도")
                            filtered_value = {**old_val, **filtered_value}
        except Exception as e:
            print("[WARN] merge 실패:", e)

        # 🔄 엑셀 백엔드로 저장 (VectorStore 비활성화 대응)
        session_id = metadata.get('session_id', 'default')
        try:
            user_name = self.user_names.get(session_id or "default")
            if not user_name or user_name == "사용자":
                print(f"[WARN] 사용자 이름이 설정되지 않아 엔티티를 저장할 수 없습니다. (session_id: {session_id})")
                return filtered_value
            
            # entity_key에서 entity_type 추출 (예: "user.취향" → "취향")
            entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
            
            # 엑셀 저장 핸들러 (버퍼링 구조 - 공유 인스턴스 사용)
            # save_entity_data()는 sheet_mapping.get(entity_type, "사용자정보KV")로 처리하므로
            # 매핑되지 않은 엔티티 타입도 사용자정보KV로 저장됨 (기타 시트 제거)
            self.excel_manager.save_entity_data(user_name, entity_type, filtered_value)
            
            # 세션 캐시 반영 (조회 성능 향상)
            if not hasattr(self, 'excel_cache'):
                self.excel_cache = {}
            session_cache = self.excel_cache.setdefault(session_id, {})
            
            # entity_type별 캐시 업데이트
            if entity_type not in session_cache:
                session_cache[entity_type] = []
            
            # 중복 체크 후 추가 (같은 identity가 없을 때만)
            existing = None
            for item in session_cache[entity_type]:
                if isinstance(item, dict):
                    # identity 기반 중복 체크 (간단한 문자열 매칭)
                    if identity and str(identity) in str(item):
                        existing = item
                        break
            
            if existing and strategy == "merge":
                # 기존 항목 업데이트
                merged = self._merge_entity_values(existing, filtered_value, entity_key)
                session_cache[entity_type] = [
                    merged if (isinstance(item, dict) and str(identity) in str(item)) else item
                    for item in session_cache[entity_type]
                ]
                filtered_value = merged
            elif not existing:
                # 새 항목 추가
                session_cache[entity_type].append(filtered_value)
            
            print(f"[INFO] 엑셀 백엔드로 엔티티 저장 완료: ({entity_type}) → {user_name}.xlsx")
        except Exception as e:
            print(f"[ERROR] _add_to_vstore 엑셀 저장 실패: {e}")
            import traceback
            traceback.print_exc()
        
        return filtered_value

    def _get_facts_text(self, session_id: str) -> str:
        """Excel 기반 사실 요약문"""
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

    # 엔티티 업서트 및 누락 필드에 대한 follow-up 질문 생성
    def _upsert_entities_and_get_confirms(self, session_id: str, entities: Dict[str, List[Dict[str, Any]]], user_input: str = None) -> Tuple[List[str], bool]:
        """엔티티 업서트 및 누락 필드에 대한 follow-up 질문 생성"""
        questions: List[str] = []
        has_schedule = False  # 일정 저장 여부 확인
        
        # 타입 안전성 체크
        if not isinstance(entities, dict):
            print(f"[ERROR] _upsert_entities_and_get_confirms: entities가 dict가 아님 {type(entities)}: {entities}")
            return [], False
        
        # 재질문 처리
        for entity_key, entity_list in entities.items():
            for entity in entity_list:
                if isinstance(entity, dict) and "질문" in entity:
                    print(f"[DEBUG] 재질문 처리 (상위): {entity['질문']}")
                    questions.append(entity["질문"])
                    return questions, has_schedule
        
        # 전역 재질문 체크
        if session_id in self.current_question:
            question = self.current_question[session_id]
            print(f"[DEBUG] _upsert_entities_and_get_confirms에서 전역 재질문 발견: {question}")
            questions.append(question)
            return questions, has_schedule

        # 정정 요청 감지 (사용자가 이미 답변했다고 명시한 경우)
        correction_keywords = ["이미 말했", "이미 말했는데", "이미 답했", "이미 답했는데", "아까 말했", "아까 말했는데"]
        is_correction = any(keyword in user_input for keyword in correction_keywords)
        
        # 모름/없음 응답 감지 (사용자가 모르거나 없다고 답한 경우)
        skip_keywords = ["모르겠", "없어", "몰라", "없다", "모름", "기억 안나", "기억안나", "잘 모르", "잘모르"]
        is_skip_response = any(keyword in user_input for keyword in skip_keywords)
        
        if is_correction:
            logger.debug("정정 요청 감지: 사용자가 이미 답변했다고 명시함")
            # 정정 요청인 경우 추가 질문 생성하지 않음
            return [], has_schedule
        
        if is_skip_response:
            logger.debug("모름/없음 응답 감지: 사용자가 모르거나 없다고 답함")
            # 모름/없음 응답인 경우 누락된 필드를 기본값으로 채우고 저장
            for entity_key, values in entities.items():
                for value in values:
                    try:
                        # N/A 값 필터링
                        filtered_value = self._filter_meaningful_data(value, user_input)
                        if not filtered_value:
                            continue
                        
                        # 재질문이 반환된 경우
                        if isinstance(filtered_value, dict) and "질문" in filtered_value:
                            print(f"[DEBUG] 재질문 처리: {filtered_value['질문']}")
                            questions.append(filtered_value["질문"])
                            return questions, has_schedule
                    except Exception as e:
                        if str(e).startswith("QUESTION:"):
                            question = str(e)[9:]  # "QUESTION:" 제거
                            print(f"[DEBUG] _upsert_entities_and_get_confirms에서 재질문 예외 처리: {question}")
                            questions.append(question)
                            return questions, has_schedule
                        raise e
                    
                    missing_fields = self._check_missing_fields(entity_key, filtered_value)
                    if missing_fields:
                        # 누락된 필드를 기본값으로 채우기
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
                        
                        # 수정된 엔티티 저장
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
            # print(f"[DEBUG] 엔티티 처리 시작: {entity_key} - {values}")
            
            # 사용자 엔티티의 경우 병합 로직 적용
            if entity_key == "user.사용자":
                # 기존 사용자 엔티티 조회
                existing_users = self._get_existing_user_entities(session_id)
                
                # 새 엔티티들과 기존 엔티티들 병합
                merged_users = existing_users
                for value in values:
                    filtered_value = self._filter_meaningful_data(value)
                    if not filtered_value:
                        continue
                    
                    if not self._is_valid_entity(entity_key, filtered_value):
                        continue
                    
                    # 병합 로직 적용
                    merged_users = self._merge_user_entities(merged_users, filtered_value)
                
                # 병합된 사용자 엔티티들을 개별적으로 처리
                for user_entity in merged_users:
                    # N/A 값 필터링
                    filtered_value = self._filter_meaningful_data(user_entity)
                    if not filtered_value:
                        continue
                    
                    # 엔티티 유효성 검사
                    if not self._is_valid_entity(entity_key, filtered_value):
                        continue
                    
                    # 사용자 엔티티 처리 계속...
                    self._process_single_entity(entity_key, filtered_value, session_id, questions, has_schedule)
            else:
                # 다른 엔티티들은 기존 로직 유지
                for value in values:
                    # N/A 값 필터링
                    filtered_value = self._filter_meaningful_data(value)
                    if not filtered_value:
                        print(f"[INFO] 의미없는 데이터 필터링: {entity_key} - {value}")
                        continue
                    
                    # 엔티티 유효성 검사
                    if not self._is_valid_entity(entity_key, filtered_value):
                        print(f"[INFO] 유효하지 않은 엔티티 스킵: {entity_key} - {filtered_value}")
                        continue
                        
                    # 엔티티 처리 계속...
                    self._process_single_entity(entity_key, filtered_value, session_id, questions, has_schedule)
                
                # 일정 저장 확인
                if entity_key.endswith("일정"):
                    has_schedule = True
                
                # 식사 엔티티에서 시간이 없으면 강제로 질문 추가 (필수 필드 체크에서 처리됨)
                
                # 약을 식사로 착각한 경우 제거
                if entity_key.endswith("식사") and "메뉴" in filtered_value:
                    menus = filtered_value["메뉴"]
                    if isinstance(menus, list):
                        # 약명이 포함된 메뉴 제거
                        filtered_menus = [menu for menu in menus if not menu.endswith("약")]
                        if not filtered_menus:
                            # 모든 메뉴가 약이면 이 식사 엔티티 제거
                            print(f"[INFO] 약을 식사로 착각한 엔티티 제거: {entity_key} - {filtered_value}")
                            continue
                        filtered_value["메뉴"] = filtered_menus
                
                # 필수 필드 체크
                missing_fields = self._check_missing_fields(entity_key, filtered_value)

                if missing_fields:
                    # 3️⃣ 필수 필드가 비면 follow-up 질문 생성 (저장은 보류)
                    logger.debug(f"누락된 필드 감지: {entity_key} - {missing_fields}, 값: {filtered_value}")
                    followup_questions = self._generate_followup_questions(entity_key, missing_fields, filtered_value)
                    questions.extend(followup_questions)
                    
                    # 식사 엔티티는 메뉴나 시간이 없어도 저장 (점진적 정보 수집)
                    if entity_key.endswith("식사") and (
                        ("메뉴" in missing_fields and filtered_value.get("메뉴") == []) or
                        ("시간" in missing_fields and not filtered_value.get("시간"))
                    ):
                        # 메뉴가 빈 리스트이거나 시간이 없는 경우에도 저장
                        final_value = self._add_to_vstore(
                            entity_key, filtered_value,
                            {"session_id": session_id, "entity_key": entity_key, "type": "entity", "created_at": datetime.now().isoformat()},
                            strategy="merge"
                        )
                        logger.debug(f"식사 엔티티 임시 저장 (메뉴/시간 누락): {filtered_value}")
                        
                        # 시간이 누락된 경우 재질문 상태 설정
                        if "시간" in missing_fields and not filtered_value.get("시간"):
                            self.pending_question[session_id] = {
                                "기존_엔티티": final_value,
                                "새_엔티티": final_value,
                                "entity_key": entity_key
                            }
                            print(f"[DEBUG] 재질문 상태 설정: {entity_key} - 시간 누락")
                    else:
                        continue

                # 2️⃣ 모든 필수 필드가 있으면 저장 (merge 정책 적용)
                final_value = self._add_to_vstore(
                    entity_key, filtered_value,
                    {"session_id": session_id, "entity_key": entity_key, "type": "entity", "created_at": datetime.now().isoformat()},
                    strategy="merge"
                )

                # 약은 복용 정보 세부 필드 확인 후 추가 질문
                if entity_key.endswith(".약"):
                    if final_value.get("복용"):
                        enriched = [self._enrich_dose_dict(d) for d in final_value["복용"]]
                        final_value["복용"] = enriched
                        
                        # enrich된 정보를 VectorStore에 업데이트
                        try:
                            self._add_to_vstore(
                                entity_key=entity_key,
                                value=final_value,
                                session_id=session_id,
                                metadata={"session_id": session_id, "type": "entity"},
                                strategy="merge",
                                identity=final_value.get("약명")
                            )
                            # print(f"[DEBUG] 복용 정보 enrich 후 VectorStore 업데이트 완료: {final_value.get('약명')}")
                        except Exception as e:
                            print(f"[WARN] 복용 정보 enrich 후 업데이트 실패: {e}")
                        
                        # 복용 정보가 이미 있는 경우 추가 질문하지 않음
                        # "복용" 필드가 있으면 충분한 정보로 간주
                        has_complete_info = True
                        
                        # 복용 정보가 완전하지 않은 경우에만 질문 (하지만 기본적인 복용 정보가 있으면 충분)
                        # if not has_complete_info:
                        #     questions.append(f"{final_value.get('약명','약')}은 하루에 몇 번 복용하나요?")

                    # 복용 정보가 없거나 불완전한 경우에만 질문
                    if not final_value.get("복용") and not final_value.get("식사와의 관계"):
                        questions.append(f"{final_value.get('약명','약')}은 언제, 하루 몇 번 복용하나요?")
                    # 복용 기간 질문은 제거 (이미 복용 정보가 있으면 충분)

        # 중복 제거
        return list(dict.fromkeys(questions)), has_schedule

    # 요약 저장 (세션 단위 격리)
    def save_final_summary(self, session_id: str):
        print(f"[DEBUG] save_final_summary 시작: session_id={session_id}")
        print(f"[DEBUG] auto_export_enabled: {self.cfg.auto_export_enabled}")
        
        # message_store에서 대화 기록 직접 조회
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
                
            # texts 생성 (message 컬럼에서 JSON 파싱하여 추출)
            texts = []
            for msg in messages:
                role = msg[2]  # role 컬럼
                content = msg[3]  # content 컬럼
                message = msg[4]  # message 컬럼
                
                # content가 있으면 사용, 없으면 message에서 JSON 파싱
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
        
        # 자동 추출 실행
        if self.cfg.auto_export_enabled:
            print(f"[DEBUG] 자동 추출 시작: session_id={session_id}")
            try:
                self.export_conversation_to_excel(session_id)
                print(f"[INFO] 대화 기록이 엑셀 파일로 저장되었습니다: conversation_extract/{session_id}.xlsx")
            except Exception as e:
                print(f"[ERROR] 엑셀 파일 생성 실패: {e}")
        
        # 사용자 이름 확정값 가져오기 (hallucination 방지)
        confirmed_name = self._get_confirmed_user_name(session_id)
        
        # 감정 상태 정보 추가
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
        
        # 세션별 요약 생성 (단순화된 프롬프트)
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
        
        # JSON 변수 이스케이프 처리
        escaped_texts = []
        for text in texts:
            # JSON 형태의 텍스트에서 중괄호 이스케이프
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
        
        # 항상 새로운 요약을 삽입 (누적 방식)
        c.execute("INSERT INTO conversation_summary (session_id, summary, created_at, updated_at) VALUES (?, ?, ?, ?)", 
                 (session_id, summary_text, datetime.now().isoformat(), datetime.now().isoformat()))
        # print(f"[DEBUG] SQLite 요약 저장 완료 (세션: {session_id}) - 누적 방식")
        
        conn.commit()
        conn.close()
        
        # 자동 추출 실행
        print(f"[DEBUG] 자동 추출 시작: session_id={session_id}, auto_export_enabled={self.cfg.auto_export_enabled}")
        self.auto_export_conversation(session_id)

    # 대화 시스템 체인
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
        """Excel 기반 약 목록 조회"""
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

    # 출력 포맷 (2차 목표: 일정/약/식사도 자연스러운 문장화)
    def _format_entities_for_output(self, user_input: str, ents: List[Document], session_id: str = "default") -> str:
        if not ents:
            # 사용자 이름이 있으면 사용, 없으면 기본 호칭
            user_name = self._get_confirmed_user_name(session_id)
            if user_name and user_name != "사용자":
                return f"아직 그 정보는 몰라요, {user_name}님. 알려주시면 기억해둘게요!"
            else:
                return "아직 그 정보는 몰라요. 알려주시면 기억해둘게요!"
        lines = []
        for d in ents:
            try:
                val = json.loads(d.page_content)
                # print(f"[DEBUG] 문서 처리: {val}")
            except Exception as e:
                print(f"[DEBUG] JSON 파싱 실패: {e}")
                continue
            etype = d.metadata.get("entity_key", "")
            # print(f"[DEBUG] 엔티티 타입: {etype}")
            
            # 사용자
            if etype.endswith("사용자") and val.get("이름"):
                # print(f"[DEBUG] 사용자 이름 추가: {val['이름']}")
                lines.append(f"네, {val['이름']}님이에요.")
            
            # 물건
            elif etype.endswith("물건"):
                if val.get("위치"):
                    # 저장 시점에서 이미 정규화되었으므로 단순히 "에 있어요"만 붙임
                    lines.append(f"{val.get('이름')}은 {val.get('위치')}에 있어요.")
                else:
                    lines.append(f"{val.get('이름')}에 대해 알고 있어요.")
            
            # 일정
            elif etype.endswith("일정"):
                title = val.get("제목", "일정")
                date = val.get("날짜", "")
                time = val.get("시간", "")
                location = val.get("장소", "")
                
                # 날짜 정규화 적용 (기존 데이터도 정규화)
                if date:
                               date = self._normalize_date(date, session_id)
                
                parts = [title]
                if date:
                    parts.append(f"{date}에")
                if time:
                    # 시간이 리스트인 경우 문자열로 변환
                    if isinstance(time, list):
                        time = ', '.join(time)
                    parts.append(f"{time}에")
                if location:
                    parts.append(f"{location}에서")
                
                lines.append(" ".join(parts) + " 예정이에요.")
            
            # 약
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
            
            # 식사
            elif etype.endswith("식사"):
                meal = val.get("끼니", "")
                menus = val.get("메뉴", [])
                date = val.get("날짜", "")
                time = val.get("시간", "")
                
                # 날짜 정규화 적용 (기존 데이터도 정규화)
                if date:
                               date = self._normalize_date(date, session_id)
                
                parts = []
                if date:
                    parts.append(f"{date}")
                if meal:
                    parts.append(f"{meal}에")
                if time:
                    # 시간이 리스트인 경우 문자열로 변환
                    if isinstance(time, list):
                        time = ', '.join(time)
                    parts.append(f"{time}에")
                if menus:
                    parts.append(f"{', '.join(menus)}을 드셨어요.")
                
                if parts:
                    lines.append(" ".join(parts))
                else:
                    lines.append("식사 정보를 알고 있어요.")
            
            # 가족
            elif etype.endswith("가족"):
                relation = val.get("관계", "")
                name = val.get("이름", "")
                if relation and name:
                    lines.append(f"{relation} {name}님에 대해 알고 있어요.")
                elif relation:
                    lines.append(f"{relation}에 대해 알고 있어요.")
            
            # 동적 관계 엔티티 처리 (하드코딩 제거)
            # 기본 엔티티 타입이 아닌 경우 동적으로 처리
            elif not etype.endswith(("사용자", "물건", "일정", "약", "식사", "가족", "기념일", "취미", "취향", "건강상태")):
                name = val.get("이름", "")
                relation_type = etype.replace("user.", "")
                if name:
                    lines.append(f"{relation_type} {name}님에 대해 알고 있어요.")
                else:
                    lines.append(f"{relation_type}에 대해 알고 있어요.")
            
            # 기념일
            elif etype.endswith("기념일"):
                title = val.get("제목", "")
                date = val.get("날짜", "")
                relation = val.get("관계", "")
                
                # 날짜 정규화 적용 (기존 데이터도 정규화)
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
            
            # 건강상태
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
            
            # 취미
            elif etype.endswith("취미"):
                hobby = val.get("이름", "")
                if hobby:
                    lines.append(f"{hobby} 취미를 가지고 계시는군요.")
            
            # 취향
            elif etype.endswith("취향"):
                category = val.get("종류", "")
                value = val.get("값", "")
                if category and value:
                    lines.append(f"{category}에서 {value}을 좋아하시는군요.")
                elif value:
                    lines.append(f"{value}을 좋아하시는군요.")
        
        # print(f"[DEBUG] _format_entities_for_output 최종 lines: {lines}")
        result = " ".join(lines) if lines else "아직 그건 몰라요."
        # print(f"[DEBUG] _format_entities_for_output 최종 결과: {result}")
        return result

    # generate
    def process_user_input(self, user_text: str, session_id: str = "default") -> str:
        """사용자 입력 처리 메인 함수"""
        print(f"[DEBUG] process_user_input 호출됨: '{user_text}'")
        try:
            # ✅ 0. 세션 초기화 (첫 발화인 경우 이전 대화 요약 불러오기)
            if not hasattr(self, '_session_initialized'):
                self._session_initialized = set()
            
            if session_id not in self._session_initialized:
                print(f"[DEBUG] 세션 {session_id} 초기화 완료")
                self._session_initialized.add(session_id)
            
            # ✅ 1. pending_question 확인 → 재질문 답변 처리
            if session_id in self.pending_question:
                print(f"[DEBUG] pending_question 발견: {self.pending_question[session_id]}")
                # ✅ 확인 응답인지 먼저 체크 (안전 가드)
                import re
                yes_pattern = re.compile(r"^(응|네|좋아|그래|ㅇㅇ|웅|맞아)\s*$", re.IGNORECASE)
                no_pattern = re.compile(r"^(아니|괜찮아|됐어|ㄴㄴ|싫어)\s*$", re.IGNORECASE)
                
                if yes_pattern.match(user_text.strip()) or no_pattern.match(user_text.strip()):
                    followup = handle_pending_answer(user_text, self, session_id)
                    if followup:
                        return followup
                else:
                    print(f"[DEBUG] pending_question이 있지만 확인 응답이 아님: '{user_text}' - 일반 처리로 진행")

            # ✅ 2. 분류 (유사 캐싱 우선 적용) - 엔티티 추출은 각 핸들러에서 처리
            print(f"[DEBUG] 분류 시작")
            
            # 먼저 유사 캐시 확인
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
                # 캐시에 없으면 LLM 호출 (엔티티 없이 분류만)
                from .task_classifier import classify_hybrid
                result = classify_hybrid(user_text, None)  # 엔티티는 None으로 전달
                print(f"[DEBUG] LLM 분류 결과: '{user_text}' -> {result.category} (신뢰도: {result.confidence:.2f})")
                
                # 결과를 캐시에 저장
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

        # ✅ 4. 카테고리별 처리
        print(f"[DEBUG] 카테고리별 처리 시작: {result.category}")
        
        # LCEL 체인에 사용자 메시지 저장
        self.conversation_memory.chat_memory.add_user_message(user_text)
        
        if result.category == "cognitive":
            print(f"[DEBUG] cognitive 처리 호출")
            try:
                from .support_chains import handle_cognitive_task_with_lcel
                response = handle_cognitive_task_with_lcel(user_text, self, session_id)
                # LCEL 체인에 AI 응답 저장
                self.conversation_memory.chat_memory.add_ai_message(response)
                # 요약 생성은 세션 종료 시에만 수행
                return response
            except Exception as e:
                import traceback
                print(f"[ERROR] cognitive 처리 실패: {traceback.format_exc()}")
                # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
                error_response = "죄송해요, 처리 중 오류가 있었어요. 다시 한 번 말씀해 주시겠어요?"
                # LCEL 체인에 AI 응답 저장
                self.conversation_memory.chat_memory.add_ai_message(error_response)
                return error_response
        elif result.category == "emotional":
            print(f"[DEBUG] emotional 처리 호출")
            try:
                response = self._handle_emotional_task(user_text, session_id)
                # LCEL 체인에 AI 응답 저장
                self.conversation_memory.chat_memory.add_ai_message(response)
                # SQLite 백엔드가 자동으로 저장하므로 별도 저장 불필요
                # 요약 생성은 세션 종료 시에만 수행
                return response
            except Exception as e:
                import traceback
                print(f"[ERROR] emotional 처리 실패: {traceback.format_exc()}")
                # 사용자 친화적 한글 메시지 (rqt fix 적용 시 안전하게 처리됨)
                error_response = "지금 많이 힘드셨죠. 곁에서 같이 이야기 들어드릴게요. 어떤 점이 가장 힘들었나요?"
                # LCEL 체인에 AI 응답 저장
                self.conversation_memory.chat_memory.add_ai_message(error_response)
                return error_response
        elif result.category == "physical":
            print(f"[DEBUG] physical 처리 호출")
            
            response = handle_physical_task(user_text, self, session_id)
            
            # 딕셔너리 응답 처리
            if isinstance(response, dict):
                message = response.get("message", str(response))
                # LCEL 체인에 AI 응답 저장 (문자열만)
                self.conversation_memory.chat_memory.add_ai_message(message)
                # 히스토리 저장: LCEL 메모리만 사용 (SQLite 비사용)
                # 요약 생성은 세션 종료 시에만 수행
                return response  # 전체 딕셔너리 반환
            else:
                # 문자열 응답
                self.conversation_memory.chat_memory.add_ai_message(response)
                # SQLite 백엔드가 자동으로 저장하므로 별도 저장 불필요
                # 요약 생성은 세션 종료 시에만 수행
                return response
        elif result.category == "query":
            print(f"[DEBUG] query 처리 호출")
            from .support_chains import handle_query_with_lcel
            response = handle_query_with_lcel(user_text, self, session_id)
            # LCEL 체인에 AI 응답 저장
            self.conversation_memory.chat_memory.add_ai_message(response)
            # SQLite 백엔드가 자동으로 저장하므로 별도 저장 불필요
            # 요약 생성은 세션 종료 시에만 수행
            return response
        else:
            print(f"[DEBUG] 알 수 없는 카테고리: {result.category}")
            return "죄송해요, 잘 이해하지 못했어요."

    
    def _handle_emotional_task(self, user_text: str, session_id: str) -> str:
        """정서적 작업 처리 - 인사, 감정 표현, 날씨, 시간 등"""
        try:
            # 1️⃣ 중복 응답 처리 체크
            if hasattr(self, 'pending_question') and self.pending_question.get(session_id):
                pending_data = self.pending_question[session_id]
                print(f"[DEBUG] 중복 응답 처리 (emotional): {user_text}")
                result = self.handle_duplicate_answer(user_text, pending_data)
                
                # 응답 처리 완료 후 pending_question 제거
                if session_id in self.pending_question:
                    del self.pending_question[session_id]
                
                return result["message"]
            
            # LCEL ConversationBuffer에서 대화 맥락 로드
            memory_vars = self.conversation_memory.load_memory_variables({})
            conversation_history = memory_vars.get('history', '')
            print(f"[DEBUG] Emotional LCEL history 길이: {len(conversation_history)}")
            
            # 메시지 저장 (직접 message_store에 저장)
            self._save_message(session_id, "human", user_text)
            
            # conversation_history를 문자열로 변환
            conversation_history = self._convert_conversation_history_to_string(conversation_history)
            
            # 엔티티 추출 및 저장 (Slot-filling 체크 포함)
            entities = self._pre_extract_entities(user_text, session_id)
            print(f"[DEBUG] emotional에서 추출된 엔티티: {entities}")
            
            # Slot-filling 응답 처리
            if isinstance(entities, dict) and entities.get("success") == False and entities.get("incomplete"):
                print(f"[DEBUG] Slot-filling 필요 (emotional): {entities['message']}")
                # pending_question에 저장
                self.pending_question[session_id] = entities.get("pending_data", {})
                return entities["message"]
            
            if entities:
                # 사용자 이름 엔티티 처리
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
                                # 중복 발견 시 pending_question에 저장
                                self.pending_question[session_id] = save_result.get("pending_data", {})
                                return save_result["message"]
                
                # 감정 엔티티 처리
                if "user.건강상태" in entities:
                    for emotion in entities["user.건강상태"]:
                        emotion_state = emotion.get("증상", "")
                        if emotion_state:
                            # 감정을 VectorStore에 JSON 구조로 저장 (정서 타입으로 통일)
                            # ✅ 감정의 원인/상황을 간결하게 요약
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
                                # 중복 발견 시 pending_question에 저장
                                self.pending_question[session_id] = save_result.get("pending_data", {})
                                return save_result["message"]
                            else:
                                print(f"[DEBUG] 정서 저장됨: {emotion_state}")
            
            # LCEL history를 고려한 감정적 응답 생성
            if conversation_history and len(conversation_history.strip()) > 0:
                # 대화 맥락을 고려한 감정적 응답
                from .support_chains import build_emotional_reply
                user_name_confirmed = bool(self._get_confirmed_user_name(session_id))
                response = build_emotional_reply(user_text, llm=self.llm, user_name_confirmed=user_name_confirmed)
            else:
                # 기본 감정적 응답
                from .support_chains import build_emotional_reply
                user_name_confirmed = bool(self._get_confirmed_user_name(session_id))
                response = build_emotional_reply(user_text, llm=self.llm, user_name_confirmed=user_name_confirmed)
            
            # AIMessage 객체를 문자열로 변환
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # 응답 저장 (직접 message_store에 저장)
            self._save_message(session_id, "ai", response_text)
            return response_text
            
        except Exception as e:
            print(f"[ERROR] 정서적 작업 처리 실패: {e}")
            return "죄송해요, 처리 중 오류가 발생했어요."
    
    def _extract_appointment_info(self, user_text: str) -> str:
        """
        사용자의 발화에서 날짜, 시간, 장소(치과, 병원, 미용실 등)를 추출
        """
        try:
            import re
            from datetime import datetime, timedelta
            
            info = {"date": None, "time": None, "place": None}

            # 장소
            place_match = re.search(r"(치과|병원|미용실|약속|회의|미팅|약국|은행|카페|식당)", user_text)
            if place_match:
                info["place"] = place_match.group(1)

            # 상대적 날짜
            if "내일" in user_text:
                info["date"] = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
            elif "모레" in user_text:
                info["date"] = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
            else:
                # "다음 주 토요일" 패턴
                dow_match = re.search(r"(다음\s*주\s*)(월|화|수|목|금|토|일)요일?", user_text)
                if dow_match:
                    weekday_map = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
                    target_weekday = weekday_map[dow_match.group(2)]
                    today = datetime.now()
                    days_ahead = (target_weekday - today.weekday() + 7) % 7
                    if days_ahead == 0:
                        days_ahead = 7
                    days_ahead += 7  # 다음 주
                    target_date = today + timedelta(days=days_ahead)
                    info["date"] = target_date.strftime("%Y-%m-%d")

            # 시간 (오전/오후 hh시)
            time_match = re.search(r"(오전|오후)?\s?(\d{1,2})시", user_text)
            if time_match:
                hour = int(time_match.group(2))
                if time_match.group(1) == "오후" and hour < 12:
                    hour += 12
                info["time"] = f"{hour:02d}:00"

            # 결과 조합
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
        """Excel 캐시 기반 엔티티 조회"""
        if not hasattr(self, "excel_cache"):
            return []
        sess = self.excel_cache.get(session_id, {})
        entities = []
        for entity_type in entity_types:
            # entity_types가 "user.약" 형태일 수 있으므로 "약"만 추출
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
        """최근 대화 메시지 가져오기"""
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
        """message_store에 메시지 저장 (SQLChatMessageHistory 호환 JSON 형식)"""
        try:
            import sqlite3
            import json
            conn = sqlite3.connect(self.sqlite_path)
            c = conn.cursor()
            
            # content 타입 변환 (문자열이 아닌 경우 문자열로 변환)
            if not isinstance(content, str):
                if isinstance(content, (list, tuple)):
                    content = str(content)
                elif hasattr(content, '__str__'):
                    content = str(content)
                else:
                    content = repr(content)
            
            # SQLChatMessageHistory 호환 JSON 형식으로 저장
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
        """기존 generate 함수 - process_user_input 호출"""
        # process_user_input 호출
        response = self.process_user_input(user_input, session_id)
        
        # 메시지 저장은 process_user_input에서 처리됨
        
        return response
    
    def _update_existing_entity(self, session_id: str, entity_key: str, existing_entity: dict, new_entity: dict):
        """기존 엔티티를 새 엔티티로 교체 (Excel 기반)"""
        if False and self.vectorstore:
            # 기존 엔티티 삭제 (서드 파티 VectorStore 코드 - 비활성화됨)
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
        
        # 새 엔티티 저장 (Excel 기반)
        entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
        user_name = self.user_names.get(session_id or "default")
        if user_name:
            self.excel_manager.save_entity_data(user_name, entity_type, new_entity)
            # 캐시 업데이트
            if hasattr(self, "excel_cache"):
                sess = self.excel_cache.setdefault(session_id, {})
                entities = sess.setdefault(entity_type, [])
                # 기존 엔티티 찾아서 교체
                for i, ent in enumerate(entities):
                    if ent.get("이름") == existing_entity.get("이름"):
                        entities[i] = new_entity
                        break
                else:
                    entities.append(new_entity)
    
    def _add_new_entity(self, session_id: str, entity_key: str, new_entity: dict):
        """새 엔티티를 추가로 저장"""
        self._store_entity_direct(session_id, entity_key, new_entity)
    
    def _cancel_schedule(self, session_id: str, title: str):
        """일정 취소 처리 - Excel에서 해당 일정 삭제"""
        try:
            user_name = self.user_names.get(session_id or "default")
            if not user_name:
                return False
            df = self.excel_manager.load_sheet_data(user_name, "일정")
            if df is None or df.empty:
                return False
            # 제목이 일치하는 행 제거
            filtered_df = df[df["제목"] != title]
            if len(filtered_df) < len(df):
                # 엑셀에 저장 (삭제 반영) - 즉시 저장 필요
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
                # 캐시 업데이트
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
        """엔티티를 Excel에 직접 저장"""
        try:
            entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
            user_name = self.user_names.get(session_id or "default")
            if not user_name:
                return False
            self.excel_manager.save_entity_data(user_name, entity_type, entity)
            # 캐시 업데이트
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
        """Excel의 엔티티 업데이트"""
        try:
            entity_type = entity_key.replace("user.", "") if entity_key.startswith("user.") else entity_key
            user_name = self.user_names.get(session_id or "default")
            if not user_name:
                return False
            # 시트 매핑 확인
            sheet_mapping = {
                "물건": "물건위치",
                "약": "복약정보",
                "일정": "일정",
                "음식": "음식기록",
                "정서": "감정기록",
                "가족": "가족관계",
            }
            sheet_name = sheet_mapping.get(entity_type, "사용자정보KV")
            # Excel에서 기존 엔티티 찾아서 업데이트
            df = self.excel_manager.load_sheet_data(user_name, sheet_name)
            if df is not None and not df.empty:
                # 이름으로 찾아서 업데이트
                name = updated_entity.get("이름", "")
                if name and "이름" in df.columns:
                    idx = df[df["이름"] == name].index
                    if len(idx) > 0:
                        # 기존 행 업데이트
                        for col in updated_entity.keys():
                            if col in df.columns:
                                df.loc[idx[0], col] = updated_entity[col]
                        # 엑셀에 저장
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
                        # 캐시 업데이트
                        if hasattr(self, "excel_cache"):
                            sess = self.excel_cache.setdefault(session_id, {})
                            entities = sess.setdefault(entity_type, [])
                            for i, ent in enumerate(entities):
                                if ent.get("이름") == name:
                                    entities[i] = updated_entity
                                    return True
            # 기존 엔티티를 찾지 못한 경우 새로 저장
            self.excel_manager.save_entity_data(user_name, entity_type, updated_entity)
            # 캐시 업데이트
            if hasattr(self, "excel_cache"):
                sess = self.excel_cache.setdefault(session_id, {})
                entities = sess.setdefault(entity_type, [])
                entities.append(updated_entity)
            return True
        except Exception as e:
            print(f"[ERROR] 엔티티 업데이트 실패: {e}")
            return False
    
    def auto_export_conversation(self, session_id: str):
        """대화 자동 추출 (엑셀 파일)"""
        if not self.cfg.auto_export_enabled:
            return None
        
        try:
            # 엑셀 파일 생성
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
        """대화 기록을 엑셀 파일로 추출 (SQLite 직접 조회)"""
        try:
            print(f"[DEBUG] export_conversation_to_excel 시작: {session_id}")
            
            # SQLite에서 직접 대화 기록 조회
            import sqlite3
            import pandas as pd
            from datetime import datetime
            import os
            import json
            
            conn = sqlite3.connect(self.sqlite_path)
            cur = conn.cursor()
            
            # 메시지 조회 (created_at 컬럼이 없을 경우를 대비한 fallback)
            try:
                cur.execute(
                    "SELECT id, session_id, role, content, created_at FROM message_store WHERE session_id = ? ORDER BY id",
                    (session_id,)
                )
                rows = cur.fetchall()
            except sqlite3.OperationalError as e:
                if "no such column: created_at" in str(e):
                    # created_at 컬럼이 없으면 id를 시간으로 사용
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
            
            # 데이터 준비
            data = []
            print(f"[DEBUG] 메시지 수: {len(rows)}")
            for row in rows:
                msg_id, session_id, role, content, created_at = row
                content = content or ""  # None 처리
                print(f"[DEBUG] 메시지 {msg_id}: role={role}, content={content[:50]}...")
                
                # JSON 파싱
                try:
                    if content.startswith('{"type":'):
                        msg_data = json.loads(content)
                        actual_type = msg_data.get("type", "unknown")
                        actual_content = msg_data.get("data", {}).get("content", content)
                        
                        # 발화자 설정 (JSON에서 추출)
                        if actual_type == "human":
                            display_role = "사용자"
                        elif actual_type == "ai":
                            display_role = "AI"
                        else:
                            display_role = "unknown"
                    else:
                        actual_content = content
                        # 발화자 설정 (role 컬럼에서 추출)
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
            
            # DataFrame 생성
            df = pd.DataFrame(data)
            
            # 엑셀 파일 저장
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

    # ✅ 대화 내용 기록용 add_dialog 메서드 추가
    def add_dialog(self, text: str, act_type: str):
        """
        사용자 발화(text)와 act_type(인지/정서/물리)을 메모리에 기록.
        """
        try:
            # ConversationBufferMemory 사용
            self.conversation_memory.save_context({"input": text}, {"output": act_type})

            # SQLite 또는 summary memory에도 반영
            if hasattr(self, "summary_memory"):
                self.summary_memory.save_context({"input": text}, {"output": act_type})

            print(f"[MEMORY] Added dialog: ({act_type}) {text}")

        except Exception as e:
            print(f"[MEMORY ERROR] add_dialog(): {e}")
    
    # -----------------------------
    # 🧩 flush 메서드 (버퍼 → Excel 반영)
    # -----------------------------
    def flush_memory_to_excel(self, session_id: str):
        """세션 캐시 → Excel 반영 (버퍼 플러시)"""
        try:
            user_name = self.user_names.get(session_id or "default_session", "사용자")
            if user_name and user_name != "사용자":
                # 버퍼 비어있으면 skip
                try:
                    buffered = getattr(self.excel_manager, "_buffered_changes", {})
                    has_user_buffers = any(k for k in buffered.keys() if k[0] == user_name)
                    if not has_user_buffers:
                        logger.info("[FLUSH] 버퍼 비어있음 - flush 생략")
                        return
                except Exception:
                    pass
                #  request_flush() 사용하여 지연 병합 처리 (race condition 방지)
                self.excel_manager.request_flush(user_name)
                logger.info(f"[FLUSH] 세션({session_id}) 데이터 엑셀로 동기화 예약 ({user_name})")
            else:
                logger.warning(f"[FLUSH] 사용자 이름이 없어 flush 건너뜀: {session_id}")
        except Exception as e:
            logger.error(f"[ERROR] flush_memory_to_excel 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
