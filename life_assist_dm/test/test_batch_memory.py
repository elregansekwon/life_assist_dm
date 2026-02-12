import sys
import os
import pandas as pd
from datetime import datetime

# ✅ 프로젝트 루트 추가 (패키지 인식 보장용)
sys.path.append('/home/keti/workspace/life_assist_dm_copy')

# ✅ 중첩 구조에 맞는 import
from life_assist_dm.llm.memory import LifeAssistMemory, MemoryConfig

def _safe_sid(session_id, fallback):
    """NaN이나 빈 값 처리"""
    s = ("" if session_id is None else str(session_id)).strip()
    return s if s and s.lower() != "nan" else fallback

def run_batch(input_excel: str, output_dir: str):
    cfg = MemoryConfig(
        sqlite_path="/tmp/test_life_assist_memory.db",
        chroma_dir="/tmp/test_chroma_db",
        auto_export_enabled=True,
        export_dir=output_dir
    )
    memory = LifeAssistMemory(cfg)

    # 엑셀 불러오기
    df = pd.read_excel(input_excel)

    # 배치 전체 공통 세션 ID 생성
    batch_sid = f"physical160_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"[INFO] 배치 세션 ID: {batch_sid}")

    results = []
    for idx, row in df.iterrows():
        user_text = str(row["user"])  # ✅ 컬럼명: user
        
        # ✅ 동일한 세션ID 사용 (첫 번째 세션과 동일)
        session_id = f"{batch_sid}_000"
        print(f"[DEBUG] 처리 중: session_id={session_id}, user_text={user_text}")
        
        try:
            # ✅ 세션별 메모리 히스토리 초기화
            from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
            memory.conversation_memory.chat_memory = SQLChatMessageHistory(
                session_id=session_id,
                connection_string=f"sqlite:///{cfg.sqlite_path}"
            )
            
            # ✅ 엔진 호출 (내부에서 메시지 저장됨)
            response = memory.process_user_input(user_text, session_id=session_id)
            
        except Exception as e:
            error_msg = f"[ERROR] {e}"
            # 에러 시에만 별도 저장
            memory.conversation_memory.chat_memory.add_ai_message(error_msg)
            response = error_msg

        results.append({
            "id": row["id"],
            "intent": row["intent"],
            "user_text": user_text,
            "response": response
        })
        print(f"[{idx}] 사용자: {user_text} → AI: {response}")

    # 결과 DataFrame 저장
    result_df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "physical160.xlsx")
    result_df.to_excel(out_path, index=False)
    print(f"\n✅ 결과 저장 완료: {out_path}")

    # ✅ 배치 세션으로 요약/엔티티 DB에 반영
    # 최종 요약 및 내보내기 (동일한 세션ID 사용)
    final_session_id = f"{batch_sid}_000"
    print(f"[DEBUG] 최종 요약 저장: session_id={final_session_id}")
    
    # ✅ 대화 기록 확인 후 처리
    try:
        from langchain_community.chat_message_histories.sql import SQLChatMessageHistory
        history = SQLChatMessageHistory(session_id=final_session_id, connection_string="sqlite:///memory.db")
        conversation = history.messages if hasattr(history, "messages") else []
        
        if not conversation:
            print(f"[WARN] No conversation logs for session {final_session_id}")
        else:
            print(f"[INFO] Found {len(conversation)} conversation messages")
            
        # 요약 저장 및 엑셀 내보내기 진행
        memory.save_final_summary(session_id=final_session_id)
        memory.export_conversation_to_excel(session_id=final_session_id)
        
    except Exception as e:
        print(f"[ERROR] Conversation check failed: {e}")
        # 에러가 있어도 요약 저장은 진행
        memory.save_final_summary(session_id=final_session_id)
        memory.export_conversation_to_excel(session_id=final_session_id)

if __name__ == "__main__":
    input_excel = "/home/keti/workspace/life_assist_dm_copy/physical160.xlsx"
    output_dir = "/home/keti/workspace/life_assist_dm_copy/conversation_extract"
    run_batch(input_excel, output_dir)

