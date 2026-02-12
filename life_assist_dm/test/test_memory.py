# test/test_memory.py
import sys
import os
import logging
import warnings
sys.path.append('/home/keti/workspace/life_assist_dm_copy')

# 불필요한 로그들 비활성화
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)
logging.getLogger("langchain_core").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# 모든 경고 비활성화
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

from life_assist_dm.llm.memory import LifeAssistMemory, MemoryConfig
import json

def run_repl():
    print("=== LifeAssistMemory REPL ===")
    
    # 고정된 세션 ID 사용 (대화 기록 유지)
    session_id = "test_session"
    print(f"[INFO] 세션 ID: {session_id}")
    
    cfg = MemoryConfig(
        sqlite_path="/tmp/test_life_assist_memory.db",
        chroma_dir="/tmp/test_chroma_db",
        auto_export_enabled=True,
        export_dir="conversation_extract"
    )
    # VectorStore 유지 (삭제하지 않음)
    memory = LifeAssistMemory(cfg, session_id=session_id)

    while True:
        try:
            q = input("사용자> ").strip()
        except UnicodeDecodeError:
            print("[WARN] 입력 인코딩 문제 발생, 다시 입력해주세요.")
            continue
        except EOFError:
            print("\n[INFO] 입력이 종료되었습니다. 대화를 종료합니다.")
            break

        if q in ("exit", "quit"):
            break

        a = memory.generate(session_id, q)
        
        # 응답을 깔끔하게 표시
        if hasattr(a, 'content'):
            clean_response = a.content
        else:
            clean_response = str(a)
        print("AI>", clean_response)

    # ===== 대화 종료 후 요약 & 엔티티 출력 =====
    print("\n=== SQLite 저장 확인 ===")
    print("[DEBUG] save_final_summary 호출 시작")
    memory.save_final_summary(session_id)
    print("[DEBUG] save_final_summary 호출 완료")
    
    # 엑셀 파일 생성 확인
    print("\n=== 엑셀 파일 생성 확인 ===")
    excel_path = memory.export_conversation_to_excel(session_id)
    if excel_path:
        print(f"✅ 엑셀 파일 생성됨: {excel_path}")
    else:
        print("❌ 엑셀 파일 생성 실패")
    
    # 실제 실행 세션의 요약을 조회
    summaries = memory._get_recent_summaries(session_id)
    print("\n[최근 요약]")
    if summaries:
        for i, summary in enumerate(summaries):
            print(f"  {i+1}. {summary}")
    else:
        print("  (요약 없음)")

    print("\n=== VectorStore 전체 저장소 확인 ===")
    docs = memory.vectorstore.get()
    ids = docs.get("ids", [])
    documents = docs.get("documents", [])
    metas = docs.get("metadatas", [])
    print("총 저장 개수:", len(ids))
    for i, (doc, meta) in enumerate(zip(documents, metas)):
        try:
            parsed = json.loads(doc)
            doc_str = json.dumps(parsed, ensure_ascii=False)
        except Exception:
            doc_str = doc
        print(f"[{i}] {doc_str} | {meta}")


if __name__ == "__main__":
    run_repl()
