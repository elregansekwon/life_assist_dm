from dotenv import load_dotenv
import os
from pathlib import Path


# API KEY 정보로드
# 현재 파일의 절대 경로에서 시작
current_file = Path(__file__).resolve()

# llm 디렉토리에서 2단계 위로 올라가서 life_assist_dm 패키지 디렉토리 찾기
package_root = current_file.parent.parent
env_file = package_root / ".env"

if env_file.exists():
    load_dotenv(env_file)
    print(f"[INFO] .env 파일 로드 완료: {env_file}")
else:
    # fallback: llm 디렉토리 내 .env 파일
    llm_env = current_file.parent / ".env"
    if llm_env.exists():
        load_dotenv(llm_env)
        print(f"[INFO] .env 파일 로드 완료 (llm 디렉토리): {llm_env}")
    else:
        print(f"[WARNING] .env 파일을 찾을 수 없습니다. 시도한 경로: {env_file}, {llm_env}")