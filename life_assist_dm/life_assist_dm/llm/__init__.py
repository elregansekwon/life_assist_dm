from dotenv import load_dotenv
import os
from pathlib import Path


def _load_env():
    """.env 파일을 다음 순서로 탐색하여 로드: 패키지 루트 → llm/ → ROS2 share 디렉터리."""
    current_file = Path(__file__).resolve()
    # llm 디렉토리에서 2단계 위로 올라가서 life_assist_dm 패키지 디렉토리 찾기
    package_root = current_file.parent.parent
    env_file = package_root / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        print(f"[INFO] .env 파일 로드 완료: {env_file}")
        return True

    llm_env = current_file.parent / ".env"
    if llm_env.exists():
        load_dotenv(llm_env)
        print(f"[INFO] .env 파일 로드 완료 (llm 디렉토리): {llm_env}")
        return True

    # ROS2 install 시 .env는 share/life_assist_dm/ 에 복사됨
    try:
        from ament_index_python.packages import get_package_share_directory
        share_dir = Path(get_package_share_directory("life_assist_dm"))
        share_env = share_dir / ".env"
        if share_env.exists():
            load_dotenv(share_env)
            print(f"[INFO] .env 파일 로드 완료 (share 디렉터리): {share_env}")
            return True
    except Exception:
        pass

    print(f"[WARNING] .env 파일을 찾을 수 없습니다. 시도한 경로: {env_file}, {llm_env}")
    return False


_load_env()