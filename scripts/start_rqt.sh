#!/bin/bash
# rqt 실행 전 UTF-8 환경 변수 설정 스크립트
# rqt_service_caller의 한글 처리 오류(UnicodeEncodeError) 방지
#
# 문제: rqt_service_caller가 한글 포함 서비스 요청을 ASCII로 출력 시도 → 크래시
# 해결: UTF-8 환경 변수 강제 설정으로 Python 출력 인코딩 변경
#
# 사용법:
#   ./scripts/start_rqt.sh              # 일반 실행
#   ./scripts/start_rqt.sh --no-sandbox # 원격 접속 시 안정화 옵션

# UTF-8 환경 변수 강제 설정
export LC_ALL=C.UTF-8
export LANG=C.UTF-8
export LANGUAGE=C.UTF-8
export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PYTHONLEGACYWINDOWSSTDIO=utf-8

# 현재 설정 확인 (디버그용)
echo "[INFO] UTF-8 환경 변수 설정 완료:"
echo "  LC_ALL=$LC_ALL"
echo "  LANG=$LANG"
echo "  PYTHONIOENCODING=$PYTHONIOENCODING"
echo "  PYTHONUTF8=$PYTHONUTF8"

# rqt 실행 (모든 인자 전달)
rqt "$@"

