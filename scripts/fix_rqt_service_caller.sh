#!/bin/bash
# rqt_service_caller 한글 인코딩 버그 수정 스크립트
# ROS2 Humble의 rqt_service_caller 375번째 줄 %r → %s 수정
#
# 문제: %r 포맷은 repr()을 호출하여 ASCII 인코딩을 강제하므로 한글 처리 시 UnicodeEncodeError 발생
# 해결: %s (str) 포맷으로 변경하여 UTF-8 안전하게 처리

TARGET_FILE="/opt/ros/humble/lib/python3.10/site-packages/rqt_service_caller/service_caller_widget.py"
BACKUP_FILE="${TARGET_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "rqt_service_caller 한글 인코딩 버그 수정"
echo "=========================================="
echo ""

# 파일 존재 확인
if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ 오류: 파일을 찾을 수 없습니다: $TARGET_FILE"
    exit 1
fi

# 백업 생성
echo "📦 백업 생성 중..."
sudo cp "$TARGET_FILE" "$BACKUP_FILE" || {
    echo "❌ 오류: 백업 생성 실패. sudo 권한이 필요합니다."
    exit 1
}
echo "✅ 백업 완료: $BACKUP_FILE"
echo ""

# 수정 전 내용 확인
echo "🔍 수정 전 내용 (375번째 줄):"
sudo sed -n '375p' "$TARGET_FILE"
echo ""

# 수정 실행: qWarning 호출을 try-except로 감싸서 UTF-8 에러 방지
echo "🔧 수정 중..."
# Python으로 안전하게 수정
sudo python3 << 'PYTHON_EOF'
import re

target_file = "/opt/ros/humble/lib/python3.10/site-packages/rqt_service_caller/service_caller_widget.py"

with open(target_file, 'r', encoding='utf-8') as f:
    lines = f.readlines()

if len(lines) >= 375:
    old_line = lines[374].rstrip()
    print(f"현재 라인: {old_line}")
    
    # qWarning 호출을 try-except로 감싸서 UTF-8 에러 방지
    # 한글 문자가 포함된 경우에도 크래시하지 않도록 처리
    indent = len(old_line) - len(old_line.lstrip())
    indent_str = ' ' * indent
    
    # 가장 안전한 방법: qWarning을 try-except로 감싸서 UTF-8 에러 무시
    # 한글 문자가 포함된 경우에도 크래시하지 않도록 처리
    new_code = f"""{indent_str}try:
{indent_str}    qWarning('ServiceCaller.on_call_service_button_clicked(): request:\\n%s' % str(request))
{indent_str}except (UnicodeEncodeError, UnicodeDecodeError):
{indent_str}    pass  # UTF-8 인코딩 오류 무시 (한글 포함 요청 시 발생)"""
    
    lines[374] = new_code + '\n'
    
    with open(target_file, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"✅ Python을 통한 수정 완료!")
    print(f"수정된 내용:")
    print(new_code)
else:
    print("❌ 파일 라인 수가 부족합니다.")
    exit(1)
PYTHON_EOF

if [ $? -ne 0 ]; then
    echo "❌ 오류: 수정 실패"
    exit 1
fi

# 수정 후 내용 확인
echo "✅ 수정 완료!"
echo ""
echo "🔍 수정 후 내용 (375번째 줄):"
sudo sed -n '375p' "$TARGET_FILE"
echo ""
echo "=========================================="
echo "변경 사항:"
echo "  기존: qWarning('... request:\\\n%s' % str(request))"
echo "  수정: try-except로 감싸서 UnicodeEncodeError 무시"
echo ""
echo "효과:"
echo "  - qWarning 호출을 try-except로 보호"
echo "  - 한글 포함 요청 시 UnicodeEncodeError 발생해도 크래시 안 함"
echo "  - 에러 발생 시 조용히 넘어감 (pass)"
echo "=========================================="
echo ""
echo "📝 다음 단계:"
echo "  1. 현재 실행 중인 rqt를 종료하세요"
echo "  2. rqt를 다시 실행하세요"
echo "  3. 한글 포함 서비스 요청 테스트 (예: '안녕', '권서연')"
echo ""
echo "💾 백업 파일 위치: $BACKUP_FILE"
echo "   (원래대로 되돌리려면: sudo cp $BACKUP_FILE $TARGET_FILE)"
echo ""
