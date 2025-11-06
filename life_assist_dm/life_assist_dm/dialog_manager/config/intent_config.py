# 의도/키워드 및 엑셀 스키마 설정

PERSONAL_INFO_KEYWORDS = {
    "age": {
        "include": ["나이", "몇 살", "몇살"],
        "suffix": ["살이야?", "살이지?"],
        "exclude": ["저장", "기록"],
    },
    "school": {
        "include": ["학교", "대학교", "초등학교", "중학교", "고등학교"],
        "exclude": ["저장", "기록"],
    },
    "job": {
        "include": ["직업", "무슨 일", "하는 일"],
        "exclude": ["저장", "기록"],
    },
    "hobby": {
        "include": ["취미"],
        "exclude": ["저장", "기록"],
    },
    # 인턴 관련 질의(설정 파일에서 덮어쓸 수 있음)
    "intern": {
        "include": ["인턴", "회사", "직장", "기관"],
        "suffix": ["어디서", "어디에서", "어디서 해", "어디서 하는지"],
        "exclude": ["물건", "위치"],
    },
}

EXCEL_SHEETS = {
    "user_info": "사용자정보",
    "user_info_kv": "사용자정보KV",
    "items": "물건위치",
    "emotions": "감정기록",
    "schedule": "일정",
    "dialog": "대화기록",
}

USER_INFO_COLUMNS = ["날짜", "이름", "나이", "학교", "직업", "취미", "엔티티타입"]


