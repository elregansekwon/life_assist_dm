from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# httpx 로그 억제 (rqt 메모리 과부하 방지)
import logging as std_logging
std_logging.getLogger("httpx").setLevel(std_logging.WARNING)
std_logging.getLogger("httpcore").setLevel(std_logging.WARNING)


class LifeAssistant:
<<<<<<< HEAD
    def __init__(self, model_name="gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name,
                              temperature=0.4)
=======
    def __init__(self, model_name="gpt-4o-mini-2024-07-18"):
        # LLM 호출 timeout 설정 (10초) - ROS2 서비스 응답 지연 방지
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0.4,
            timeout=10.0,  # 10초 timeout - rqt 서비스 응답 지연 방지
            max_retries=1  # 재시도 최소화
        )
>>>>>>> 9f3045d (2025-11-06 수정 사항 반영)
        self.output_parser = StrOutputParser()
        self.prompt = PromptTemplate.from_template(
            "입력된 내용은 사람이 로봇에게 요청한 명령입니다."
            "로봇은 사람에게 [인지], [정서], [물리적 지원] 서비스를 제공할 수 있습니다."
            "로봇이 서비스를 제공하기 위해 어떤 작업을 수행할지 [인지], [정서], [물리적 지원]으로 분류되어야 합니다."
            "[인지] 서비스: 사람의 기억력, 반복되는 행위, 스케줄 등과 관련된 서비스입니다."
            "[정서] 서비스: 사람의 감정과 관련된 서비스이며, 일상적인 대화나 스트레스 관리, 날씨 안내 등 보통의 대화에 해당합니다."
            "[물리적 지원] 서비스: 사람이 로봇에게 어떤 물건을 갖다달라거나, 찾아달라고 하는 등 로봇이 물체를 조작하거나 관찰해야하는 서비스에 해당합니다."
            "사람의 명령을 보고 [인지],[정서],[물리적 지원] 서비스 중에서 골라주세요. "
            "서비스 결과가 [인지], [정서]인 경우 명령에 적합한 대답을 해주고, "
            "[물리적 지원]인 경우 명령을 읊어주며 실행할지 물어보는 대답을 생성한 후, 초기 명령을 영어로 번역해주세요."
            "[물리적 지원]의 응답은 수행할지 묻는 것과 영어로 번역한 문장을 '/'로 구분해서 응답해주세요."
            "응답 예시는 다음과 같습니다. '[인지] 오늘 감기약 드셨나요?', '[정서] 오늘 날씨가 좋아요', '[물리적지원] 물 갖다 드릴까요? / Would you like me to bring you some water?"
            "위 예시처럼 서비스 종류가 문장 앞, 응답이 서비스 종류 다음에 배치된 대답을 해야합니다."
            "만약 사용자 명령에서 [인지], [정서], [물리적지원] 키워드가 문장 앞에 입력되어 있다면, 서비스는 구분된 상태입니다."
            "그러므로 더 서비스를 구분하지 않고, 이미 응답했던 내용의 대화를 이어나가야 합니다."
            "\n\n"
            "입력: {stt_text}\n\n"
            "결과: "
        )
        self.chain = self.prompt | self.llm | self.output_parser

    def __call__(self, text: str) -> str:
        if not text:
            return ""
        return self.chain.invoke({"stt_text": text})


class SentenceCorrector:
    """
    """
    def __init__(self, model_name="gpt-4o-mini-2024-07-18"):
        """
        """
        # LLM 호출 timeout 설정 (10초) - ROS2 서비스 응답 지연 방지
        self.llm = ChatOpenAI(
            model=model_name,
            timeout=10.0,  # 10초 timeout
            max_retries=1
        )
        self.output_parser = StrOutputParser()
        self.prompt = PromptTemplate.from_template(
            "다음 문장은 음성 인식(STT)을 통해 자동으로 생성된 텍스트입니다. "
            "띄어쓰기, 문법 오류, 부자연스러운 표현이 있을 수 있습니다. "
            "문맥에 맞게 자연스럽고 완전한 문장으로 수정해주세요. "
            #"문장으로 수정 후, 수정된 문장을 영어로 번역해주세요."
            "\n\n"
            "입력: {stt_text}\n\n"
            "수정된 문장:"
        )
        self.chain = self.prompt | self.llm | self.output_parser
        print("LangChain 교정기 초기화 완료.")

    def correct(self, text: str) -> str:
        """
        text is corrected by LLM
        """
        if not text:
            return ""
        return self.chain.invoke({"stt_text": text})


<<<<<<< HEAD
#추가
from langchain_openai import OpenAIEmbeddings

def get_llm(model_name: str = "gpt-4o-mini-2024-07-18", temperature: float = 0.4):
    """기본 LLM 객체 반환 (LifeAssistMemory에서 사용)"""
    return ChatOpenAI(
        model=model_name, 
        temperature=temperature,
        request_timeout=10,  # 10초 타임아웃 설정 (성능 향상)
        max_retries=1  # 1번만 재시도 (빠른 fallback)
    )

def get_embedding():
    """벡터스토어용 임베딩 함수"""
    return OpenAIEmbeddings()
#
=======
# =============================
# 🔧 Compatibility Utilities
# =============================

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_llm(model_name: str = "gpt-4o-mini-2024-07-18"):
    """기존 코드 호환용: LangChain LLM 인스턴스 반환"""
    return ChatOpenAI(
        model=model_name, 
        temperature=0.4,
        timeout=10.0,  # 10초 timeout
        max_retries=1
    )

def get_embedding(model_name: str = "text-embedding-3-small"):
    """기존 코드 호환용: OpenAI Embedding 인스턴스 반환"""
    return OpenAIEmbeddings(model=model_name)
>>>>>>> 9f3045d (2025-11-06 수정 사항 반영)
