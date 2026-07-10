"""
Employment Strategy Agent
=========================
단순 Planner + Research Agent + Evidence Evaluator + Final Analyst 구조

실행:
    python -m uvicorn server:app --reload

필수 .env:
    OPENAI_API_KEY=...
    TAVILY_API_KEY=...

선택 .env:
    OPENAI_MODEL=gpt-4o-mini
    RAG_DATA_DIR=./data
    CHROMA_DIR=./chroma_db
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable
from datetime import date
from uuid import uuid4
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import unquote

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain.tools.tool_node import ToolCallRequest
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command


load_dotenv()


# ============================================================
# 1. 기본 설정
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("employment-strategy-agent")

MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RAG_DATA_DIR = Path(os.getenv("RAG_DATA_DIR", "./data"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "./chroma_db"))

model = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
)


# ============================================================
# 2. API / Structured Output 스키마
# ============================================================

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str = Field(
        default="default",
        min_length=1,
        max_length=100,
    )


class ChatResponse(BaseModel):
    answer: str
    thread_id: str
    route: str
    retry_count: int
    used_tools: list[str] = Field(default_factory=list)
    evidence_sources: list[str] = Field(default_factory=list)
    plan: dict | None = None
    structured_result: dict | None = None

    # 상담 후 생성되는 실행 관리 데이터
    career_plan: dict | None = None
    todo_items: list[dict] = Field(default_factory=list)


class UserProfile(BaseModel):
    university: str | None = None
    major: str | None = None
    grade_status: str | None = None
    military_status: str | None = None

    certificates: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    experiences: list[str] = Field(default_factory=list)

    language_status: str | None = None
    target_company: str | None = None
    target_role: str | None = None


class ExecutionPlan(BaseModel):
    """
    단일 intent가 아니라 현재 요청의 목표와 필요한 작업 순서를 표현한다.
    """
    goal: str
    steps: list[str] = Field(default_factory=list)
    needs_research: bool = False
    research_focus: list[str] = Field(default_factory=list)

    # 사용자가 상담 막바지의 제안을 수락하거나 직접 요청했을 때만 True
    create_career_plan: bool = False
    create_todo_list: bool = False


class CareerPlanPhase(BaseModel):
    period: str
    title: str
    tasks: list[str] = Field(default_factory=list)
    recruitment_event: str | None = None
    basis: str | None = None


class CareerPlan(BaseModel):
    target_company: str | None = None
    target_role: str | None = None
    schedule_basis: str
    phases: list[CareerPlanPhase] = Field(default_factory=list)


class TodoDraft(BaseModel):
    title: str
    category: str = "기타"
    priority: Literal["high", "medium", "low"] = "medium"
    due_date: str | None = None
    reason: str | None = None


class ExecutionAssetsDraft(BaseModel):
    career_plan: CareerPlan | None = None
    todo_items: list[TodoDraft] = Field(default_factory=list)


class TodoUpdateRequest(BaseModel):
    completed: bool


class EvidenceEvaluation(BaseModel):
    sufficient: bool
    missing_information: list[str] = Field(default_factory=list)
    reason: str


class AnalysisSummary(BaseModel):
    """
    UI 표시 및 과제의 OutputParser 요구사항을 위한 보조 구조화 결과.
    메인 답변 생성과 분리되어 있으므로 파싱 실패가 대화를 망치지 않는다.
    """
    target_company: str | None = None
    target_role: str | None = None
    strengths: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    priority_actions: list[str] = Field(default_factory=list)
    action_plan: list[str] = Field(default_factory=list)
    evidence_summary: list[str] = Field(default_factory=list)


# ============================================================
# 3. LangGraph State
# ============================================================

class CareerState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]

    # 현재 세션 식별자
    thread_id: str

    # 장기 세션 상태
    user_profile: dict
    target_changed: bool

    # 상담 후 실행 관리
    career_plan: dict
    todo_items: list[dict]

    # Planner
    route: str
    plan: dict

    # 조사 상태
    research_result: str
    evidence: list[str]
    used_tools: list[str]
    evidence_sources: list[str]

    # 검증 / loop
    evidence_evaluation: dict
    retry_count: int
    retry_query: str

    # 최종 출력
    final_answer: str
    structured_result: dict


# ============================================================
# 4. RAG
# ============================================================

_retriever = None

# InMemorySaver와 동일하게 서버 프로세스가 살아 있는 동안 유지되는
# 일정 계획 / 체크리스트 저장소.
execution_store: dict[str, dict] = {}




def get_vectorstore() -> Chroma:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    return Chroma(
        collection_name="career_documents",
        embedding_function=embedding_model,
        persist_directory=str(CHROMA_DIR),
    )


def load_and_split_pdf(pdf_path: Path):
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = pdf_path.name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    return splitter.split_documents(documents)


def index_pdf_file(pdf_path: Path) -> int:
    global _retriever

    chunks = load_and_split_pdf(pdf_path)
    if not chunks:
        return 0

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    _retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4},
    )

    logger.info(
        "PDF INDEXED | file=%s | chunks=%d",
        pdf_path.name,
        len(chunks),
    )
    return len(chunks)


def get_retriever():
    """
    최초 RAG 검색 시 data 폴더 PDF를 읽어 Retriever를 준비한다.
    """
    global _retriever

    if _retriever is not None:
        return _retriever

    pdf_paths = sorted(RAG_DATA_DIR.glob("*.pdf"))
    if not pdf_paths:
        logger.warning(
            "RAG PDF 없음 | path=%s",
            RAG_DATA_DIR.resolve(),
        )
        return None

    documents = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(str(pdf_path))
        loaded = loader.load()

        for doc in loaded:
            doc.metadata["source_file"] = pdf_path.name

        documents.extend(loaded)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    _retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4},
    )

    logger.info(
        "RAG READY | pdf=%d | chunks=%d",
        len(pdf_paths),
        len(chunks),
    )
    return _retriever


# ============================================================
# 5. Tools
# ============================================================

@tool
def search_career_documents(query: str) -> str:
    """
    업로드된 채용공고, 직무기술서, NCS 자료 등 PDF에서
    목표 기업/직무의 요구사항, 우대사항, 일정, 직무 내용을 검색합니다.
    """
    retriever = get_retriever()

    if retriever is None:
        return (
            "RAG 검색 불가: 등록된 PDF가 없습니다. "
            "필요하면 공식 웹 검색 Tool을 사용하세요."
        )

    docs = retriever.invoke(query)

    if not docs:
        return "관련 PDF 근거를 찾지 못했습니다."

    results = []

    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get(
            "source_file",
            doc.metadata.get("source", "unknown"),
        )
        page = doc.metadata.get("page", "?")
        content = " ".join(doc.page_content.split())

        results.append(
            f"[문서 {index}] source={source}, page={page}\n"
            f"{content[:1600]}"
        )

    return "\n\n".join(results)


_tavily = TavilySearch(
    max_results=6,
    topic="general",
    search_depth="advanced",
)


@tool
def search_official_recruitment_web(
    query: str,
    official_domain: str = "",
) -> str:
    """
    기업/기관의 공식 채용 홈페이지, 공식 채용공고,
    직무기술서, NCS 자료, 우대사항 등을 검색합니다.

    공식 도메인을 알고 있으면 official_domain에
    예: ksure.or.kr 형태로 전달할 수 있습니다.
    """
    payload = {
        "query": (
            f"{query} 공식 채용 채용공고 직무기술서 "
            "자격요건 우대사항 가산점"
        )
    }

    cleaned_domain = official_domain.strip().lower()

    if cleaned_domain:
        cleaned_domain = (
            cleaned_domain
            .removeprefix("https://")
            .removeprefix("http://")
            .split("/", 1)[0]
        )
        payload["include_domains"] = [cleaned_domain]

    result = _tavily.invoke(payload)

    return json.dumps(
        {
            "search_type": "official_recruitment",
            "query": query,
            "official_domain": cleaned_domain or None,
            "result": result,
        },
        ensure_ascii=False,
        default=str,
    )


@tool
def general_web_search(query: str) -> str:
    """
    공식 자료나 업로드 PDF만으로 부족할 때
    추가 위치 파악 및 보조 정보 검색에 사용합니다.
    """
    result = _tavily.invoke({"query": query})

    return json.dumps(
        {
            "search_type": "general_web",
            "query": query,
            "result": result,
        },
        ensure_ascii=False,
        default=str,
    )


# ============================================================
# 6. Middleware
# ============================================================

@wrap_tool_call
def monitor_tool_calls(
    request: ToolCallRequest,
    handler: Callable[
        [ToolCallRequest],
        ToolMessage | Command,
    ],
) -> ToolMessage | Command:
    """
    Tool 실행 로깅 + 오류 기록 Middleware.
    """
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call["args"]

    logger.info(
        "TOOL START | name=%s | args=%s",
        tool_name,
        tool_args,
    )

    try:
        result = handler(request)

        logger.info(
            "TOOL SUCCESS | name=%s",
            tool_name,
        )
        return result

    except Exception:
        logger.exception(
            "TOOL ERROR | name=%s",
            tool_name,
        )
        raise


# ============================================================
# 7. Research Agent
# ============================================================

research_agent = create_agent(
    model=model,
    tools=[
        search_career_documents,
        search_official_recruitment_web,
        general_web_search,
    ],
    system_prompt=(
        "당신은 Employment Strategy Agent의 조사 담당 Research Agent입니다. "
        "한국어로 작업하세요. "
        "사용자의 목표를 해결하는 데 필요한 정보를 스스로 판단하고 Tool을 선택하세요. "
        "한 번의 검색으로 충분하지 않으면 여러 번 검색해도 됩니다. "
        "특정 기업/기관 취업 분석이라면 필요에 따라 "
        "채용공고, 지원자격, 우대사항, 가산점, 직무기술서/NCS, "
        "직무 요구역량, 채용절차를 조사하세요. "
        "관련 업로드 PDF가 있으면 RAG를 활용하세요. "
        "기업/기관의 사실은 공식 출처를 우선하세요. "
        "최신 공식 공고에서 필요한 기준이 확인되지 않으면, "
        "과거 공식 채용공고와 과거 공식 직무자료까지 추가 탐색할 수 있습니다. "
        "과거 자료는 연도/채용차수를 표시하고 현재 기준과 구분하세요. "
        "일반 웹 검색은 부족한 정보를 보완하는 용도로 사용하세요. "
        "최종 사용자 답변을 쓰기보다, 다음 분석 Agent가 활용할 수 있도록 "
        "찾은 사실과 출처를 명확히 정리하세요. "
        "근거가 없는 내용을 임의로 만들지 마세요."
    ),
    middleware=[monitor_tool_calls],
)


# ============================================================
# 8. Helper
# ============================================================

def _latest_user_text(state: CareerState) -> str:
    for message in reversed(state.get("messages", [])):
        if isinstance(message, HumanMessage):
            return str(message.content)
    return ""


def _recent_conversation_text(
    state: CareerState,
    limit: int = 12,
) -> str:
    recent = state.get("messages", [])[-limit:]
    lines = []

    for message in recent:
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            role = "message"

        lines.append(f"{role}: {message.content}")

    return "\n".join(lines)


def _extract_sources(text: str) -> list[str]:
    sources = []

    # URL
    for url in re.findall(
        r'https?://[^\s"\'\]\[()<>{},]+',
        text,
    ):
        sources.append(url.rstrip(".,;"))

    # RAG source=파일명
    for source in re.findall(
        r"source=([^,\n]+)",
        text,
    ):
        sources.append(source.strip())

    return list(dict.fromkeys(sources))


# ============================================================
# 9. LangGraph Nodes
# ============================================================

def update_context_node(state: CareerState) -> dict:
    """
    현재 발화에서 사용자 프로필을 추출해 누적한다.
    분석은 하지 않고 Context만 갱신한다.
    """
    current_text = _latest_user_text(state)

    old_profile = UserProfile.model_validate(
        state.get("user_profile", {}),
    )

    extractor = model.with_structured_output(UserProfile)

    extracted = extractor.invoke(
        f"""
        아래 '현재 사용자 발화'에서 명시적으로 확인 가능한
        취업 관련 사용자 정보만 추출하세요.

        규칙:
        - 추측하지 마세요.
        - 언급되지 않은 값은 null 또는 빈 리스트로 두세요.
        - 현재 발화만 보고 추출하세요.

        추출 대상:
        - university
        - major
        - grade_status
        - military_status
        - certificates
        - skills
        - experiences
        - language_status
        - target_company
        - target_role

        현재 사용자 발화:
        {current_text}
        """
    )

    merged = UserProfile(
        university=extracted.university or old_profile.university,
        major=extracted.major or old_profile.major,
        grade_status=extracted.grade_status or old_profile.grade_status,
        military_status=extracted.military_status or old_profile.military_status,

        certificates=list(dict.fromkeys(
            old_profile.certificates
            + extracted.certificates
        )),
        skills=list(dict.fromkeys(
            old_profile.skills
            + extracted.skills
        )),
        experiences=list(dict.fromkeys(
            old_profile.experiences
            + extracted.experiences
        )),

        language_status=(
            extracted.language_status
            or old_profile.language_status
        ),
        target_company=(
            extracted.target_company
            or old_profile.target_company
        ),
        target_role=(
            extracted.target_role
            or old_profile.target_role
        ),
    )

    target_changed = (
        merged.target_company != old_profile.target_company
        or merged.target_role != old_profile.target_role
    )

    updates = {
        "user_profile": merged.model_dump(),
        "target_changed": target_changed,

        # 현재 턴 임시 상태
        "route": "planner",
        "plan": {},
        "evidence_evaluation": {},
        "retry_count": 0,
        "retry_query": "",
        "final_answer": "",
        "structured_result": {},

        # 같은 thread에서는 기존 생성 결과를 유지
        "career_plan": state.get("career_plan", {}),
        "todo_items": state.get("todo_items", []),
    }

    # 목표가 바뀌면 이전 회사/직무의 조사 근거를 버린다.
    if target_changed:
        updates.update(
            {
                "research_result": "",
                "evidence": [],
                "used_tools": [],
                "evidence_sources": [],
                "career_plan": {},
                "todo_items": [],
            }
        )

    return updates


def planner_node(state: CareerState) -> dict:
    """
    하나의 intent를 선택하지 않는다.
    현재 요청을 해결하기 위한 목표와 작업 순서를 세운다.
    """
    current_text = _latest_user_text(state)
    conversation = _recent_conversation_text(state)
    profile = state.get("user_profile", {})
    has_previous_evidence = bool(state.get("evidence", []))
    has_career_plan = bool(state.get("career_plan", {}))
    has_todo_items = bool(state.get("todo_items", []))

    planner = model.with_structured_output(ExecutionPlan)

    plan = planner.invoke(
        f"""
        당신은 Employment Strategy Agent의 Planner입니다.

        현재 질문을 하나의 intent 카테고리로 분류하지 마세요.
        사용자가 실제로 원하는 목표를 파악하고,
        그 목표를 해결하기 위한 자연스러운 작업 순서를 계획하세요.

        현재 날짜:
        {date.today().isoformat()}

        현재 사용자 요청:
        {current_text}

        최근 대화:
        {conversation}

        누적 사용자 프로필:
        {json.dumps(profile, ensure_ascii=False)}

        이전 조사 근거 보유 여부:
        {has_previous_evidence}

        기존 채용 일정 계획 존재 여부:
        {has_career_plan}

        기존 해야 할 일 목록 존재 여부:
        {has_todo_items}

        계획 지침:
        - 특정 회사/기관의 특정 직무에 취업하고 싶다는 요청이라면,
          현재 요청이 단순 잡담이 아닌 이상 최근 공식 채용공고를 확인하는 것을 우선 검토하세요.
        - 이런 취업 분석에서는 보통 다음 정보를 실제 근거로 확인하는 것이 유용합니다:
          최근 공식 채용공고, 지원자격, 우대사항, 가산점,
          직무기술서/NCS, 전형절차, 직무 요구역량.
        - 그 뒤 사용자 현재 상태와 비교하여
          이미 충족한 부분, 부족한 부분, 아직 판단이 어려운 부분,
          앞으로의 준비 우선순위를 설명할 수 있도록 계획하세요.
        - 다만 위 순서를 무조건 강제하지 말고 현재 요청에 필요한 것만 계획하세요.
        - 최신 채용 일정, 실제 우대사항, 가산점, 기업별 요구조건처럼
          외부 사실 확인이 필요하면 needs_research=True.
        - 사용자가 새 프로필 정보를 제공했고 이전 회사/직무 분석이 이어지는 상황이면,
          이전 근거를 활용해 수정 분석할 수 있으므로 매번 재검색할 필요는 없습니다.
        - 이전 조사 근거가 있어도 최신성 확인이 필요한 질문이면 다시 조사하세요.
        - 최신 공식 채용공고에서 사용자가 비교하고 싶어 하는 핵심 항목
          (예: 가산점 자격증 목록, 우대사항, 어학 기준, 전형 기준)이 확인되지 않으면,
          과거 공식 채용공고나 과거 공식 직무자료를 추가로 확인하는 단계를 계획할 수 있습니다.
        - 과거 자료는 현재 기준으로 단정하기 위한 것이 아니라
          '과거에는 어떤 기준이 적용되었는지'를 보조적으로 설명하기 위한 근거입니다.
        - 사용자가 직접 채용 일정 계획이나 할 일 체크리스트 생성을 요청했다면
          create_career_plan / create_todo_list를 필요한 만큼 True로 설정하세요.
        - 사용자가 "응", "좋아", "그래", "만들어줘"처럼 짧게 답한 경우에는
          반드시 최근 대화를 확인하세요. 직전 Assistant가
          "채용 일정에 맞춘 준비 계획과 해야 할 일 체크리스트를 만들어드릴까요?"라고
          제안한 상황이라면 해당 수락으로 해석하여 두 값을 True로 설정할 수 있습니다.
        - 단순한 긍정 표현을 무조건 계획 생성 요청으로 해석하지 마세요.
        - 계획/체크리스트를 생성할 때 채용 일정의 최신 확인이 필요하고
          기존 조사 근거가 부족하면 needs_research=True로 둘 수 있습니다.
        - 이미 필요한 조사 근거가 충분하면 기존 근거를 재사용하세요.
        - steps에는 사람이 읽어도 이해되는 작업 순서를 작성하세요.
        - research_focus에는 Research Agent가 찾아야 할 구체적 정보만 작성하세요.
        """
    )

    return {
        "route": "planner",
        "plan": plan.model_dump(),
    }


def route_after_planner(
    state: CareerState,
) -> Literal[
    "research_agent",
    "final_analysis",
]:
    plan = ExecutionPlan.model_validate(
        state.get("plan", {}),
    )

    return (
        "research_agent"
        if plan.needs_research
        else "final_analysis"
    )


def research_agent_node(state: CareerState) -> dict:
    """
    Planner의 목표를 받아 Tool을 자율 선택해 조사한다.
    """
    current_text = _latest_user_text(state)
    conversation = _recent_conversation_text(state)
    profile = state.get("user_profile", {})
    plan = ExecutionPlan.model_validate(
        state.get("plan", {}),
    )
    retry_query = state.get("retry_query", "")

    prompt = f"""
    현재 날짜:
    {date.today().isoformat()}

    Planner 목표:
    {plan.goal}

    Planner 작업 순서:
    {json.dumps(plan.steps, ensure_ascii=False)}

    조사 초점:
    {json.dumps(plan.research_focus, ensure_ascii=False)}

    추가 보완 조사 요청:
    {retry_query or "없음"}

    현재 사용자 요청:
    {current_text}

    최근 대화:
    {conversation}

    사용자 프로필:
    {json.dumps(profile, ensure_ascii=False)}

    조사 원칙:
    - 위 목표를 실제로 해결할 수 있을 만큼 필요한 정보를 조사하세요.
    - Tool은 스스로 선택하세요.
    - 한 번의 Tool 호출로 부족하면 추가 호출하세요.
    - 특정 기업 취업 분석이라면 먼저 최근 공식 채용공고를 찾고,
      그 공고와 공식 직무자료를 기준으로 다음을 확인하세요:
      지원자격, 우대사항, 가산점, 자격증/어학 기준,
      직무기술서/NCS, 요구역량, 채용절차.
    - 최근 공식 채용공고가 여러 개면 목표 직무와 가장 관련 있는 공고를 우선하세요.
    - 최신 공식 공고에서 핵심 비교 항목이 확인되지 않으면 바로 포기하지 마세요.
      필요한 경우 과거 공식 채용공고와 과거 공식 직무자료까지 추가 검색하세요.
    - 특히 다음 항목이 최신 공고에서 빠져 있으면 과거 공식 자료 확인을 고려하세요:
      가산점 대상 자격증, 우대 자격증, 어학 기준, 전형별 가점,
      직무별 세부 요구조건, 채용절차.
    - 과거 자료를 찾았으면 반드시 연도/채용차수/공고 시점을 함께 정리하세요.
    - 과거 기준을 현재 기준으로 단정하지 마세요.
      '과거 공식 공고에서는 확인됨 / 현재 적용 여부는 별도 확인 필요'로 구분하세요.
    - 최신 자료와 과거 자료가 충돌하면 최신 공식 자료를 우선하세요.
    - 업로드 PDF가 관련되면 RAG를 사용하세요.
    - 공식 출처를 우선하세요.
    - 일반 웹 자료를 사용했다면 공식 근거와 구분하세요.
    - 근거 없는 내용을 만들지 마세요.
    - 다음 분석 Agent가 사용자의 현재 상태와 비교할 수 있도록,
      '무엇이 요구되는지/우대되는지/가산점인지'를 출처와 함께 명확히 정리하세요.
    """

    result = research_agent.invoke(
        {
            "messages": [
                HumanMessage(content=prompt),
            ]
        }
    )

    evidence = list(state.get("evidence", []))
    used_tools = list(state.get("used_tools", []))
    evidence_sources = list(
        state.get("evidence_sources", [])
    )

    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            content = str(message.content)
            evidence.append(content)

            tool_name = getattr(message, "name", None)
            if tool_name:
                used_tools.append(tool_name)

            evidence_sources.extend(
                _extract_sources(content)
            )

    final_message = result["messages"][-1]

    research_text = (
        str(final_message.content)
        if isinstance(final_message, AIMessage)
        else str(final_message)
    )

    old_research = str(
        state.get("research_result", "")
    ).strip()

    combined_research = (
        f"{old_research}\n\n[추가 조사]\n{research_text}"
        if old_research
        else research_text
    )

    return {
        "research_result": combined_research,
        "evidence": list(dict.fromkeys(evidence)),
        "used_tools": list(dict.fromkeys(used_tools)),
        "evidence_sources": list(
            dict.fromkeys(evidence_sources)
        ),
        "retry_query": "",
    }


def evidence_evaluator_node(state: CareerState) -> dict:
    """
    조사 결과가 현재 목표에 답하기에 충분한지 평가한다.
    부족하면 무엇이 빠졌는지만 알려주고 Research Agent로 되돌린다.
    """
    plan = ExecutionPlan.model_validate(
        state.get("plan", {}),
    )
    profile = state.get("user_profile", {})
    research_result = state.get(
        "research_result",
        "",
    )
    evidence = state.get("evidence", [])

    evaluator = model.with_structured_output(
        EvidenceEvaluation
    )

    evaluation = evaluator.invoke(
        f"""
        당신은 조사 품질 평가자입니다.

        Planner 목표:
        {plan.goal}

        Planner 작업 순서:
        {json.dumps(plan.steps, ensure_ascii=False)}

        사용자 프로필:
        {json.dumps(profile, ensure_ascii=False)}

        조사 요약:
        {research_result}

        Tool 조사 근거:
        {json.dumps(evidence, ensure_ascii=False)}

        판단 기준:
        - 현재 목표에 책임 있게 답할 수 있는가?
        - 특정 회사 취업 분석이라면 최근 공식 채용공고 또는 이에 준하는 공식 자료가 있는가?
        - 사용자와 비교할 수 있을 정도로
          지원자격/우대사항/가산점/직무 요구 정보가 확보됐는가?
        - 사용자의 보유 자격증·어학·전공·경험이
          어떤 공식 기준과 연결되는지 판단할 근거가 있는가?
        - 최신 사실 질문이면 최신 근거가 있는가?
        - 최신 공식 자료에 핵심 항목이 없지만 과거 공식 공고에서
          보조적으로 확인할 가치가 있는가?
        - 예를 들어 가산점 자격증 목록, 우대 자격증, 어학 기준이
          최신 공고에서 확인되지 않았다면 과거 공식 공고 탐색이 필요한지 판단한다.
        - 모든 세부정보를 완벽히 찾을 필요는 없지만,
          강점/부족점/우선순위에 대한 핵심 결론을 만들 수 있어야 한다.
        - 부족하면 missing_information에
          Research Agent가 추가로 찾아야 할 정보만 구체적으로 적는다.
        - 과거 자료가 필요하면 '과거 공식 채용공고에서 ○○ 기준 확인'처럼
          검색 목표를 구체적으로 작성한다.
        """
    )

    retry_count = state.get("retry_count", 0)

    if (
        not evaluation.sufficient
        and retry_count < 2
    ):
        retry_count += 1
        retry_query = (
            "다음 누락 정보를 추가 조사하세요: "
            + "; ".join(
                evaluation.missing_information
            )
        )
    else:
        retry_query = ""

    return {
        "evidence_evaluation": evaluation.model_dump(),
        "retry_count": retry_count,
        "retry_query": retry_query,
    }


def route_after_evidence(
    state: CareerState,
) -> Literal[
    "research_agent",
    "final_analysis",
]:
    evaluation = EvidenceEvaluation.model_validate(
        state.get(
            "evidence_evaluation",
            {
                "sufficient": True,
                "missing_information": [],
                "reason": "",
            },
        )
    )

    if (
        not evaluation.sufficient
        and state.get("retry_query")
        and state.get("retry_count", 0) <= 2
    ):
        return "research_agent"

    return "final_analysis"


def final_analysis_node(state: CareerState) -> dict:
    """
    조사 + 사용자 상태 + 최근 대화를 한 번에 보고
    자연스럽게 최종 답변한다.
    """
    current_text = _latest_user_text(state)
    conversation = _recent_conversation_text(state)
    profile = state.get("user_profile", {})
    plan = ExecutionPlan.model_validate(
        state.get("plan", {}),
    )

    research_result = state.get(
        "research_result",
        "",
    )
    evidence = state.get("evidence", [])
    evaluation = state.get(
        "evidence_evaluation",
        {},
    )
    existing_career_plan = state.get("career_plan", {})
    existing_todo_items = state.get("todo_items", [])

    response = model.invoke(
        f"""
        당신은 한국어 Employment Strategy Agent입니다.

        현재 사용자 요청:
        {current_text}

        최근 대화:
        {conversation}

        누적 사용자 프로필:
        {json.dumps(profile, ensure_ascii=False)}

        Planner 목표:
        {plan.goal}

        Planner 작업 순서:
        {json.dumps(plan.steps, ensure_ascii=False)}

        조사 요약:
        {research_result}

        Tool 근거:
        {json.dumps(evidence, ensure_ascii=False)}

        근거 평가:
        {json.dumps(evaluation, ensure_ascii=False)}

        기존 채용 일정 계획:
        {json.dumps(existing_career_plan, ensure_ascii=False)}

        기존 해야 할 일 목록:
        {json.dumps(existing_todo_items, ensure_ascii=False)}

        이번 Planner의 계획 생성 요청:
        create_career_plan={plan.create_career_plan}
        create_todo_list={plan.create_todo_list}

        답변 원칙:
        - 현재 질문에 직접 답하세요.
        - 고정된 메뉴나 템플릿을 무조건 사용하지 마세요.
        - 자연스러운 대화 흐름을 유지하세요.
        - 이전 대화에서 사용자가 새 정보를 제공했다면
          단순히 프로필을 다시 읊지 말고 기존 판단과 계획을 수정하세요.
        - 특정 기업/기관 취업 목표라면, 현재 질문에 적절한 경우
          가장 최근에 확인된 공식 채용공고와 공식 직무자료를 우선 근거로 사용하세요.
        - 우대사항, 가산점, 지원자격, 직무 요구조건을 사용자 현재 상태와 비교하여
          1) 이미 충족하거나 강점으로 볼 수 있는 부분,
          2) 명확히 부족한 부분,
          3) 근거가 부족해 추가 확인이 필요한 부분,
          4) 앞으로의 준비 우선순위
          를 구분해 설명하세요.
        - 가능하면 각 판단 옆에 어떤 공고/우대사항/가산점/직무요건을 근거로 했는지 밝혀 주세요.
        - 단순히 'IT에서 좋다', '있으면 유리하다'는 일반론보다
          해당 회사의 실제 채용 기준과의 연결을 우선하세요.
        - 사용자가 이미 충족한 부분은 부족하다고 하지 마세요.
        - 부족한 정보와 실제 부족 역량을 구분하세요.
        - 최근 공식 채용공고에서 확인된 우대사항/가산점/요구조건과
          사용자 프로필의 연결을 최우선으로 설명하세요.
        - 최신 공고에서 확인되지 않았지만 과거 공식 채용공고에서 확인된 내용이 있으면,
          답변에 보조 근거로 포함할 수 있습니다.
        - 이 경우 반드시 다음을 구분하세요:
          1) 현재 공식 공고에서 확인된 기준,
          2) 과거 공식 공고에서 확인된 사례,
          3) 현재 채용에 동일 적용되는지는 미확인인지 여부.
        - 과거 사례에는 가능하면 연도/채용차수/공고 시점을 함께 적으세요.
        - 과거 기준을 현재 기준처럼 단정하지 마세요.
        - 최신 자료와 과거 자료가 충돌하면 최신 공식 자료를 우선하세요.
        - 공고에 없는 자격증이나 역량을
          단순히 '하면 좋다'는 이유로 추천하지 마세요.
        - 공식 근거가 부족한 부분은 그 한계만 솔직히 밝히되,
          이미 확보된 정보까지 버리지 마세요.
        - 가산점/우대 여부는 확인된 근거가 있을 때만 단정하세요.
        - 출처가 있으면 답변에서 자연스럽게 밝혀 주세요.
        - 사용자가 묻지 않은 내용을 과도하게 늘리지 마세요.
        - 충분한 취업 상담이 진행되어 목표 기업/직무, 현재 강점·부족점,
          준비 우선순위가 어느 정도 정리되었고 아직 일정 계획과 체크리스트가 없다면,
          답변 마지막에 자연스럽게
          "이 분석을 바탕으로 채용 일정에 맞춘 준비 계획과 해야 할 일 체크리스트를 만들어드릴까요?"
          라고 한 번 제안할 수 있습니다.
        - 이미 일정 계획과 체크리스트가 존재하면 같은 제안을 반복하지 마세요.
        - 이번 Planner에서 create_career_plan 또는 create_todo_list가 True라면
          지금은 생성 단계로 이어질 것이므로 다시 생성 여부를 묻지 마세요.
        """
    )

    return {
        "final_answer": str(response.content),
    }


def execution_asset_builder_node(state: CareerState) -> dict:
    """
    기존 상담 결과와 조사 근거를 바탕으로
    채용 일정 계획과 해야 할 일 체크리스트를 구조화해 생성한다.
    """
    profile = state.get("user_profile", {})
    plan = ExecutionPlan.model_validate(
        state.get("plan", {}),
    )
    conversation = _recent_conversation_text(state)
    research_result = state.get("research_result", "")
    evidence = state.get("evidence", [])
    final_answer = state.get("final_answer", "")
    existing_career_plan = state.get("career_plan", {})
    existing_todo_items = state.get("todo_items", [])

    builder = model.with_structured_output(
        ExecutionAssetsDraft
    )

    draft = builder.invoke(
        f"""
        당신은 취업 실행계획 설계자입니다.

        사용자 프로필:
        {json.dumps(profile, ensure_ascii=False)}

        최근 대화:
        {conversation}

        Planner 목표:
        {plan.goal}

        조사 요약:
        {research_result}

        Tool 근거:
        {json.dumps(evidence, ensure_ascii=False)}

        직전 상담 답변:
        {final_answer}

        기존 채용 일정 계획:
        {json.dumps(existing_career_plan, ensure_ascii=False)}

        기존 해야 할 일:
        {json.dumps(existing_todo_items, ensure_ascii=False)}

        생성 요청:
        - 채용 일정 계획: {plan.create_career_plan}
        - 해야 할 일 체크리스트: {plan.create_todo_list}

        생성 규칙:
        - 사용자의 목표 기업/직무와 현재 상태를 기준으로 작성하세요.
        - 조사된 실제 채용 일정이 있으면 그 일정에 맞춰 역산하세요.
        - 정확한 날짜가 확인되지 않으면 날짜를 만들지 말고
          "공고 발표 후", "지원 마감 2주 전", "필기 4주 전" 같은 상대 기간을 사용하세요.
        - 사용자가 이미 보유한 자격증을 다시 취득 과제로 넣지 마세요.
        - 실제 부족점과 준비 우선순위를 중심으로 작성하세요.
        - 과거 채용 일정만 확인된 경우 현재 일정처럼 단정하지 말고 basis에 명시하세요.
        - career_plan은 create_career_plan=True일 때 의미 있게 작성하세요.
        - todo_items는 create_todo_list=True일 때 의미 있게 작성하세요.
        - Todo는 실행 가능한 한 문장으로 쓰고, 우선순위와 기한을 가능한 범위에서 설정하세요.
        - 새로운 사실을 임의로 만들지 마세요.
        """
    )

    career_plan = state.get("career_plan", {})
    if plan.create_career_plan and draft.career_plan is not None:
        career_plan = draft.career_plan.model_dump()

    todo_items = list(state.get("todo_items", []))
    if plan.create_todo_list:
        # 같은 제목의 기존 항목이 있으면 완료 상태를 유지한다.
        existing_by_title = {
            str(item.get("title", "")).strip().lower(): item
            for item in todo_items
            if item.get("title")
        }

        new_items = []
        for item in draft.todo_items:
            key = item.title.strip().lower()
            old = existing_by_title.get(key, {})

            new_items.append(
                {
                    "id": old.get("id") or uuid4().hex,
                    "title": item.title,
                    "category": item.category,
                    "priority": item.priority,
                    "due_date": item.due_date,
                    "reason": item.reason,
                    "completed": bool(old.get("completed", False)),
                }
            )

        todo_items = new_items

    generated_parts = []
    if plan.create_career_plan:
        generated_parts.append("채용 일정 계획")
    if plan.create_todo_list:
        generated_parts.append("해야 할 일 체크리스트")

    answer = state.get("final_answer", "").rstrip()
    if generated_parts:
        answer += (
            "\n\n"
            + " · ".join(generated_parts)
            + "를 생성해 전용 탭에 반영했습니다."
        )

    return {
        "career_plan": career_plan,
        "todo_items": todo_items,
        "final_answer": answer,
    }


def route_after_final_analysis(
    state: CareerState,
) -> Literal[
    "execution_asset_builder",
    "structured_summary",
]:
    plan = ExecutionPlan.model_validate(
        state.get("plan", {}),
    )

    if (
        plan.create_career_plan
        or plan.create_todo_list
    ):
        return "execution_asset_builder"

    return "structured_summary"


def structured_summary_node(state: CareerState) -> dict:
    """
    메인 답변과 분리된 보조 Structured Output.
    PydanticOutputParser 사용.
    실패해도 메인 답변은 유지된다.
    """
    parser = PydanticOutputParser(
        pydantic_object=AnalysisSummary,
    )

    profile = state.get("user_profile", {})
    answer = state.get("final_answer", "")

    prompt = f"""
    아래 최종 답변의 내용만 사용해
    보조 분석 요약 JSON을 작성하세요.

    새로운 사실을 추가하지 마세요.
    답변에 없는 항목은 빈 리스트로 두세요.

    사용자 프로필의 목표:
    {json.dumps(profile, ensure_ascii=False)}

    최종 답변:
    {answer}

    {parser.get_format_instructions()}
    """

    try:
        raw = model.invoke(prompt)
        parsed = parser.parse(str(raw.content))

        return {
            "structured_result": parsed.model_dump(),
        }

    except Exception as exc:
        logger.warning(
            "STRUCTURED SUMMARY FAIL | %s",
            exc,
        )

        fallback = AnalysisSummary(
            target_company=profile.get("target_company"),
            target_role=profile.get("target_role"),
        )

        return {
            "structured_result": fallback.model_dump(),
        }


def commit_response_node(state: CareerState) -> dict:
    answer = state.get("final_answer", "").strip()

    return {
        "messages": [
            AIMessage(content=answer),
        ]
    }


# ============================================================
# 10. LangGraph
# ============================================================

workflow = StateGraph(CareerState)

workflow.add_node(
    "update_context",
    update_context_node,
)
workflow.add_node(
    "planner",
    planner_node,
)
workflow.add_node(
    "research_agent",
    research_agent_node,
)
workflow.add_node(
    "evidence_evaluator",
    evidence_evaluator_node,
)
workflow.add_node(
    "final_analysis",
    final_analysis_node,
)
workflow.add_node(
    "execution_asset_builder",
    execution_asset_builder_node,
)
workflow.add_node(
    "structured_summary",
    structured_summary_node,
)
workflow.add_node(
    "commit_response",
    commit_response_node,
)

workflow.add_edge(
    START,
    "update_context",
)
workflow.add_edge(
    "update_context",
    "planner",
)

workflow.add_conditional_edges(
    "planner",
    route_after_planner,
    {
        "research_agent": "research_agent",
        "final_analysis": "final_analysis",
    },
)

workflow.add_edge(
    "research_agent",
    "evidence_evaluator",
)

workflow.add_conditional_edges(
    "evidence_evaluator",
    route_after_evidence,
    {
        "research_agent": "research_agent",
        "final_analysis": "final_analysis",
    },
)

workflow.add_conditional_edges(
    "final_analysis",
    route_after_final_analysis,
    {
        "execution_asset_builder": "execution_asset_builder",
        "structured_summary": "structured_summary",
    },
)

workflow.add_edge(
    "execution_asset_builder",
    "structured_summary",
)
workflow.add_edge(
    "structured_summary",
    "commit_response",
)
workflow.add_edge(
    "commit_response",
    END,
)


# ============================================================
# 11. Memory
# ============================================================

checkpointer = InMemorySaver()

career_graph = workflow.compile(
    checkpointer=checkpointer,
)


# ============================================================
# 12. FastAPI
# ============================================================

app = FastAPI(
    title="Employment Strategy Agent",
    version="2.1.0",
)


@app.get("/", response_class=HTMLResponse)
def home() -> HTMLResponse:
    html_path = Path("./templates/index.html")

    if html_path.exists():
        return HTMLResponse(
            html_path.read_text(encoding="utf-8")
        )

    return HTMLResponse(
        """
        <!doctype html>
        <html lang="ko">
        <head>
          <meta charset="utf-8">
          <title>Employment Strategy Agent</title>
        </head>
        <body>
          <h1>Employment Strategy Agent</h1>
          <p>POST /chat 엔드포인트를 사용하세요.</p>
          <p><a href="/docs">Swagger UI</a></p>
        </body>
        </html>
        """
    )


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.get("/rag/documents")
def list_rag_documents() -> dict:
    RAG_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    documents = []

    for pdf_path in sorted(
        RAG_DATA_DIR.glob("*.pdf")
    ):
        stat = pdf_path.stat()

        documents.append(
            {
                "filename": pdf_path.name,
                "size_bytes": stat.st_size,
                "modified_at": stat.st_mtime,
            }
        )

    return {
        "count": len(documents),
        "documents": documents,
    }


@app.delete("/rag/documents/{filename}")
def delete_rag_document(filename: str) -> dict:
    global _retriever

    safe_name = Path(
        unquote(filename)
    ).name

    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="PDF 파일만 삭제할 수 있습니다.",
        )

    file_path = RAG_DATA_DIR / safe_name

    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "등록된 PDF를 찾을 수 없습니다: "
                f"{safe_name}"
            ),
        )

    deleted_chunks = 0

    try:
        vectorstore = get_vectorstore()

        stored = vectorstore.get(
            where={"source_file": safe_name}
        )

        ids = (
            stored.get("ids", [])
            if isinstance(stored, dict)
            else []
        )

        if ids:
            vectorstore.delete(ids=ids)
            deleted_chunks = len(ids)

        file_path.unlink()

        _retriever = vectorstore.as_retriever(
            search_kwargs={"k": 4},
        )

        logger.info(
            "PDF DELETED | file=%s | chunks=%d",
            safe_name,
            deleted_chunks,
        )

        return {
            "message": f"{safe_name} 삭제 완료",
            "filename": safe_name,
            "deleted_chunks": deleted_chunks,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "PDF DELETE ERROR | file=%s",
            safe_name,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                f"{safe_name} 삭제 중 오류가 발생했습니다: "
                f"{exc}"
            ),
        ) from exc


@app.post("/rag/upload")
async def upload_rag_documents(
    files: list[UploadFile] = File(...),
) -> dict:
    if not files:
        raise HTTPException(
            status_code=400,
            detail="업로드할 PDF 파일이 없습니다.",
        )

    RAG_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    uploaded_files = []
    total_chunks = 0

    for upload in files:
        filename = Path(
            upload.filename or ""
        ).name

        if not filename:
            raise HTTPException(
                status_code=400,
                detail="파일명이 비어 있습니다.",
            )

        if not filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=(
                    "PDF 파일만 업로드할 수 있습니다: "
                    f"{filename}"
                ),
            )

        save_path = RAG_DATA_DIR / filename

        try:
            content = await upload.read()

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "빈 파일은 업로드할 수 없습니다: "
                        f"{filename}"
                    ),
                )

            save_path.write_bytes(content)

            chunk_count = index_pdf_file(
                save_path
            )
            total_chunks += chunk_count

            uploaded_files.append(
                {
                    "filename": filename,
                    "saved_path": str(save_path),
                    "chunks": chunk_count,
                }
            )

        except HTTPException:
            raise

        except Exception as exc:
            logger.exception(
                "PDF UPLOAD ERROR | file=%s",
                filename,
            )

            if save_path.exists():
                try:
                    save_path.unlink()
                except OSError:
                    logger.warning(
                        "FAILED FILE CLEANUP ERROR | path=%s",
                        save_path,
                    )

            raise HTTPException(
                status_code=500,
                detail=(
                    f"{filename} 처리 중 오류가 발생했습니다: "
                    f"{exc}"
                ),
            ) from exc

        finally:
            await upload.close()

    return {
        "message": (
            f"{len(uploaded_files)}개 PDF 업로드 및 "
            f"{total_chunks}개 Chunk 인덱싱 완료"
        ),
        "files": uploaded_files,
        "total_chunks": total_chunks,
    }


@app.get("/career/plan")
def get_career_plan(thread_id: str = "default") -> dict:
    data = execution_store.get(thread_id, {})
    return {
        "thread_id": thread_id,
        "career_plan": data.get("career_plan"),
    }


@app.get("/career/todos")
def get_todo_items(thread_id: str = "default") -> dict:
    data = execution_store.get(thread_id, {})
    return {
        "thread_id": thread_id,
        "todo_items": data.get("todo_items", []),
    }


@app.patch("/career/todos/{todo_id}")
def update_todo_item(
    todo_id: str,
    request: TodoUpdateRequest,
    thread_id: str = "default",
) -> dict:
    data = execution_store.setdefault(
        thread_id,
        {
            "career_plan": None,
            "todo_items": [],
        },
    )

    items = data.setdefault("todo_items", [])

    for item in items:
        if item.get("id") == todo_id:
            item["completed"] = request.completed
            return {
                "thread_id": thread_id,
                "todo_item": item,
            }

    raise HTTPException(
        status_code=404,
        detail="해야 할 일 항목을 찾을 수 없습니다.",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    config = {
        "configurable": {
            "thread_id": request.thread_id,
        }
    }

    try:
        stored_assets = execution_store.get(
            request.thread_id,
            {},
        )

        result = career_graph.invoke(
            {
                "thread_id": request.thread_id,
                "messages": [
                    HumanMessage(
                        content=request.message
                    )
                ],
                "career_plan": stored_assets.get(
                    "career_plan",
                    {},
                ) or {},
                "todo_items": stored_assets.get(
                    "todo_items",
                    [],
                ),
            },
            config=config,
        )

    except Exception as exc:
        logger.exception(
            "GRAPH EXECUTION ERROR"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Agent 실행 중 오류가 발생했습니다: "
                f"{exc}"
            ),
        ) from exc

    execution_store[request.thread_id] = {
        "career_plan": result.get(
            "career_plan",
            {},
        ) or None,
        "todo_items": result.get(
            "todo_items",
            [],
        ),
    }

    return ChatResponse(
        answer=result.get(
            "final_answer",
            "",
        ),
        thread_id=request.thread_id,
        route=result.get(
            "route",
            "planner",
        ),
        retry_count=result.get(
            "retry_count",
            0,
        ),
        used_tools=result.get(
            "used_tools",
            [],
        ),
        evidence_sources=result.get(
            "evidence_sources",
            [],
        ),
        plan=result.get("plan"),
        structured_result=result.get(
            "structured_result",
        ),
        career_plan=result.get(
            "career_plan",
        ) or None,
        todo_items=result.get(
            "todo_items",
            [],
        ),
    )



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "server:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
