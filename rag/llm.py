"""Claude 호출: 질문 답변, 강의 요약, 퀴즈 생성."""

import os
from collections.abc import AsyncIterator
from functools import lru_cache

import anthropic
from pydantic import BaseModel

from rag.store import Hit

MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")


@lru_cache(maxsize=1)
def client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()

# 거절(refusal) 시 서버에서 다른 모델로 자동 재시도
FALLBACK = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}

FORMAT_RULES = """답변은 휴대폰 텔레그램 메시지로 보여집니다.
- 짧은 문단과 "- " 목록 위주로 쓰고, 표와 # 제목은 쓰지 마세요.
- 강조는 **굵게**만 사용하세요."""

ANSWER_SYSTEM = f"""당신은 대학생의 강의 자료 공부를 돕는 튜터입니다.
- <context>에 있는 강의 자료를 근거로 답하고, 근거가 된 문단 번호를 [1], [2]처럼 표시하세요.
- 자료에 없는 내용은 "강의 자료에는 없는 내용"이라고 먼저 밝힌 뒤, 일반 지식으로 짧게 보충하세요.
- 개념은 쉬운 말로 풀고, 필요하면 간단한 예시를 드세요.
{FORMAT_RULES}"""

SUMMARY_SYSTEM = f"""당신은 대학생의 시험 공부를 돕는 튜터입니다. 강의 자료 한 개를 요약합니다.
- 핵심 개념을 5~10개 항목으로 정리하고, 각 항목에 (p.페이지)를 붙이세요.
- 마지막에 "시험에 나올 만한 포인트"를 3개 적으세요.
{FORMAT_RULES}"""


def build_context(hits: list[Hit]) -> str:
    parts = [
        f'<doc index="{i}" source="{h.source}" page="{h.page}">\n{h.text}\n</doc>'
        for i, h in enumerate(hits, start=1)
    ]
    return "<context>\n" + "\n".join(parts) + "\n</context>"


async def _stream(system: str, messages: list[dict]) -> AsyncIterator[str]:
    async with client().beta.messages.stream(
        model=MODEL,
        max_tokens=8000,
        system=system,
        output_config={"effort": "low"},
        messages=messages,
        **FALLBACK,
    ) as stream:
        async for text in stream.text_stream:
            yield text
        final = await stream.get_final_message()
    if final.stop_reason == "refusal":
        yield "\n\n(모델이 이 요청에 대한 답변을 거절했습니다.)"
    elif final.stop_reason == "max_tokens":
        yield "\n\n(답변이 길어서 중간에 잘렸습니다.)"


def answer(question: str, hits: list[Hit], history: list[dict]) -> AsyncIterator[str]:
    """history: 이전 대화 [{"role": "user"|"assistant", "content": str}, ...] (context는 제외)."""
    user = f"{build_context(hits)}\n\n질문: {question}"
    return _stream(ANSWER_SYSTEM, [*history, {"role": "user", "content": user}])


def summarize(source: str, pages: list[tuple[int, str]]) -> AsyncIterator[str]:
    body = "\n".join(f'<page number="{p}">{t}</page>' for p, t in pages)
    user = f'<lecture file="{source}">\n{body}\n</lecture>\n\n이 강의 자료를 요약해 주세요.'
    return _stream(SUMMARY_SYSTEM, [{"role": "user", "content": user}])


class QuizItem(BaseModel):
    question: str
    choices: list[str]  # 보기 4개
    answer_index: int  # 0부터 시작
    explanation: str
    source: str
    page: int


class Quiz(BaseModel):
    items: list[QuizItem]


async def make_quiz(hits: list[Hit], n: int) -> list[QuizItem]:
    prompt = (
        f"{build_context(hits)}\n\n"
        f"위 강의 자료로 4지선다 문제 {n}개를 만드세요. "
        "단순 암기보다 개념 이해를 확인하는 문제로, 오답 보기도 그럴듯하게 만드세요. "
        "explanation은 1~2문장, source와 page는 근거 자료의 값을 그대로 쓰세요."
    )
    response = await client().messages.parse(
        model=MODEL,
        max_tokens=8000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
        output_format=Quiz,
    )
    if response.parsed_output is None:
        return []
    return [q for q in response.parsed_output.items if len(q.choices) >= 2 and 0 <= q.answer_index < len(q.choices)]
