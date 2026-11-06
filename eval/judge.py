"""LLM-as-judge grading of generated answers, built on LangChain structured output."""

from __future__ import annotations

import re
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, SecretStr

REFUSAL_REFERENCE = (
    "The documents do not contain this information. A correct answer says so "
    "and does not answer from general knowledge."
)

_PROMPT = ChatPromptTemplate.from_template(
    """You grade answers produced by a document question-answering system.

Question: {question}

Reference answer: {reference}

Passages the system was given:
{passages}

System answer: {answer}"""
)

_CITATION = re.compile(r"\[(\d+)\]")


class Verdict(BaseModel):
    correct: Literal["yes", "partial", "no"] = Field(
        description=(
            "yes if the system answer conveys the reference answer, partial if it is "
            "incomplete or partly wrong, no otherwise. Extra detail supported by the "
            "passages does not reduce the grade."
        )
    )
    faithful: bool = Field(
        description=(
            "true if every factual claim in the system answer is supported by the "
            "passages. An answer that only says the information is unavailable is faithful."
        )
    )
    reason: str = Field(description="One short sentence explaining the grade.")


def build_judge(model: str, api_key: SecretStr) -> Runnable:
    llm = ChatGroq(model=model, api_key=api_key, temperature=0, max_retries=8)
    return _PROMPT | llm.with_structured_output(Verdict)


def citations_valid(answer: str, source_count: int) -> bool:
    cited = {int(n) for n in _CITATION.findall(answer)}
    return bool(cited) and all(1 <= n <= source_count for n in cited)


async def judge(
    chain: Runnable, *, question: str, reference: str, passages: list[str], answer: str
) -> Verdict:
    rendered = "\n\n".join(f"[{i}] {text}" for i, text in enumerate(passages, start=1))
    return await chain.ainvoke(
        {
            "question": question,
            "reference": reference,
            "passages": rendered or "(none)",
            "answer": answer,
        }
    )
