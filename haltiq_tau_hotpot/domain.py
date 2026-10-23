"""A custom closed-corpus HotpotQA domain for native Tau2 text orchestration.

Only the public question/passages and submitted prediction live in the DB.
Gold references are carried exclusively by evaluator criteria and the caller.
"""
from __future__ import annotations

import json

from pydantic import Field
from tau2.data_model.message import UserMessage
from tau2.data_model.tasks import EnvAssertion, EvaluationCriteria, InitialState, RewardType, Task, UserScenario
from tau2.environment.db import DB
from tau2.environment.environment import Environment
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool
from tau2.registry import registry

from .scoring import normalize_answer


DOMAIN = "haltiq_hotpotqa_custom_v1"
TAU_REVISION = "cc1df47baa1544a34bb8c8e5c4ca72ad223b96a2"
POLICY = (
    "Answer the user's question using only the original public passages in this episode. "
    "Search and read the selected passages, preserve their titles and zero-based sentence IDs, "
    "and cite the sentences supporting the answer. Never invent a tool result or a citation. "
    "Submit exactly one final answer through submit_answer. A submitted answer ends the episode. "
    "A rejected internal draft was not submitted; review and revision must obey the episode limits."
)


def json_copy(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def validate_public(value: dict) -> dict:
    if type(value) is not dict or set(value) != {"id", "question", "type", "level", "context"}:
        raise ValueError("public case requires exactly id/question/type/level/context; no evaluator fields")
    for key in ("id", "question", "type", "level"):
        if type(value[key]) is not str or not value[key].strip():
            raise ValueError("public case scalar fields must be nonempty strings")
    context = value["context"]
    if type(context) is not list or not context:
        raise ValueError("preserve a nonempty original public passage list")
    for paragraph in context:
        if (type(paragraph) is not list or len(paragraph) != 2
                or type(paragraph[0]) is not str or not paragraph[0]
                or type(paragraph[1]) is not list
                or not all(type(sentence) is str for sentence in paragraph[1])):
            raise ValueError("context must preserve original [title, sentences] records")
    return json_copy(value)


def validate_submission(value: dict, public: dict) -> dict:
    if type(value) is not dict or set(value) != {"answer", "supporting_facts"}:
        raise ValueError("submission requires exactly answer and supporting_facts")
    if type(value["answer"]) is not str or not value["answer"].strip():
        raise ValueError("answer must be nonempty text")
    facts = value["supporting_facts"]
    if type(facts) is not list:
        raise ValueError("supporting_facts must be a list")
    for fact in facts:
        if (type(fact) is not list or len(fact) != 2 or type(fact[0]) is not str
                or type(fact[1]) is not int or fact[1] < 0
                or not any(title == fact[0] and fact[1] < len(sentences)
                           for title, sentences in public["context"])):
            raise ValueError("citation must identify an original public title and sentence index")
    return json_copy(value)


def validate_case(case: dict) -> tuple[dict, dict]:
    if type(case) is not dict or set(case) != {"public", "evaluator"}:
        raise ValueError("case must separate public inputs from evaluator references")
    public = validate_public(case["public"])
    gold = validate_submission(case["evaluator"], public)
    return public, gold


def validate_indices(indices: list[int] | None, public: dict) -> list[int]:
    # Production callers supply frozen question-only MiniLM title indices.
    selected = list(range(min(3, len(public["context"])))) if indices is None else indices
    if (type(selected) is not list or not selected or len(set(selected)) != len(selected)
            or any(type(index) is not int or not 0 <= index < len(public["context"]) for index in selected)):
        raise ValueError("passage indices must be a nonempty unique public index list")
    return list(selected)


class HotpotDB(DB):
    public: dict = Field(description="Original public question and passages only")
    passage_indices: list[int] = Field(description="Frozen question-only retrieval order")
    submission: dict | None = None


class HotpotTools(ToolKitBase):
    def __init__(self, db: HotpotDB):
        super().__init__(db)

    @is_tool(ToolType.READ)
    def search_passages(self, query: str) -> str:
        """Return this question's frozen question-only passage ranking.

        Args:
            query: The current question, without an answer or hidden target.
        """
        if query != self.db.public["question"]:
            raise ValueError("initial retrieval requires the exact public question")
        # Tau's generic container serializer stringifies scalar numeric leaves.
        # Already-encoded JSON preserves exact original integer identifiers.
        return json.dumps([{"paragraph_index": index, "title": self.db.public["context"][index][0]}
                           for index in self.db.passage_indices], ensure_ascii=False)

    @is_tool(ToolType.READ)
    def read_passage(self, paragraph_index: int) -> str:
        """Read one selected original public passage and its sentence IDs.

        Args:
            paragraph_index: Original zero-based paragraph index from search_passages.
        """
        if type(paragraph_index) is not int or paragraph_index not in self.db.passage_indices:
            raise ValueError("passage is outside this episode's frozen retrieval selection")
        title, sentences = self.db.public["context"][paragraph_index]
        return json.dumps({"paragraph_index": paragraph_index, "title": title,
            "sentences": [{"sentence_id": i, "text": text} for i, text in enumerate(sentences)]}, ensure_ascii=False)

    @is_tool(ToolType.WRITE)
    def submit_answer(self, answer: str, supporting_facts: list[list]) -> str:
        """Record the final answer and public supporting sentence IDs once.

        Args:
            answer: Final answer text.
            supporting_facts: Pairs of original passage title and zero-based sentence index.
        """
        if self.db.submission is not None:
            raise ValueError("a final answer is already submitted")
        self.db.submission = validate_submission({"answer": answer, "supporting_facts": supporting_facts}, self.db.public)
        return json.dumps({"submitted": True})

    # Deliberately NOT @is_tool: only the native evaluator can supply a reference.
    def assert_answer_em(self, expected_answer: str) -> bool:
        return self.db.submission is not None and normalize_answer(
            self.db.submission["answer"]) == normalize_answer(expected_answer)


def get_environment(*, public_case: dict, passage_indices: list[int] | None = None,
                    solo_mode: bool = False) -> Environment:
    if solo_mode:
        raise ValueError("use the deterministic native user, not solo mode")
    public = validate_public(public_case)
    db = HotpotDB(public=public, passage_indices=validate_indices(passage_indices, public))
    return Environment(DOMAIN, POLICY, tools=HotpotTools(db))


def register_domain() -> None:
    if DOMAIN in registry.get_domains():
        if registry.get_env_constructor(DOMAIN) is not get_environment:
            raise ValueError("custom domain registration conflicts with another implementation")
    else:
        registry.register_domain(get_environment, DOMAIN)


def validate_task(task: Task) -> None:
    criteria = task.evaluation_criteria
    if (criteria is None or criteria.reward_basis != [RewardType.ENV_ASSERTION]
            or not criteria.env_assertions
            or any(assertion.func_name != "assert_answer_em" or assertion.env_type != "assistant"
                   or set(assertion.arguments) != {"expected_answer"}
                   or type(assertion.arguments["expected_answer"]) is not str
                   or not assertion.arguments["expected_answer"].strip()
                   or assertion.assert_value is not True for assertion in criteria.env_assertions)):
        raise ValueError("explicit nonempty answer-EM evaluator assertion required")


def make_task(public: dict, gold: dict) -> Task:
    public = validate_public(public)
    gold = validate_submission(gold, public)
    task = Task(id=public["id"], user_scenario=UserScenario(instructions=public["question"]),
        initial_state=InitialState(message_history=[UserMessage.text(public["question"])]),
        evaluation_criteria=EvaluationCriteria(reward_basis=[RewardType.ENV_ASSERTION],
            env_assertions=[EnvAssertion(env_type="assistant", func_name="assert_answer_em",
                arguments={"expected_answer": gold["answer"]}, assert_value=True)]))
    validate_task(task)
    return task
