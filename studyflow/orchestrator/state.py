"""
State schema for the StudyFlow LangGraph.

This defines the TypedDict that gets passed between all nodes in the graph.
"""

import operator
from typing import Annotated, TypedDict

from studyflow.agents.schemas import (
    LectureResult,
    NotesResult,
    PrerequisiteWarning,
    QuizResult,
    ReadingResult,
    StudyPlan,
)


class StudyFlowState(TypedDict):
    """
    The global state for the StudyFlow graph execution.

    Keys:
        topic: The topic the user wants to study.
        plan: The generated study plan containing tasks for each sub-agent.
        prerequisite_warnings: Any warnings about missing prerequisites.
        lecture_result: Result from the Lecture Agent.
        reading_result: Result from the Reading Agent.
        notes_result: Result from the Notes Agent.
        quiz_result: Result from the Quiz Agent.
        errors: A list of aggregated errors across all nodes (appended to).
    """

    topic: str
    plan: StudyPlan | None
    prerequisite_warnings: list[PrerequisiteWarning]
    lecture_result: LectureResult | None
    reading_result: ReadingResult | None
    notes_result: NotesResult | None
    quiz_result: QuizResult | None
    errors: Annotated[list[str], operator.add]
