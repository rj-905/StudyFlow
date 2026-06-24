"""
Phase 4 tests: LangGraph Orchestration.
"""

from __future__ import annotations

import operator
from unittest.mock import AsyncMock, patch

import pytest
from langgraph.graph import END

from studyflow.agents.schemas import (
    AgentRole,
    LectureResult,
    NotesResult,
    PrerequisiteWarning,
    QuizResult,
    ReadingResult,
    StudyPlan,
    SubAgentTask,
    TaskStatus,
)
from studyflow.db.crud import TopicRepository
from studyflow.db.schema import init_database
from studyflow.orchestrator.graph import build_graph
from studyflow.orchestrator.state import StudyFlowState


@pytest.fixture
def db_conn():
    conn = init_database(":memory:")
    yield conn
    conn.close()


@pytest.fixture
def repo(db_conn):
    return TopicRepository(db_conn)


@pytest.mark.asyncio
async def test_graph_compiles_and_routes_missing_prerequisites(repo):
    """
    Test that if prerequisites are missing, the graph short-circuits to END.
    """
    from studyflow.orchestrator.orchestrator_agent import OrchestratorAgent

    # Mock Orchestrator to return a prerequisite warning
    mock_orchestrator = AsyncMock()
    mock_orchestrator.run.return_value = {
        "prerequisite_warnings": [
            PrerequisiteWarning(
                topic="Advanced Topic",
                missing_prerequisites=["Basics"],
                message="Missing basics.",
            )
        ]
    }

    graph = build_graph()
    
    with patch(
        "studyflow.orchestrator.graph.OrchestratorAgent",
        return_value=mock_orchestrator,
    ), patch(
        "studyflow.orchestrator.graph.get_provider"
    ):
        initial_state: StudyFlowState = {
            "topic": "Advanced Topic",
            "plan": None,
            "prerequisite_warnings": [],
            "lecture_result": None,
            "reading_result": None,
            "notes_result": None,
            "quiz_result": None,
            "errors": [],
        }

        # Run graph
        result = await graph.ainvoke(initial_state)

        # Graph should have returned warnings and skipped the sub-agents
        assert len(result["prerequisite_warnings"]) == 1
        assert result["prerequisite_warnings"][0].missing_prerequisites == ["Basics"]
        assert result["plan"] is None
        assert result["lecture_result"] is None


@pytest.mark.asyncio
async def test_graph_full_execution_hybrid_model():
    """
    Test the hybrid execution model (parallel sourcing -> sequential synthesis).
    """
    graph = build_graph()

    topic = "Amortized Analysis"

    # Mock the LLM / Agents to avoid network calls
    mock_plan = StudyPlan(
        topic=topic,
        tasks=[
            SubAgentTask(agent=AgentRole.LECTURE, topic=topic, instructions=""),
            SubAgentTask(agent=AgentRole.READING, topic=topic, instructions=""),
            SubAgentTask(agent=AgentRole.NOTES, topic=topic, instructions=""),
            SubAgentTask(agent=AgentRole.QUIZ, topic=topic, instructions=""),
        ],
    )

    # 1. Orchestrator returns the plan
    mock_orch = AsyncMock()
    mock_orch.run.return_value = {"plan": mock_plan}

    # 2. Lecture Agent returns results
    mock_lecture = AsyncMock()
    l_res = LectureResult(
        task_id=mock_plan.tasks[0].task_id,
        topic=topic,
        status=TaskStatus.COMPLETED,
        videos=[],
    )
    mock_lecture.run.return_value = l_res

    # 3. Reading Agent returns results
    mock_reading = AsyncMock()
    r_res = ReadingResult(
        task_id=mock_plan.tasks[1].task_id,
        topic=topic,
        status=TaskStatus.COMPLETED,
        resources=[],
    )
    mock_reading.run.return_value = r_res

    # 4. Notes Agent checks if context was passed and returns notes
    mock_notes = AsyncMock()
    n_res = NotesResult(
        task_id=mock_plan.tasks[2].task_id,
        topic=topic,
        status=TaskStatus.COMPLETED,
        synthesized_notes="Done",
        prior_context_used=[],
        chunk_ids_stored=[],
    )
    mock_notes.run.return_value = n_res

    # 5. Quiz Agent
    mock_quiz = AsyncMock()
    q_res = QuizResult(
        task_id=mock_plan.tasks[3].task_id,
        topic=topic,
        status=TaskStatus.COMPLETED,
        questions=[],
    )
    mock_quiz.run.return_value = q_res

    # Patch the agent classes inside graph.py
    with patch("studyflow.orchestrator.graph.OrchestratorAgent", return_value=mock_orch), \
         patch("studyflow.orchestrator.graph.LectureAgent", return_value=mock_lecture), \
         patch("studyflow.orchestrator.graph.ReadingAgent", return_value=mock_reading), \
         patch("studyflow.orchestrator.graph.NotesAgent", return_value=mock_notes), \
         patch("studyflow.orchestrator.graph.QuizAgent", return_value=mock_quiz), \
         patch("studyflow.orchestrator.graph.get_provider"):

        initial_state: StudyFlowState = {
            "topic": topic,
            "plan": None,
            "prerequisite_warnings": [],
            "lecture_result": None,
            "reading_result": None,
            "notes_result": None,
            "quiz_result": None,
            "errors": [],
        }

        # Run graph
        result = await graph.ainvoke(initial_state)

        # Verify end state
        assert result["plan"] is not None
        assert result["lecture_result"].status == TaskStatus.COMPLETED
        assert result["reading_result"].status == TaskStatus.COMPLETED
        assert result["notes_result"].status == TaskStatus.COMPLETED
        assert result["quiz_result"].status == TaskStatus.COMPLETED

        # Verify that Notes Agent was called AFTER Lecture/Reading
        # We can inspect the task context that was passed to Notes Agent in the graph
        mock_notes.run.assert_called_once()
        task_passed_to_notes: SubAgentTask = mock_notes.run.call_args[0][0]
        assert "resource_summaries" in task_passed_to_notes.context
