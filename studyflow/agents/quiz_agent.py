"""
Quiz Agent: Tiered question generation and LLM-assisted grading.

Pipeline — Generation (Gemini 2.5 Pro)
---------------------------------------
1. Receive topic + difficulty parameter from ``SubAgentTask``.
2. Generate tiered questions: MCQ, Conceptual, Applied.
3. Return ``QuizResult`` with structured ``QuizQuestion`` objects.

Pipeline — Grading (Gemini 3.5 Flash)
--------------------------------------
1. Receive ``QuizSubmission`` with user answers.
2. Compare against correct answers with LLM semantic matching.
3. Compute SM-2 quality score (0–5) from percentage.
4. Call ``SM2Engine.update()`` to recalculate scheduling.
5. Persist ``QuizAttemptRecord`` to SQLite.
6. Return ``QuizGradingResult``.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date
from typing import Any, Optional
from uuid import UUID, uuid4

from studyflow.agents.schemas import (
    AgentRole,
    QuizDifficulty,
    QuizGradingResult,
    QuizQuestion,
    QuizResult,
    QuizSubmission,
    SubAgentTask,
    TaskStatus,
)
from studyflow.config.llm_provider import GeminiProvider, TaskWeight, get_provider
from studyflow.db.crud import TopicRepository
from studyflow.models import DifficultyLevel, QuizAttemptRecord, TopicStatus
from studyflow.scheduler.sm2_engine import SM2Engine


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_QUIZ_GEN_SYSTEM = """You are an expert educator who creates assessment 
questions to test understanding of academic topics. Generate questions at 
the specified difficulty tier.

DIFFICULTY TIERS:
- MCQ: Multiple-choice with 4 options. Include plausible distractors.
- CONCEPTUAL: Short-answer questions testing understanding of core concepts.
- APPLIED: Problem-solving questions requiring application of the concept.

RULES:
1. Questions must directly test the specified topic.
2. Each question must have a clear, unambiguous correct answer.
3. Provide a brief but helpful explanation for each correct answer.
4. For MCQ, always provide exactly 4 options labeled A, B, C, D.
5. Vary the difficulty within the tier — don't make all questions identical.

Return a JSON object with a "questions" array. Each question has:
- "difficulty": "mcq" | "conceptual" | "applied"
- "question_text": the question body
- "options": ["A) ...", "B) ...", "C) ...", "D) ..."] (null for non-MCQ)
- "correct_answer": the correct answer text
- "explanation": brief explanation

Return ONLY the JSON object."""

_GRADING_SYSTEM = """You are a fair, precise grader. For each question-answer 
pair, determine if the student's answer is correct.

RULES:
1. Accept semantically equivalent answers (e.g., "O(n log n)" and 
   "n log n" are equivalent).
2. For MCQ, the answer must match the correct option.
3. For conceptual/applied, accept answers that demonstrate understanding 
   even if wording differs.
4. Be generous with partial credit for partially correct answers.

Return a JSON object with:
- "results": array of {"question_id": str, "is_correct": bool, "feedback": str}
- "total_correct": int
- "total_questions": int

Return ONLY the JSON object."""


# ---------------------------------------------------------------------------
# Quiz Agent
# ---------------------------------------------------------------------------

class QuizAgent:
    """
    Generates tiered quiz questions and grades user responses.

    Parameters
    ----------
    provider : GeminiProvider | None
        LLM provider.  Uses Pro for generation, Flash for grading.
    db_conn : sqlite3.Connection | None
        Database connection for persisting quiz attempts and
        updating SM-2 scheduling parameters.
    sm2_engine : SM2Engine | None
        Spaced repetition engine instance.
    """

    def __init__(
        self,
        provider: Optional[GeminiProvider] = None,
        db_conn: Optional[sqlite3.Connection] = None,
        sm2_engine: Optional[SM2Engine] = None,
    ) -> None:
        self._provider = provider or get_provider()
        self._conn = db_conn
        self._sm2 = sm2_engine or SM2Engine()

    # ── Question Generation ───────────────────────────────────────────────

    async def run(self, task: SubAgentTask) -> QuizResult:
        """
        Generate quiz questions for a topic.

        Parameters
        ----------
        task : SubAgentTask
            Must have ``agent == AgentRole.QUIZ``.
            ``task.parameters`` may contain:
            - ``"difficulty"`` (str): "mcq", "conceptual", "applied",
              or "mixed" (default).
            - ``"num_questions"`` (int): number of questions (default 5).

        Returns
        -------
        QuizResult
        """
        topic = task.topic
        difficulty = task.parameters.get("difficulty", "mixed")
        num_questions = task.parameters.get("num_questions", 5)

        try:
            prompt = self._build_generation_prompt(topic, difficulty, num_questions)

            response = self._provider.generate_json(
                prompt=prompt,
                weight=TaskWeight.HEAVY,
                system_instruction=_QUIZ_GEN_SYSTEM,
                temperature=0.6,
            )

            questions = self._parse_questions(response, difficulty)

            return QuizResult(
                task_id=task.task_id,
                topic=topic,
                status=TaskStatus.COMPLETED,
                questions=questions,
            )

        except Exception as e:
            return QuizResult(
                task_id=task.task_id,
                topic=topic,
                status=TaskStatus.FAILED,
                error=str(e),
            )

    # ── Grading ───────────────────────────────────────────────────────────

    async def grade(
        self,
        submission: QuizSubmission,
        questions: list[QuizQuestion],
    ) -> QuizGradingResult:
        """
        Grade a quiz submission using Gemini 3.5 Flash.

        After grading, updates the topic's SM-2 scheduling parameters
        and persists a ``QuizAttemptRecord``.

        Parameters
        ----------
        submission : QuizSubmission
            User's answers.
        questions : list[QuizQuestion]
            The original quiz questions (for comparison).

        Returns
        -------
        QuizGradingResult
        """
        try:
            # Build grading prompt
            qa_pairs = self._build_grading_prompt(submission, questions)

            grading_response = self._provider.generate_json(
                prompt=qa_pairs,
                weight=TaskWeight.LIGHT,
                system_instruction=_GRADING_SYSTEM,
                temperature=0.1,
            )

            total_correct = grading_response.get("total_correct", 0)
            total_questions = grading_response.get("total_questions", len(questions))

            # Compute percentage and SM-2 quality score
            percentage = (
                (total_correct / total_questions * 100)
                if total_questions > 0
                else 0.0
            )
            quality_score = self._percentage_to_quality(percentage)

            # Extract per-question feedback
            feedback = [
                r.get("feedback", "")
                for r in grading_response.get("results", [])
            ]

            # Map QuizDifficulty to DifficultyLevel for DB
            difficulty_map = {
                QuizDifficulty.MCQ: DifficultyLevel.MCQ,
                QuizDifficulty.CONCEPTUAL: DifficultyLevel.CONCEPTUAL,
                QuizDifficulty.APPLIED: DifficultyLevel.APPLIED,
            }
            db_difficulty = difficulty_map.get(
                submission.difficulty, DifficultyLevel.MCQ
            )

            # Persist to DB and update SM-2 if connection is available
            if self._conn is not None:
                self._update_scheduling(
                    topic_id=submission.topic_id,
                    quality_score=quality_score,
                    difficulty=db_difficulty,
                )

            return QuizGradingResult(
                topic_id=submission.topic_id,
                total_questions=total_questions,
                correct_count=total_correct,
                percentage=round(percentage, 1),
                quality_score=quality_score,
                difficulty=submission.difficulty,
                feedback=feedback,
            )

        except Exception as e:
            # Return a failed grading with score 0
            return QuizGradingResult(
                topic_id=submission.topic_id,
                total_questions=len(questions),
                correct_count=0,
                percentage=0.0,
                quality_score=0,
                difficulty=submission.difficulty,
                feedback=[f"Grading error: {str(e)}"],
            )

    # ── SM-2 Integration ──────────────────────────────────────────────────

    def _update_scheduling(
        self,
        topic_id: UUID,
        quality_score: int,
        difficulty: DifficultyLevel,
    ) -> None:
        """
        Update the topic's SM-2 scheduling parameters after a quiz.

        1. Record the quiz attempt.
        2. Recalculate SM-2 parameters.
        3. Update the topic record in SQLite.
        """
        repo = TopicRepository(self._conn)
        topic = repo.get_topic_by_id(topic_id)

        if topic is None:
            return

        # Record the attempt
        attempt = QuizAttemptRecord(
            topic_id=topic_id,
            difficulty=difficulty,
            quality_score=quality_score,
        )
        repo.insert_quiz_attempt(attempt)

        # Recalculate SM-2 parameters
        sm2_result = self._sm2.update(
            quality_score=quality_score,
            easiness_factor=topic.easiness_factor,
            interval_days=topic.interval_days,
            repetition_number=topic.repetition_number,
            review_date=date.today(),
        )

        # Update the topic
        topic.easiness_factor = sm2_result.easiness_factor
        topic.interval_days = sm2_result.interval_days
        topic.repetition_number = sm2_result.repetition_number
        topic.next_review_date = sm2_result.next_review_date
        topic.latest_performance_score = float(quality_score)
        topic.status = TopicStatus.REVIEWING
        repo.upsert_topic(topic)

    # ── Private helpers ───────────────────────────────────────────────────

    @staticmethod
    def _percentage_to_quality(percentage: float) -> int:
        """
        Map quiz percentage to SM-2 quality score (0–5).

        +------------+-------+
        | Percentage | Score |
        +============+=======+
        | 100%       |   5   |
        | 80–99%     |   4   |
        | 60–79%     |   3   |
        | 40–59%     |   2   |
        | 20–39%     |   1   |
        | <20%       |   0   |
        +------------+-------+
        """
        if percentage >= 100:
            return 5
        elif percentage >= 80:
            return 4
        elif percentage >= 60:
            return 3
        elif percentage >= 40:
            return 2
        elif percentage >= 20:
            return 1
        else:
            return 0

    def _build_generation_prompt(
        self, topic: str, difficulty: str, num_questions: int
    ) -> str:
        """Build the question generation prompt."""
        if difficulty == "mixed":
            tier_instruction = (
                f"Generate {num_questions} questions with a mix of all "
                f"three tiers (MCQ, Conceptual, Applied)."
            )
        else:
            tier_instruction = (
                f"Generate {num_questions} questions at the "
                f"'{difficulty}' difficulty tier."
            )

        return (
            f"Topic: {topic}\n\n"
            f"{tier_instruction}\n\n"
            f"Ensure questions test genuine understanding, not just "
            f"surface-level memorisation."
        )

    @staticmethod
    def _build_grading_prompt(
        submission: QuizSubmission,
        questions: list[QuizQuestion],
    ) -> str:
        """Build the grading prompt with Q&A pairs."""
        pairs: list[str] = []
        for q in questions:
            q_id = str(q.question_id)
            user_answer = submission.answers.get(q_id, "[No answer provided]")
            pairs.append(
                f"Question ID: {q_id}\n"
                f"Question: {q.question_text}\n"
                f"Correct Answer: {q.correct_answer}\n"
                f"Student Answer: {user_answer}"
            )

        return "Grade the following answers:\n\n" + "\n---\n".join(pairs)

    @staticmethod
    def _parse_questions(
        response: dict[str, Any], default_difficulty: str
    ) -> list[QuizQuestion]:
        """Parse the LLM's JSON response into QuizQuestion objects."""
        raw_questions = response.get("questions", [])
        parsed: list[QuizQuestion] = []

        difficulty_map = {
            "mcq": QuizDifficulty.MCQ,
            "conceptual": QuizDifficulty.CONCEPTUAL,
            "applied": QuizDifficulty.APPLIED,
        }

        for q in raw_questions:
            diff_str = q.get("difficulty", default_difficulty).lower()
            difficulty = difficulty_map.get(diff_str, QuizDifficulty.MCQ)

            options = q.get("options")
            if options and not isinstance(options, list):
                options = None

            parsed.append(
                QuizQuestion(
                    difficulty=difficulty,
                    question_text=q.get("question_text", ""),
                    options=options,
                    correct_answer=q.get("correct_answer", ""),
                    explanation=q.get("explanation", ""),
                )
            )

        return parsed
