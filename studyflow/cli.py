"""
StudyFlow Command Line Interface.

Usage:
  python -m studyflow.cli learn "Topic Name"
  python -m studyflow.cli review
  python -m studyflow.cli digest
"""

import argparse
import asyncio
import os
from datetime import UTC, date, datetime
from pathlib import Path
from uuid import uuid4

from studyflow.agents.quiz_agent import QuizAgent
from studyflow.agents.schemas import (
    AgentRole,
    QuizDifficulty,
    QuizSubmission,
    SubAgentTask,
)
from studyflow.config.llm_provider import TaskWeight, get_provider
from studyflow.db.crud import TopicRepository
from studyflow.db.schema import get_connection, init_database
from studyflow.orchestrator.graph import build_graph
from studyflow.orchestrator.state import StudyFlowState


# Ensure DB is initialized
_conn = init_database()
_repo = TopicRepository(_conn)


def _get_downloads_dir() -> Path:
    """Return the path to the user's Downloads folder."""
    # This works reliably on Windows, macOS, and standard Linux setups.
    return Path.home() / "Downloads"


async def handle_learn(topic: str) -> None:
    """Run the Orchestrator graph and save notes to Downloads."""
    print(f"🚀 Starting StudyFlow for: '{topic}'")
    
    graph = build_graph()
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

    print("🧠 Orchestrator is analyzing prerequisites and generating a study plan...")
    # Run the graph using astream to show progress
    final_state = initial_state.copy()
    try:
        async for step in graph.astream(initial_state):
            for node_name, state_update in step.items():
                final_state.update(state_update)
                
                if node_name == "orchestrator_node":
                    print("✅ Study plan generated. Sourcing materials...")
                elif node_name == "lecture_node":
                    print("🎥 Lecture Agent found and ranked video content.")
                elif node_name == "reading_node":
                    print("📚 Reading Agent curated academic papers and articles.")
                elif node_name == "notes_node":
                    print("📝 Notes Agent synthesized a comprehensive study guide.")
                elif node_name == "quiz_node":
                    print("🎯 Quiz Agent prepared your initial assessment.")
    except Exception as e:
        print(f"❌ Execution failed: {e}")
        return

    # Check for prerequisite warnings
    if final_state.get("prerequisite_warnings"):
        print("\n⚠️  PREREQUISITES MISSING")
        for warning in final_state["prerequisite_warnings"]:
            print(f"  - {warning.message}")
        return

    # Check for general errors
    if final_state.get("errors"):
        print("\n❌ Errors occurred during execution:")
        for err in final_state["errors"]:
            print(f"  - {err}")
        return

    # Check if agents failed (e.g., due to rate limits)
    failed_agents = []
    if final_state.get("notes_result") and final_state["notes_result"].status.value == "failed":
        failed_agents.append(f"Notes Agent Error: {final_state['notes_result'].error}")
    if final_state.get("quiz_result") and final_state["quiz_result"].status.value == "failed":
        failed_agents.append(f"Quiz Agent Error: {final_state['quiz_result'].error}")
        
    if failed_agents:
        print("\n❌ Some agents failed to complete their tasks:")
        for err in failed_agents:
            print(f"  - {err}")

    # Save outputs to Downloads
    downloads = _get_downloads_dir()
    
    # 1. Save Notes
    if final_state.get("notes_result") and final_state["notes_result"].synthesized_notes:
        safe_topic = topic.replace(" ", "_").replace("/", "_")
        notes_path = downloads / f"StudyFlow_Notes_{safe_topic}.md"
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(final_state["notes_result"].synthesized_notes)
        print(f"✅ Notes saved to: {notes_path}")

    # 2. Output Quiz to terminal
    if final_state.get("quiz_result") and final_state["quiz_result"].questions:
        print(f"\n🎯 Initial Quiz for '{topic}' Generated!")
        print("Run 'python -m studyflow.cli review' later to practice.\n")


async def handle_review() -> None:
    """Run interactive terminal quizzes for all due topics."""
    print("📅 Checking for due reviews...")
    
    due_topics = _repo.get_due_reviews(date.today())
    if not due_topics:
        print("🎉 You are all caught up! No reviews due today.")
        return

    print(f"📚 Found {len(due_topics)} topics to review.\n")
    agent = QuizAgent(db_conn=_conn)

    for i, topic in enumerate(due_topics, 1):
        print(f"--- Review {i}/{len(due_topics)}: {topic.title} ---")
        
        # 1. Generate a quiz
        print("Generating questions...")
        task = SubAgentTask(
            agent=AgentRole.QUIZ,
            topic=topic.title,
            instructions="Generate a quick review quiz.",
            parameters={"difficulty": "mcq", "num_questions": 3},
            context={"topic_id": str(topic.id)}
        )
        
        result = await agent.run(task)
        if not result.questions:
            print(f"⚠️ Could not generate quiz for {topic.title}.\n")
            continue

        # 2. Interactive Loop
        user_answers = {}
        for q_idx, q in enumerate(result.questions, 1):
            print(f"\nQ{q_idx}: {q.question_text}")
            if q.options:
                for opt in q.options:
                    print(f"  {opt}")
            
            # Get user input
            answer = input("\nYour answer: ").strip()
            user_answers[str(q.question_id)] = answer

        # 3. Grade and update SM-2
        print("\n📝 Grading your answers...")
        submission = QuizSubmission(
            topic_id=topic.id,
            answers=user_answers,
            difficulty=QuizDifficulty.MCQ,
        )
        
        grading = await agent.grade(submission, result.questions)
        print(f"Score: {grading.percentage}% ({grading.correct_count}/{grading.total_questions})")
        for feedback in grading.feedback:
            print(f" - {feedback}")
            
        print("\nSM-2 schedule updated.\n")


async def handle_digest() -> None:
    """Generate a weekly digest using the LLM and save it to Downloads."""
    print("📊 Generating your Weekly Digest...")
    
    # 1. Gather data
    topics = _repo.get_due_reviews(date.today()) # To see what's due
    # We'll just grab all learned topics for simplicity of the prompt
    c = _conn.cursor()
    c.execute("SELECT title, status, next_review_date FROM topics WHERE status != 'not_started'")
    all_active = [dict(row) for row in c.fetchall()]
    
    c.execute("SELECT t.title, q.quality_score, q.attempted_at FROM quiz_attempts q JOIN topics t ON q.topic_id = t.id ORDER BY q.attempted_at DESC LIMIT 10")
    recent_quizzes = [dict(row) for row in c.fetchall()]

    # 2. Build Prompt
    prompt = (
        "You are the StudyFlow AI. Generate a 'Weekly Digest' for the student.\n\n"
        "Here is their current topic database state:\n"
        f"{all_active}\n\n"
        "Here are their 10 most recent quiz attempts (quality score 0-5):\n"
        f"{recent_quizzes}\n\n"
        "Write a highly encouraging, beautifully formatted Markdown report. "
        "Summarize what they've learned, praise their good quiz scores, gently highlight "
        "areas that need review, and list what they should focus on next week. "
        "Keep it concise (under 400 words)."
    )

    provider = get_provider()
    try:
        digest_md = provider.generate(
            prompt=prompt,
            weight=TaskWeight.HEAVY,
            system_instruction="You are a supportive academic mentor."
        )
    except Exception as e:
        print(f"❌ Failed to generate digest: {e}")
        return

    # 3. Save to Downloads
    downloads = _get_downloads_dir()
    date_str = datetime.now().strftime("%Y-%m-%d")
    digest_path = downloads / f"StudyFlow_WeeklyDigest_{date_str}.md"
    
    with open(digest_path, "w", encoding="utf-8") as f:
        f.write(digest_md)
        
    print(f"✅ Weekly Digest saved to: {digest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="StudyFlow CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `learn`
    learn_parser = subparsers.add_parser("learn", help="Start learning a new topic.")
    learn_parser.add_argument("topic", type=str, help="The topic to learn")

    # `review`
    subparsers.add_parser("review", help="Run spaced-repetition quizzes for due topics.")

    # `digest`
    subparsers.add_parser("digest", help="Generate a weekly progress digest.")

    args = parser.parse_args()

    if args.command == "learn":
        asyncio.run(handle_learn(args.topic))
    elif args.command == "review":
        asyncio.run(handle_review())
    elif args.command == "digest":
        asyncio.run(handle_digest())


if __name__ == "__main__":
    main()
