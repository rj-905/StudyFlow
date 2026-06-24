"""
Orchestrator Node for LangGraph.

This node is the entry point of the graph. It checks for prerequisites and generates a StudyPlan.
"""

import json
from uuid import uuid4

from studyflow.agents.schemas import AgentRole, PrerequisiteWarning, StudyPlan, SubAgentTask
from studyflow.config.llm_provider import GeminiProvider, TaskWeight
from studyflow.db.crud import TopicRepository
from studyflow.db.schema import get_connection
from studyflow.orchestrator.state import StudyFlowState


class OrchestratorAgent:
    """The central planner for StudyFlow."""

    def __init__(self, llm_provider: GeminiProvider, repo: TopicRepository = None):
        self.llm = llm_provider
        self.repo = repo or TopicRepository(get_connection())

    async def run(self, state: StudyFlowState) -> dict:
        """
        Executes the Orchestrator logic.
        
        Args:
            state: The current StudyFlowState containing at least a 'topic'.
            
        Returns:
            A dict containing state updates (e.g., 'plan' or 'prerequisite_warnings').
        """
        topic = state["topic"]
        errors = []

        # 1. Check prerequisites
        try:
            missing = self.repo.get_missing_prerequisites(topic)
            if missing:
                warning = PrerequisiteWarning(
                    topic=topic,
                    missing_prerequisites=missing,
                    message=f"You must learn the following topics first: {', '.join(missing)}",
                )
                return {"prerequisite_warnings": [warning]}
        except Exception as e:
            errors.append(f"Error checking prerequisites: {e}")
            # Non-fatal, continue

        # 2. Generate StudyPlan
        prompt = f"""
        You are the Orchestrator Agent for StudyFlow.
        Create a study plan for the topic: "{topic}".
        Generate tasks for four sub-agents: lecture, reading, notes, quiz.
        Return the result as JSON matching the StudyPlan schema.
        """
        try:
            # We construct a default plan manually instead of relying entirely on LLM schema generation,
            # to ensure strict adherence to our parallel fan-out structure.
            plan = StudyPlan(
                topic=topic,
                tasks=[
                    SubAgentTask(
                        agent=AgentRole.LECTURE,
                        topic=topic,
                        instructions=f"Find highly relevant video lectures on {topic}.",
                    ),
                    SubAgentTask(
                        agent=AgentRole.READING,
                        topic=topic,
                        instructions=f"Find high-quality academic papers or reading materials on {topic}.",
                    ),
                    SubAgentTask(
                        agent=AgentRole.NOTES,
                        topic=topic,
                        instructions=f"Synthesize the provided video and reading materials into comprehensive notes on {topic}.",
                    ),
                    SubAgentTask(
                        agent=AgentRole.QUIZ,
                        topic=topic,
                        instructions=f"Generate a quiz to test knowledge of {topic}.",
                        parameters={"difficulty": "mcq"}
                    ),
                ]
            )
            return {"plan": plan, "errors": errors}
        except Exception as e:
            errors.append(f"Error creating StudyPlan: {e}")
            return {"errors": errors}
