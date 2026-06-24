"""
LangGraph definition for StudyFlow.

Implements the Hybrid Execution Model:
1. Orchestrator checks prerequisites.
2. If clear, fan-out to Lecture and Reading agents in parallel.
3. Fan-in to Notes agent (which uses their outputs).
4. Finally, route to Quiz agent.
"""

from typing import cast

from langgraph.graph import END, START, StateGraph

from studyflow.agents.lecture_agent import LectureAgent
from studyflow.agents.notes_agent import NotesAgent
from studyflow.agents.quiz_agent import QuizAgent
from studyflow.agents.reading_agent import ReadingAgent
from studyflow.agents.schemas import AgentRole
from studyflow.config.llm_provider import get_provider
from studyflow.orchestrator.orchestrator_agent import OrchestratorAgent
from studyflow.orchestrator.state import StudyFlowState


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

async def orchestrator_node(state: StudyFlowState) -> dict:
    agent = OrchestratorAgent(llm_provider=get_provider())
    return await agent.run(state)


async def lecture_node(state: StudyFlowState) -> dict:
    agent = LectureAgent()
    task = next(t for t in state["plan"].tasks if t.agent == AgentRole.LECTURE)
    res = await agent.run(task)
    return {"lecture_result": res}


async def reading_node(state: StudyFlowState) -> dict:
    agent = ReadingAgent()
    task = next(t for t in state["plan"].tasks if t.agent == AgentRole.READING)
    res = await agent.run(task)
    return {"reading_result": res}


async def notes_node(state: StudyFlowState) -> dict:
    agent = NotesAgent()
    task = next(t for t in state["plan"].tasks if t.agent == AgentRole.NOTES)
    
    # Fan-in: synthesize context from sourcing agents
    summaries = []
    if state.get("lecture_result") and state["lecture_result"].videos:
        summaries.append("### Video Resources")
        for v in state["lecture_result"].videos[:3]:
            desc = (v.description or "")[:200]
            summaries.append(f"- **{v.title}**: {desc}")
            
    if state.get("reading_result") and state["reading_result"].resources:
        summaries.append("### Text Resources")
        for r in state["reading_result"].resources[:3]:
            abs_text = (r.abstract or "")[:200]
            summaries.append(f"- **{r.title}**: {abs_text}")
            
    task.context["resource_summaries"] = summaries
    
    res = await agent.run(task)
    return {"notes_result": res}


async def quiz_node(state: StudyFlowState) -> dict:
    agent = QuizAgent()
    task = next(t for t in state["plan"].tasks if t.agent == AgentRole.QUIZ)
    res = await agent.run(task)
    return {"quiz_result": res}


# ---------------------------------------------------------------------------
# Routing Logic
# ---------------------------------------------------------------------------

def route_from_orchestrator(state: StudyFlowState) -> list[str]:
    """
    Conditional routing from Orchestrator.
    If prerequisites are missing, end the graph.
    Otherwise, fan-out to Lecture and Reading in parallel.
    """
    if state.get("prerequisite_warnings"):
        return [END]
    
    # LangGraph allows returning a list of nodes to execute in parallel
    return ["lecture_node", "reading_node"]


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    builder = StateGraph(StudyFlowState)
    
    # Add nodes
    builder.add_node("orchestrator", orchestrator_node)
    builder.add_node("lecture_node", lecture_node)
    builder.add_node("reading_node", reading_node)
    builder.add_node("notes_node", notes_node)
    builder.add_node("quiz_node", quiz_node)
    
    # Edges
    builder.add_edge(START, "orchestrator")
    
    # Orchestrator conditionally routes to [lecture, reading] OR END
    builder.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        ["lecture_node", "reading_node", END]
    )
    
    # Fan-in: both sourcing nodes point to Notes
    builder.add_edge("lecture_node", "notes_node")
    builder.add_edge("reading_node", "notes_node")
    
    # Notes points to Quiz
    builder.add_edge("notes_node", "quiz_node")
    
    # Quiz points to END
    builder.add_edge("quiz_node", END)
    
    return builder.compile()
