import asyncio
from studyflow.cli import build_graph, StudyFlowState

async def test():
    graph = build_graph()
    initial_state: StudyFlowState = {
        "topic": "Partition Function",
        "plan": None,
        "prerequisite_warnings": [],
        "lecture_result": None,
        "reading_result": None,
        "notes_result": None,
        "quiz_result": None,
        "errors": [],
    }
    
    final_state = initial_state.copy()
    async for step in graph.astream(initial_state):
        for node_name, state_update in step.items():
            print(f"Update from {node_name}: {state_update.keys()}")
            final_state.update(state_update)
            
    print("Final state notes_result:", final_state.get("notes_result"))
    print("Final state quiz_result:", final_state.get("quiz_result"))

if __name__ == "__main__":
    asyncio.run(test())
