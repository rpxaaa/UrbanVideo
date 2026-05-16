from langgraph.graph import StateGraph, END
from .state import GraphState
from .reasoner import reasoner_node
from .validator import validator_node

def should_continue(state: GraphState):
    extracted_option = state.get("extracted_option")
    if extracted_option is not None:
        return END
    
    retry_count = state.get("retry_count", 0)
    if retry_count >= 2:
        return END
        
    return "reasoner"

def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("validator", validator_node)
    
    workflow.set_entry_point("reasoner")
    
    workflow.add_edge("reasoner", "validator")
    
    workflow.add_conditional_edges(
        "validator",
        should_continue,
        {
            "reasoner": "reasoner",
            END: END
        }
    )
    
    return workflow.compile()
