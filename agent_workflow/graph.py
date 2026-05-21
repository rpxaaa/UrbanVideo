from langgraph.graph import StateGraph, END
from .state import GraphState
from .reasoner import reasoner_node
from .verifier import verifier_node
from .validator import validator_node


def after_verifier(state: GraphState) -> str:
    """Route after verifier:
    - PASS (or baseline/skip) → validator for option extraction
    - FAIL → back to reasoner for re-reasoning with verifier challenge
    """
    if state.get("verification_passed", True):
        return "validator"
    return "reasoner"


def after_validator(state: GraphState) -> str:
    """Route after validator:
    - Option extracted → END
    - Extraction failed + retries remaining → back to reasoner
    - Extraction failed + retries exhausted → END (give up)
    """
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
    workflow.add_node("verifier", verifier_node)
    workflow.add_node("validator", validator_node)

    workflow.set_entry_point("reasoner")

    workflow.add_edge("reasoner", "verifier")

    workflow.add_conditional_edges(
        "verifier",
        after_verifier,
        {
            "validator": "validator",
            "reasoner": "reasoner",
        }
    )

    workflow.add_conditional_edges(
        "validator",
        after_validator,
        {
            "reasoner": "reasoner",
            END: END,
        }
    )

    return workflow.compile()
