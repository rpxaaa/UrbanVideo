import re
from langchain_core.messages import HumanMessage
from .state import GraphState

def validator_node(state: GraphState):
    messages = state.get("messages", [])
    if not messages:
        return {"extracted_option": None}
        
    last_message = messages[-1].content
    if not isinstance(last_message, str):
        if isinstance(last_message, list) and len(last_message) > 0 and "text" in last_message[0]:
            last_message = last_message[0]["text"]
        else:
            last_message = str(last_message)
            
    retry_count = state.get("retry_count", 0)
    
    pattern = r"Option\s*:\s*[\[\s]*([A-Ga-g])"
    match = re.search(pattern, last_message)
    
    if match:
        extracted_option = match.group(1).upper()
        print(f"  -> [Validator] Successfully extracted option: {extracted_option}")
        return {"extracted_option": extracted_option}
    else:
        print(f"  -> [Validator] Failed to extract option from LLM response: {repr(last_message[:100])}...")
        if retry_count < 2:
            print(f"  -> [Validator] Retrying... (Attempt {retry_count + 1}/2)")
            error_prompt = "Your previous response didn't contain the option in the correct format. Please provide your answer in the format 'Option: [X]' where X is the letter of the correct choice."
            return {
                "messages": [HumanMessage(content=error_prompt)],
                "retry_count": retry_count + 1
            }
        else:
            return {"extracted_option": ""}
