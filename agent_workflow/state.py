from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
    video_path: str
    question: str
    question_category: str
    messages: Annotated[List[BaseMessage], operator.add]
    retry_count: int
    extracted_option: Optional[str]
    spatial_memory: dict | None  # 空间记忆状态，用于存储跨帧/轮次的地标和位置信息
    evidence_summary: Optional[str]  # 视觉证据摘要，供 verifier 校验
    verification_passed: bool  # verifier 是否通过
