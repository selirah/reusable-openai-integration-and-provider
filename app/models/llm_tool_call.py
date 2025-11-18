from typing import Literal
from pydantic import BaseModel, Field
from app.models.llm_function_call_info import LLMFunctionCallInfo

class LLMToolCall(BaseModel):
    id: str = Field(
        description="The id of the tool call."
    )

    function: LLMFunctionCallInfo = Field(
        description="The function that the llm called."
    )

    type: Literal["function"] = Field(
        default="function",
        description="The type of the tool. Currently, only `function` is supported."
    )