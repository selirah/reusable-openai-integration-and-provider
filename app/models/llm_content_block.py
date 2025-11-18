from typing import Optional, Literal
from pydantic import BaseModel, Field

LLMContentBlockType = Literal[
    "text",
    "image",
    "audio",
    "file"
]

class LLMContentBlock(BaseModel):
    user_id: Optional[str] = Field(
        default=None,
        description="ID of the user that this content block is associated with."
    )

    type: LLMContentBlockType = Field(
        default="text",
        description="The type of content this part is. This is used to determine how "
                    "to render the content or how to display it on the frontend."
    )

    text: str = Field(
        default="",
        description="If this is a text content block, this will contain the text.",
    )

    image_id: str = Field(
        default="",
        description="If this is an image content block, this will contain the image ID within our image system.",
    )

    audio_id: str = Field(
        default="",
        description="If this is an audio content block, this will contain the ID of the audio file.",
    )

    file_id: str = Field(
        default="",
        description="If this is a file content block, this will contain the ID of the file.",
    )

