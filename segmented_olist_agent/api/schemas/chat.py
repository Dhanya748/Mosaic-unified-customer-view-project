from typing import List, Optional
from pydantic import BaseModel

class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    message: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    messages: List[ChatMessage]

class ConversationCreateResponse(BaseModel):
    id: str

class ConversationHistoryResponse(BaseModel):
    id: str
    messages: List[ChatMessage]
