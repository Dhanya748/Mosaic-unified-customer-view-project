import uuid
from typing import List
from fastapi import APIRouter, Request, HTTPException
from langchain_core.messages import HumanMessage, BaseMessage
from segmented_olist_agent.api.schemas.chat import ChatRequest, ChatResponse, ChatMessage, ConversationCreateResponse, ConversationHistoryResponse
from pydantic import BaseModel

router = APIRouter()

def _lc_to_simple(msgs: List[BaseMessage]) -> List[ChatMessage]:
    simple = []
    for m in msgs:
    
        role = "assistant"
        if hasattr(m, "type"):
            if m.type == "human":
                role = "user"
            elif m.type == "ai":
                role = "assistant"
            elif m.type == "tool":
                role = "tool"
            elif m.type == "system":
                role = "system"
        simple.append(ChatMessage(role=role, content=str(m.content)))
    return simple

@router.post("/v1/conversations", response_model=ConversationCreateResponse)
def create_conversation(request: Request):
    conv_id = str(uuid.uuid4())
    request.app.state.sessions[conv_id] = []
    return ConversationCreateResponse(id=conv_id)

@router.get("/v1/conversations/{conv_id}", response_model=ConversationHistoryResponse)
def get_conversation(conv_id: str, request: Request):
    sessions = request.app.state.sessions
    if conv_id not in sessions:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationHistoryResponse(id=conv_id, messages=_lc_to_simple(sessions[conv_id]))

@router.delete("/v1/conversations/{conv_id}")
def delete_conversation(conv_id: str, request: Request):
    sessions = request.app.state.sessions
    if conv_id in sessions:
        del sessions[conv_id]
    return {"status": "deleted"}

@router.post("/v1/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    graph = request.app.state.graph
    sessions = request.app.state.sessions

    conv_id = req.conversation_id or str(uuid.uuid4())
    history = sessions.get(conv_id, [])
    history.append(HumanMessage(content=req.message))

    # Run the compiled graph 
    try:
        resp = graph({"messages": history})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {e}")

    # Append any new messages returned by the graph
    new_msgs = resp["messages"][len(history)-1:]  
    history.extend(new_msgs)
    sessions[conv_id] = history

    
    reply = str(resp["messages"][-1].content)
    return ChatResponse(
        conversation_id=conv_id,
        reply=reply,
        messages=_lc_to_simple(history)
    )
