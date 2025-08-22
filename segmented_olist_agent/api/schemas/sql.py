from typing import Any, Dict, List
from pydantic import BaseModel

class SqlPreviewRequest(BaseModel):
    sql: str

class SqlPreviewResponse(BaseModel):
    rows: List[Dict[str, Any]]
    markdown: str
