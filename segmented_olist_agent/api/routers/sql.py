from fastapi import APIRouter, HTTPException
from segmented_olist_agent.agent.agent_graph import clean_and_correct_sql, _format_rows, engine
from segmented_olist_agent.db.db_utils import run_sql
from segmented_olist_agent.api.schemas.sql import SqlPreviewRequest, SqlPreviewResponse

router = APIRouter()

@router.post("/v1/sql/preview", response_model=SqlPreviewResponse)
def sql_preview(req: SqlPreviewRequest):
    try:
        cleaned = clean_and_correct_sql(req.sql)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        rows = run_sql(engine, cleaned)
        md = _format_rows(rows)
        return SqlPreviewResponse(rows=rows, markdown=md)
    except Exception as e:
        
        raise HTTPException(status_code=422, detail=f"SQL failed: {e}")
