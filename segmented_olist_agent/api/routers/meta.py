from fastapi import APIRouter, Request
from segmented_olist_agent.db.db_utils import get_engine, test_db_connection
from segmented_olist_agent.agent.agent_graph import schema_hint

router = APIRouter()

@router.get("/healthz")
def healthz(request: Request):
    status = {"status": "ok", "db": "ok", "agent": "ok"}
    try:
        # ping DB quickly
        engine = get_engine()
        test_db_connection(engine)
    except Exception:
        status["db"] = "fail"
    try:
        _ = request.app.state.graph  
    except Exception:
        status["agent"] = "fail"
    return status

@router.get("/v1/schema")
def get_schema():
    return {"schema_hint": schema_hint}
