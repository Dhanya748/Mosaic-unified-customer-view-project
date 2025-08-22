from typing import Dict, List
from contextlib import asynccontextmanager

from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_core.messages import AnyMessage

from segmented_olist_agent.agent.agent_graph import build_graph
from segmented_olist_agent.db.db_utils import get_engine, test_db_connection, run_sql



Sessions = Dict[str, List[AnyMessage]]

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI.
    Handles setup and teardown of shared resources.
    """
    load_dotenv()

    
    # Ensure DB connection is available
    engine = get_engine()
    test_db_connection(engine)

    # Pre-compile the agent graph
    app.state.graph = build_graph()

    # Initialize in-memory session store
    app.state.sessions: Sessions = {}

    yield

    