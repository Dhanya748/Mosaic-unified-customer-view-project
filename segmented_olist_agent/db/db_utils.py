import os
import sys
import re
from typing import List, Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


dotenv_path = Path(__file__).resolve().parent.parent / ".env"
if not dotenv_path.exists():
    print(f"[WARN] .env file not found at {dotenv_path}")
load_dotenv(dotenv_path)

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def _fail(msg: str) -> None:
    print(f"\n[SETUP ERROR] {msg}\n", file=sys.stderr)
    raise SystemExit(1)

def get_engine() -> Engine:
    if not DATABASE_URL:
        _fail("DATABASE_URL is not set. Put it in .env (see template).")
    if re.search(r"//[^/]+@[^/]+/", DATABASE_URL) and "%40" not in DATABASE_URL:
        print("[WARN] Your DB URL contains '@'. If it's part of the password, URL-encode it as %40.")
    try:
        engine = create_engine(
            DATABASE_URL,
            pool_pre_ping=True, 
            future=True,
        )
        return engine
    except Exception as e:
        _fail(f"Failed to create SQLAlchemy engine from DATABASE_URL.\n{e}")

def test_db_connection(engine: Engine) -> None:
    try:
        with engine.connect() as conn:
            one = conn.execute(text("SELECT 1")).scalar()
            print(f"[OK] PostgreSQL connection works. Test query returned: {one}")
    except Exception as e:
        _fail(f"Database connection failed. Check DATABASE_URL, server, creds.\n{e}")

def run_sql(engine: Engine, sql: str) -> List[Dict[str, Any]]:
    """Execute a SELECT query and return list of dict rows."""
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        rows = [dict(r._mapping) for r in result]
    return rows

def get_table_info() -> str:
    """
    Return a schema description of all tables in the connected database.
    This is used by the agent to understand the DB structure.
    """
    engine = get_engine()
    sql = """
    SELECT table_name, column_name, data_type
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, ordinal_position;
    """
    rows = run_sql(engine, sql)

    schema_doc = {}
    for r in rows:
        t = r["table_name"]
        if t not in schema_doc:
            schema_doc[t] = []
        schema_doc[t].append(f"{r['column_name']} ({r['data_type']})")

    # Pretty string
    return "\n".join(
        f"{t}: {', '.join(cols)}" for t, cols in schema_doc.items()
    )
