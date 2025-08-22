import os
import re
from typing import List, Dict, Annotated, TypedDict
import operator

from dotenv import load_dotenv
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langgraph.graph import StateGraph, END

from tabulate import tabulate
from segmented_olist_agent.db.db_utils import get_engine, test_db_connection, run_sql, get_table_info



# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

# --- Validation and Helper Functions---

def _ensure_openai_key():
    """Ensure the OpenAI API key is set."""
    if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
        raise RuntimeError("OPENAI_API_KEY not set in .env file.")

def _ensure_db_url():
    """Ensure the Database URL is set."""
    if not DATABASE_URL or not DATABASE_URL.strip():
        raise RuntimeError("DATABASE_URL not set in .env file.")

ALLOW_PREFIXES = ("SELECT", "WITH")

def _apply_customers_aliases(q: str) -> str:
    """Applies domain-specific corrections to SQL queries involving the customers table."""
    low = q.lower()
    touches_customers = bool(re.search(r"\bfrom\s+customers\b|\bjoin\s+customers\b", low))
    if not touches_customers:
        return q
    q = re.sub(r"\bstate\b", "customer_state", q, flags=re.IGNORECASE)
    q = re.sub(r"\bcity\b", "customer_city", q, flags=re.IGNORECASE)
    if re.search(r"\bcount\s*\(\s*\*\s*\)", q, flags=re.IGNORECASE) or re.search(r"\bcount\s*\(\s*1\s*\)", q, flags=re.IGNORECASE):
        q = re.sub(r"\bcount\s*\(\s*\*\s*\)", "COUNT(DISTINCT customer_unique_id)", q, flags=re.IGNORECASE)
        q = re.sub(r"\bcount\s*\(\s*1\s*\)", "COUNT(DISTINCT customer_unique_id)", q, flags=re.IGNORECASE)
    return q

def clean_and_correct_sql(raw: str) -> str:
    """Cleans, validates, and applies corrections to a raw SQL query string."""
    q = re.sub(r"```sql|```", "", raw, flags=re.IGNORECASE).strip()
    q = q.strip('"').strip("'").strip()
    if q.endswith(";"):
        q = q[:-1]
    upper = q.lstrip().upper()
    if not upper.startswith(ALLOW_PREFIXES):
        raise ValueError(f"Only read-only SELECT/WITH queries are allowed. Got: {raw[:120]}...")
    if ";" in q:
        raise ValueError("Multiple statements detected. Submit a single SELECT/WITH query.")
    q = _apply_customers_aliases(q)
    print(f"\n[DEBUG] Cleaned SQL to execute:\n{q}\n")
    return q

def _format_rows(rows: List[Dict]) -> str:
    """Formats database rows into a markdown table."""
    if not rows:
        return "No rows returned from the database."
    return tabulate(rows, headers="keys", tablefmt="github")

# --- LangGraph State Definition ---

class AgentState(TypedDict):
    """Represents the state of our agent graph."""
    messages: Annotated[list[AnyMessage], operator.add]

# --- Tool Definitions ---

engine = None
schema_hint = ""

def initialize_globals():
    """Initializes global variables for the database engine and schema hint."""
    global engine, schema_hint
    _ensure_openai_key()
    _ensure_db_url()
    engine = get_engine()
    test_db_connection(engine)
    db = SQLDatabase.from_uri(
        DATABASE_URL,
        include_tables=[
            "customers", "geolocation", "sellers", "products",
            "product_category_name_translation", "orders",
            "order_items", "order_payments", "order_reviews",
        ],
        sample_rows_in_table_info=2
    )
    schema_doc = schema_doc = get_table_info()

    schema_rules = (
        "Rules:\n"
        "- Use exact column names from the schema below.\n"
        "- For customer counts, use COUNT(DISTINCT customer_unique_id).\n"
        "- Join orders->customers on orders.customer_id = customers.customer_id.\n"
        "- Use customers.customer_state and customers.customer_city (not 'state' or 'city').\n"
        "- Return only a single SELECT/WITH statement.\n"
    )
    schema_hint = f"{schema_rules}\n--- SCHEMA ---\n{schema_doc}"

def _run_sql_tool(sql: str, tool_name: str, allowed_tables: List[str]) -> str:
    """A helper function to clean, execute, and format SQL for a given tool."""
    low = sql.lower()
    if not any(t in low for t in allowed_tables):
        print(f"[WARN] Query may be out-of-scope for '{tool_name}'. Allowed tables: {allowed_tables}")
    try:
        cleaned_sql = clean_and_correct_sql(sql)
        rows = run_sql(engine, cleaned_sql)
        return _format_rows(rows)
    except Exception as e:
        return f"[ERROR] SQL failed: {e}\n\n--- Schema reference ---\n{schema_hint}"

@tool
def sales_node_sql(sql: str) -> str:
    """For sales, revenue, product, and seller analytics. Use tables: orders, order_items, order_payments, products, sellers. Input is a single SELECT/WITH statement."""
    return _run_sql_tool(sql, "sales_node_sql", ["orders", "order_items", "order_payments", "products", "sellers"])

@tool
def customers_node_sql(sql: str) -> str:
    """For customer details, reviews, and geographic questions. Use tables: customers, orders, order_reviews, geolocation. Input is a single SELECT/WITH statement."""
    return _run_sql_tool(sql, "customers_node_sql", ["customers", "orders", "order_reviews", "geolocation"])

@tool
def logistics_node_sql(sql: str) -> str:
    """For shipping, delivery times, freight, and seller locations. Use tables: orders, order_items, sellers, geolocation. Input is a single SELECT/WITH statement."""
    return _run_sql_tool(sql, "logistics_node_sql", ["orders", "order_items", "sellers", "geolocation"])

@tool
def general_sql(sql: str) -> str:
    """For general SQL queries over the Olist schema when other tools are not specific enough. Input is a single SELECT/WITH statement."""
    return _run_sql_tool(sql, "general_sql", ["customers", "geolocation", "sellers", "products", "product_category_name_translation", "orders", "order_items", "order_payments", "order_reviews"])


# --- Graph Construction ---

def build_graph():
    """Builds and compiles the LangGraph agent."""
    initialize_globals()

    tools = [sales_node_sql, customers_node_sql, logistics_node_sql, general_sql]
    
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState):
        """The primary node that calls the AI model with the current state."""
        print("---CALLING AGENT---")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def tool_node(state: AgentState):
        """Executes the tool chosen by the agent and returns the result."""
        print("---CALLING TOOLS---")
        tool_messages = []
        last_message = state["messages"][-1]
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            for tool_func in tools:
                if tool_func.name == tool_name:
                    try:
                        result = tool_func.invoke(tool_call["args"])
                        tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
                    except Exception as e:
                        tool_messages.append(ToolMessage(content=f"Error executing tool {tool_name}: {e}", tool_call_id=tool_call["id"]))
                    break
        return {"messages": tool_messages}

    def should_continue(state: AgentState):
        """Conditional logic to decide whether to continue or end the graph."""
        if not state["messages"][-1].tool_calls:
            return END
        return "tools"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", "agent")

    system_message = SystemMessage(
        content=(
            "You are an expert Postgres analyst for the Olist database.\n"
            "When you use a tool, pass ONLY a valid SQL query as the `sql` parameter. "
            "Use table/column names exactly as in the schema reference below. "
            "For counts of customers, ALWAYS use COUNT(DISTINCT customer_unique_id). "
            "After receiving a result from a tool, summarize it in a clear, natural language answer to the user.\n\n"
            + schema_hint
        )
    )
    
    graph = workflow.compile()
    
    def graph_with_system_prompt(inputs):
        """Injects the system prompt into every graph invocation."""
        return graph.invoke({"messages": [system_message, *inputs["messages"]]})

    print("[OK] LangGraph agent built successfully.")
    return graph_with_system_prompt