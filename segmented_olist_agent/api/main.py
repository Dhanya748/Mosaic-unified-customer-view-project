from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from segmented_olist_agent.api.deps import lifespan
from segmented_olist_agent.api.routers import chat, sql, meta


# Initialize FastAPI app
app = FastAPI(
    title="Olist Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(meta.router, prefix="/meta", tags=["Meta"])
app.include_router(chat.router, prefix="/chat", tags=["Chat"])
app.include_router(sql.router, prefix="/sql", tags=["SQL"])

@app.get("/")
async def root():
    return {"msg": "Olist Agent API is running"}
