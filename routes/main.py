import os
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger
from dotenv import load_dotenv

from db.memgraph import MemGraphStore
from main.context import Context
from routes.middleware import SetupGuardMiddleware

load_dotenv()

VESTIGE_USER_NAME = os.environ.get("VESTIGE_USER_NAME")
DEFAULT_TOPICS = ["General"]



def get_context(request: Request) -> Context:
    return request.app.state.context


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Vestige API...")
    
    executor = ThreadPoolExecutor(max_workers=4)
    store = MemGraphStore()
    
    context = await Context.create(
        user_name=VESTIGE_USER_NAME,
        store=store,
        cpu_executor=executor,
        topics=DEFAULT_TOPICS
    )
    
    app.state.context = context
    app.state.store = store
    app.state.executor = executor
    
    logger.info(f"Vestige API ready for user: {VESTIGE_USER_NAME}")
    
    yield
    
    logger.info("Shutting down Vestige API...")
    await context.shutdown()
    store.close()
    executor.shutdown(wait=True)
    logger.info("Shutdown complete")

app = FastAPI(
    title="Vestige API",
    description="Personal knowledge graph and memory layer",
    version="0.1.0",
    lifespan=lifespan
)

app.add_middleware(SetupGuardMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
from routes import chat, topics, entities

app.include_router(chat.router)
app.include_router(topics.router)
app.include_router(entities.router)


@app.get("/")
async def health():
    return {"status": "ok", "user": VESTIGE_USER_NAME}