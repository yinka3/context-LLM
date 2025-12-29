from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from routes.main import get_context
from routes.models import TopicList, TopicUpdate
from main.context import Context

router = APIRouter(prefix="/topics", tags=["topics"])

MAX_ACTIVE = 10
MAX_HOT = 5


@router.get("", response_model=TopicList)
async def list_topics(context: Context = Depends(get_context)):
    topics = context.store.get_topics_by_status()
    return TopicList(
        active=topics.get("active", []),
        hot=topics.get("hot", []),
        inactive=topics.get("inactive", [])
    )


@router.patch("/{name}")
async def update_topic(name: str, update: TopicUpdate, context: Context = Depends(get_context)):
    topics = context.store.get_topics_by_status()
    
    all_topics = topics["active"] + topics["hot"] + topics["inactive"]
    if name not in all_topics:
        raise HTTPException(status_code=404, detail=f"Topic '{name}' not found")
    
    current_active = len(topics["active"])
    current_hot = len(topics["hot"])
    
    is_currently_active = name in topics["active"]
    is_currently_hot = name in topics["hot"]
    
    if update.status == "hot":
        if not is_currently_hot and current_hot >= MAX_HOT:
            raise HTTPException(status_code=400, detail=f"Maximum {MAX_HOT} hot topics allowed")
    
    if update.status == "active":
        if not is_currently_active and not is_currently_hot and current_active >= MAX_ACTIVE:
            raise HTTPException(status_code=400, detail=f"Maximum {MAX_ACTIVE} active topics allowed")
    
    context.store.set_topic_status(name, update.status)
    
    if update.status == "inactive" and name in context.active_topics:
        context.active_topics.remove(name)
    elif update.status in ("active", "hot") and name not in context.active_topics:
        context.active_topics.append(name)
    
    logger.info(f"Topic '{name}' status changed to '{update.status}'")
    
    return {"name": name, "status": update.status}


@router.post("")
async def create_topic(name: str, context: Context = Depends(get_context)):
    topics = context.store.get_topics_by_status()
    all_topics = topics["active"] + topics["hot"] + topics["inactive"]
    
    if name in all_topics:
        raise HTTPException(status_code=400, detail=f"Topic '{name}' already exists")
    
    if len(topics["active"]) >= MAX_ACTIVE:
        raise HTTPException(status_code=400, detail=f"Maximum {MAX_ACTIVE} active topics allowed")
    
    context.store.set_topic_status(name, "active")
    context.active_topics.append(name)
    
    logger.info(f"Topic '{name}' created")
    
    return {"name": name, "status": "active"}