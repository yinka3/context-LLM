from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from routes.main import get_context
from routes.models import EntitySummary, EntityProfile
from main.context import Context

router = APIRouter(prefix="/entities", tags=["entities"])


@router.get("", response_model=list[EntitySummary])
async def list_entities(
    topic: Optional[str] = Query(None, description="Filter by topic"),
    limit: int = Query(50, ge=1, le=200),
    context: Context = Depends(get_context)
):
    entities = context.store.get_entities_list(topic=topic, limit=limit)
    
    results = []
    for ent in entities:
        summary = ent.get("summary") or ""
        snippet = summary[:100] + "..." if len(summary) > 100 else summary
        
        results.append(EntitySummary(
            name=ent["canonical_name"],
            type=ent.get("type", "unknown"),
            summary_snippet=snippet or None,
            topic=ent.get("topic")
        ))
    
    return results


@router.get("/{name}", response_model=EntityProfile)
async def get_entity(name: str, context: Context = Depends(get_context)):
    profile = context.store.get_entity_profile(name)
    
    if not profile:
        raise HTTPException(status_code=404, detail=f"Entity '{name}' not found")
    
    return EntityProfile(
        id=profile["id"],
        canonical_name=profile["canonical_name"],
        type=profile.get("type"),
        aliases=profile.get("aliases") or [],
        summary=profile.get("summary"),
        topic=profile.get("topic"),
        last_mentioned=profile.get("last_mentioned"),
        last_updated=profile.get("last_updated")
    )