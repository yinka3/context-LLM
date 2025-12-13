from datetime import datetime, timezone
from loguru import logger
import threading
from typing import Dict, List
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rapidfuzz import process as fuzzy_process, fuzz
from graph.memgraph import MemGraphStore



class EntityResolver:

    def __init__(self, embedding_model='dunzhang/stella_en_1.5B_v5'):
        self.embedding_model = SentenceTransformer(embedding_model, trust_remote_code=True, device='cpu')
        self.cross_encoder = CrossEncoder('BAAI/bge-reranker-v2-m3', device='cpu')
        self.embedding_dim = 1024
        self.faiss_index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.embedding_dim)
        self.index_id_map: faiss.IndexIDMap2 = faiss.IndexIDMap2(self.faiss_index)
        self.ref = {"me", "i", "myself", "user"}
        
        self.fuzzy_choices: Dict[str, int] = {}
        self.entity_profiles: Dict[int, Dict] = {}

        self._lock = threading.RLock()
    
    def _init_from_db(self):
        logger.info("Hydrating Resolver from Memgraph...")

        temp_profiles = {}
        temp_fuzzy = {}
        vectors = []
        ids = []
        self.ref = {"me", "i", "myself", "user"}

        try:
            store = MemGraphStore()
            
            query = """
            MATCH (n:Entity)
            RETURN n.id, n.canonical_name, n.summary, n.type, n.embedding, n.last_profiled_msg_id
            """
            
            with store.driver.session() as session:
                results = session.run(query)

                for r in results:
                    e_id = r["n.id"]
                    raw_last_msg = r.get("n.last_profiled_msg_id")
                    last_msg_id = raw_last_msg if raw_last_msg is not None else 0
                    canonical_name = r["n.canonical_name"]
                    summary = r["n.summary"] or ""
                    
                    temp_profiles[e_id] = {
                        "canonical_name": canonical_name,
                        "summary": summary or "",
                        "type": r["n.type"],
                        "last_profiled_msg_id": last_msg_id
                    }
                    
                    temp_fuzzy[canonical_name] = e_id
                    summary = summary or ""
                    resolution_text = f"{canonical_name}. {summary[:200]}"
                    embedding = self.embedding_model.encode([resolution_text])[0]
                    vectors.append(embedding)
                    ids.append(e_id)
                
            with self._lock:
                self.entity_profiles = temp_profiles
                self.fuzzy_choices = temp_fuzzy

                self.index_id_map.reset()

                if vectors:
                    logger.info(f"Loading {len(vectors)} vectors into FAISS.")
                    vec_np = np.array(vectors, dtype=np.float32)
                    ids_np = np.array(ids, dtype=np.int64)
                    
                    faiss.normalize_L2(vec_np)
                    self.index_id_map.add_with_ids(vec_np, ids_np)

            return True
        except Exception as e:
            logger.error(f"Failed to hydrate from DB: {e}")
            return False

    def add_entity(self, entity_id: int, profile: Dict) -> List[float]:

        canonical_name = profile.get("canonical_name", "")
        summary = profile.get("summary", "") or ""

        resolution_text = f"{canonical_name}. {summary[:200]}"
        embedding_np = self.embedding_model.encode([resolution_text])[0]
        faiss.normalize_L2(embedding_np.reshape(1, -1))


        with self._lock:
            logger.info(f"Adding entity {entity_id} to resolver indexes.")

            profile.setdefault("topic", "General")
            profile.setdefault("first_seen", datetime.now(timezone.utc).isoformat())
            profile["last_seen"] = datetime.now(timezone.utc).isoformat()
            
            self.index_id_map.add_with_ids(
                np.array([embedding_np]), 
                np.array([entity_id], dtype=np.int64)
            )

            self.entity_profiles[entity_id] = profile
            self.fuzzy_choices[canonical_name] = entity_id
        
        return embedding_np.tolist()
    
    
    def resolve(self, text: str, context: str, fuzzy_cutoff: int = 80):

        with self._lock:
            has_data = bool(self.fuzzy_choices) or self.index_id_map.ntotal > 0
        
        if not has_data:
            return {"resolved": None, "ambiguous": [], "new": True, "mention": text}
        
        set_text = set(word.lower().strip(".,!'?") for word in text.split())
        if any(word in self.ref for word in set_text):
            for _, _id in self.fuzzy_choices.items():
                profile = self.entity_profiles.get(_id, {})
                if profile.get("type") == "PERSON" and _id == 1:
                    return {
                        "resolved": {"id": _id, "profile": profile},
                        "ambiguous": [],
                        "new": False,
                        "mention": text
                    }
        
        query_text = f"{text} mentioned in context of: {context}"
        query_embedding = self.embedding_model.encode([query_text])
        faiss.normalize_L2(query_embedding)
        
        candidates_map: Dict[int, Dict] = {}
        with self._lock:
            if self.fuzzy_choices:
                fuzzy_matches = fuzzy_process.extract(query=text, choices=self.fuzzy_choices.keys(),
                                                    scorer=fuzz.WRatio, score_cutoff=fuzzy_cutoff, limit=10)
                for match_name, score, _ in fuzzy_matches:
                    entity_id = self.fuzzy_choices[match_name]
                    norm_score = score / 100.0

                    candidates_map[entity_id] = {
                        "id": entity_id,
                        "match_detail" : {
                            "source": "fuzzy",
                            "score": score,
                            "norm_score": norm_score,
                            "matched_aliases": match_name
                        },
                        "profile": self.entity_profiles.get(entity_id, {})
                    }
            
            if self.index_id_map.ntotal > 0:
                scores, ann = self.index_id_map.search(query_embedding, k=min(10, self.index_id_map.ntotal))

                for index_id, score in zip(ann[0], scores[0]):
                    logger.debug(f"FAISS result: query='{text}' -> id={index_id}, score={score:.3f}")
                    if index_id != -1 and score >= 0.5:
                        entity_id = int(index_id)
                        norm_score = float(score)

                        existing = candidates_map.get(entity_id)

                        if existing:
                            existing["match_detail"]["source"] = "hybrid"
                            existing["match_detail"]["vector_store"] = float(score)

                            if norm_score > existing["match_detail"]["norm_score"]:
                                existing["match_detail"]["norm_score"] = norm_score
                        else:
                            candidates_map[entity_id] = {
                                "id": entity_id,
                                "match_detail": {
                                    "source": "faiss",
                                    "score": float(score),
                                    "norm_score": norm_score
                                },
                                "profile": self.entity_profiles[entity_id]
                            }
        if not candidates_map:
            return {"resolved": None, "ambiguous": [], "new": True, "mention": text}
        
        sorted_candidates = sorted(
        candidates_map.values(),
        key=lambda x: x["match_detail"]["norm_score"],
        reverse=True)

        best_score = sorted_candidates[0]["match_detail"]["norm_score"]
    
        if best_score >= 0.70:
            return {"resolved": sorted_candidates[0], "ambiguous": [], "new": False, "mention": text}
        elif best_score >= 0.50:
            viable = [c for c in sorted_candidates if c["match_detail"]["norm_score"] >= 0.50]
            return {"resolved": None, "ambiguous": viable, "new": False, "mention": text}
        else:
            return {"resolved": None, "ambiguous": [], "new": True, "mention": text}


    def detect_merge_candidates(self) -> list:
        """Detect potential entity merges based on summary similarity."""
        
        if not self._init_from_db():
            logger.error("Failed to refresh from database, aborting merge detection")
            return []
    
        logger.info(f"Merge detection started, {len(self.entity_profiles)} entities to scan")
        
        candidates = []
        seen_pairs = set()
        
        for entity_id, profile in self.entity_profiles.items():
            summary = profile.get("summary", "")
            entity_type = profile.get("type")
            canonical_name = profile.get("canonical_name", "Unknown")
            if not summary:
                continue
            
            import re
            match = re.match(r'^(.+?[.!?])(?:\s+[A-Z]|$)', summary)
            first_sentence = match.group(1).strip() if match else summary[:200].strip()
            
            s2s_prompt = "Instruct: Retrieve semantically similar text.\nQuery: "
            embedding = self.embedding_model.encode([s2s_prompt + first_sentence])
            faiss.normalize_L2(embedding)
            
            with self._lock:
                if self.index_id_map.ntotal == 0:
                    continue
                
                scores, indices = self.index_id_map.search(embedding, k=5)
            
            for match_id, faiss_score in zip(indices[0], scores[0]):
                match_id = int(match_id)
                
                if match_id == -1 or match_id == entity_id:
                    continue
                
                match_profile = self.entity_profiles.get(match_id)
                if not match_profile:
                    continue

                match_type = match_profile.get("type")
                match_name = match_profile.get("canonical_name", "Unknown")
        
                if entity_type != match_type:
                    logger.debug(f"Type mismatch: {canonical_name} ({entity_type}) vs {match_name} ({match_type}), skipping")
                    continue

                if canonical_name == match_name:
                    pair = tuple(sorted([entity_id, match_id]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    
                    logger.info(f"Merge verification: ({entity_id}, {match_id}) {canonical_name} <- {match_name} | "
                                f"Exact name match Decision=APPROVED")
                    continue
                
                if faiss_score < 0.50:
                    continue

                primary_text = f"{canonical_name}. {profile.get('summary', '')[:150]}"
                secondary_text = f"{match_name}. {match_profile.get('summary', '')[:150]}"
                cross_score = float(self.cross_encoder.predict([[primary_text, secondary_text]])[0])
                
                pair = tuple(sorted([entity_id, match_id]))
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                
                decision = "APPROVED" if cross_score >= 0.65 else "REJECTED"
                logger.info(
                    f"Merge verification: ({entity_id}, {match_id}) {canonical_name} <- {match_name} | "
                    f"FAISS={faiss_score:.3f} CrossEncoder={cross_score:.3f} Decision={decision}"
                )
                
                if cross_score >= 0.65:
                    primary_id, secondary_id = pair
                    candidates.append({
                        "primary_id": primary_id,
                        "secondary_id": secondary_id,
                        "primary_name": self.entity_profiles[primary_id].get("canonical_name", "Unknown"),
                        "secondary_name": self.entity_profiles[secondary_id].get("canonical_name", "Unknown"),
                        "faiss_score": float(faiss_score),
                        "cross_score": cross_score
                    })
        
        logger.info(f"Merge detection complete: {len(candidates)} candidates found")
        
        return candidates