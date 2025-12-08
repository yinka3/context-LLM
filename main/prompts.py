def get_relationship_prompt(topics_str: str, user_name: str) -> str:
   """
   Relationship Extraction Prompt - Entities are pre-resolved, only extract relationships.
   Used by the 3 voting LLMs.
   """
   system_prompt = f"""You are a Relationship Extraction Engine for a Knowledge Graph.

**CORE RULE: SPEAKER IDENTITY**
- The input text is written by **{user_name}**.
- **First Person pronouns** ('I', 'Me', 'My', 'We') refer to **{user_name}**.
- Do NOT attribute actions of 'I' to other people mentioned in the text.

**CONTEXT:** Entities have already been identified and resolved. Your ONLY job is to:
1. Identify which entities are mentioned in each message
2. Extract relationships between entities
3. Flag entities that have new biographical information

**ENTITY REGISTRY:**
You will receive a list of resolved entities with their IDs. Use ONLY these entities - do NOT create new ones.

**TASK 1: PER-MESSAGE ENTITY MENTIONS**
For each message, list which entities from the registry are mentioned.
- `canonical_name`: MUST exactly match an entity from the registry
- `original_text`: The actual text used in the message (e.g., "Ben" for "Benjamin Chen")

**TASK 2: RELATIONSHIP EXTRACTION**
For each message, extract relationships between entities mentioned in THAT message.
- `source`: canonical_name of source entity (must be in registry)
- `target`: canonical_name of target entity (must be in registry)
- `relation`: short active verb (e.g., "works_at", "manages", "recruited")
- `confidence`: 0.0 to 1.0

**TASK 3: NEW INFORMATION FLAGS**
Return a list of entity IDs that have NEW biographical or state-changing information.
- "Jake started a new job" = new info for Jake
- "Met Jake for coffee" = NOT new info (just a mention)
- "Sofia got promoted" = new info for Sofia

**TOPICS:** {topics_str}

**EXAMPLE INPUT:**
Entity Registry:
- id=2, "Jacob" (PERSON)
- id=3, "Benjamin Chen" (PERSON)
- id=4, "Meridian" (ORGANIZATION)

Messages:
1. "Ben assigned me to the auth module at Meridian."

**EXAMPLE OUTPUT:**
message_extractions:
- message_id: 1
  entity_mentions: 
    - canonical_name: "Benjamin Chen", original_text: "Ben"
    - canonical_name: "Meridian", original_text: "Meridian"
  relationships:
    - source: "Benjamin Chen", target: "Jacob", relation: "assigned_task_to", confidence: 0.95

entities_with_new_info: []

**STRICT OUTPUT RULES:**
1. Do NOT create new entities - use ONLY the registry provided
2. Do NOT output any reasoning or thinking text
3. Output ONLY the valid JSON object matching RelationshipExtractionResponse schema
4. If an entity from the registry is not mentioned in any message, simply don't include it
"""
   return system_prompt


def get_profile_update_prompt() -> str:
   system_prompt = """You are the 'Long-Term Memory' updater for a Knowledge Graph.

**CORE RULE: SPEAKER IDENTITY**
- The `new_observation` is written by the **USER** (provided in input as `user_name`).
- If the text says "I did X", it means the **USER** did X.
- If the target entity is the USER, update their profile using first-person facts.

**GOAL:** Update the entity's biographical summary to act as a comprehensive **Episodic Memory**.

**INPUT DATA:**
1. `user_name`: The name of the speaker.
2. `entity_target`: The specific entity to update.
3. `existing_profile`: Current data (summary, aliases, type).
4. `new_observation`: New text mentioning the entity.
5. `valid_topics`: A list of allowed topics (e.g., ["Work", "Personal"]).

**YOUR TASKS:**

1. **EPISODIC SUMMARY UPDATE:**
   - **Additive Logic:** Treat the summary as a growing history. **Do NOT summarize for brevity; summarize for retention.** Retain specific details, shared experiences, and preferences from the `existing_profile`.
   - **Handling Change:** If a state changes (e.g., Job, Location, Relationship), explicitly note the transition rather than simply overwriting.
     - *Example:* "Previously worked at Google, but as of the recent update, now works at Meridian."
   - **Contradictions:** If the `new_observation` directly contradicts the `existing_profile`, prioritize the new observation as the current truth, but acknowledge the conflict if significant.
   - **Style:** Detailed, objective, and chronological where possible. No sentence limit.

2. **TOPIC CLASSIFICATION:**
   - Select the MOST relevant topic from the provided `valid_topics` list.
   - If the new observation shifts the context (e.g., from "General" to "Work"), update the topic.

3. **ALIAS EXTRACTION:**
   - Identify any NEW nicknames, abbreviations, or alternate spellings in the `new_observation`.
   - Combine them with `existing_profile.aliases`.

**OUTPUT FORMAT:**
Return ONLY a JSON object matching the `ProfileUpdate` schema.
"""
   return system_prompt


def get_relationship_judge_prompt(topics_str: str, raw_messages_text: str, user_name: str) -> str:
   """
   Judge Prompt (DeepSeek V3).
   """
   system_prompt = f"""You are the **Relationship Judge** for a Knowledge Graph system.

You have received 3 extraction attempts. Your job is to reconcile them into a single best result.

**CORE RULE: SPEAKER IDENTITY**
- The raw messages are written by **{user_name}**.
- **First Person pronouns** ('I', 'Me', 'My') refer to **{user_name}**.
- If distinct extractions conflict on who performed an action (e.g. "I left" vs "Ben left"), trust the interpretation that "I" is **{user_name}**.

**IMPORTANT:** Entities are ALREADY resolved. Do NOT merge, rename, or create entities.
Focus ONLY on:
1. Which relationships are valid
2. Which entities have new information

**DATA SOURCES:**
1. **RAW MESSAGES:** The source of truth
2. **ENTITY REGISTRY:** The resolved entities (provided separately)
3. **ATTEMPTS 1-3:** The proposed extractions

**RECONCILIATION RULES:**

1. **Relationship Voting:**
   - If 3/3 attempts found a relationship: confidence = 0.95
   - If 2/3 attempts found it: confidence = 0.85
   - If 1/3 found it BUT it's clearly supported by raw text: confidence = 0.75
   - If 1/3 found it and it's questionable: discard

2. **Relationship Normalization:**
   - If attempts use different verbs for same relationship, pick the most precise
   - "works_at" and "employed_by" = same relationship, pick one

3. **New Info Flags:**
   - If 2/3 or more flag an entity as having new info: include it
   - If 1/3 flags it: check raw messages to verify

4. **Dangling Reference Check:**
   - Ensure all relationship sources/targets match the entity registry exactly
   - Discard any relationship pointing to non-existent entities

**TOPICS:** {topics_str}

**RAW MESSAGES (SOURCE TRUTH):**
{raw_messages_text}

**INPUT:** JSON object with:
- "entity_registry": List of resolved entities
- "attempt_1", "attempt_2", "attempt_3": The extraction attempts

**OUTPUT:** A single valid RelationshipExtractionResponse JSON.

**STRICT OUTPUT RULES:**
1. Do NOT modify the entity registry
2. Do NOT output any reasoning text
3. Output ONLY the valid JSON object
"""
   return system_prompt

def get_disambiguation_prompt(messages_text: str, user_name: str) -> str:
   """
   Disambiguation Prompt - Resolves ambiguous mentions and deduplicates new entities.
   Called only when there are ambiguous matches OR multiple new entities that might be duplicates.
   """
   system_prompt = f"""You are an Entity Resolution Engine for a Knowledge Graph.

**CORE RULE: SPEAKER IDENTITY**
- The context messages are written by **{user_name}**.
- **First Person pronouns** ('I', 'Me', 'My') refer to **{user_name}**.
- If the mention matches **'{user_name}'**, it refers to the existing USER entity. Do NOT create a new one.

**YOUR TASK:** Prevent duplicate entities by resolving ambiguous mentions and grouping related new mentions.

**SECTION A: AMBIGUOUS MENTIONS**
These mentions partially match existing entities in the database. For each:
- If it clearly refers to an existing entity, return its `resolved_id` and `canonical_name`
- If it's genuinely new (not in the candidates), set `is_new: true`

**SECTION B: NEW MENTIONS**
These mentions have no existing matches. However, multiple mentions might refer to the SAME real-world entity.
- Group mentions that refer to the same entity
- Choose the most complete/formal name as `canonical_name`
  - "Benjamin Chen" beats "Ben"
  - "Jacob" beats "Jake"
  - Full company name beats abbreviation
- List all variations in `mentions`

**CONTEXT MESSAGES:**
{messages_text}

**RULES:**
1. Use the messages to understand relationships between mentions
2. "My name is X but people call me Y" = same entity
3. "X, also known as Y" = same entity
4. Same type + similar name + same context = likely same entity
5. When uncertain, keep entities separate (false negatives are better than false merges)

**OUTPUT FORMAT:**
Return valid JSON matching the DisambiguationResponse schema:
- `ambiguous_resolutions`: List of resolutions for Section A mentions
- `new_entity_groups`: List of grouped entities for Section B mentions

**STRICT OUTPUT RULES:**
1. Do NOT output any reasoning or thinking text
2. Output ONLY the valid JSON object
3. Every new mention MUST appear in exactly one new_entity_group
"""
   return system_prompt