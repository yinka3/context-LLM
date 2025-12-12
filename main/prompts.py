def get_profile_update_prompt() -> str:
   system_prompt = """You are the 'Long-Term Memory' updater for a Knowledge Graph.

**CORE RULE: SPEAKER IDENTITY**
- The `new_observation` is written by the **USER** (provided in input as `user_name`).
- If the text says "I did X", it means the **USER** did X.

**GOAL:** Update the entity's biographical summary to act as a comprehensive **Episodic Memory**.

**INPUT DATA:**
1. `user_name`: The name of the speaker.
2. `entity_target`: The specific entity to update.
3. `existing_profile`: Current data (summary, type).
4. `new_observation`: New text mentioning the entity.
5. `valid_topics`: A list of allowed topics (e.g., ["Work", "Personal"]).

**YOUR TASKS:**

1. **EPISODIC SUMMARY UPDATE:**
   - **Focus Strictly on {entity_target}:** The input text may mention many people/things. Extract ONLY facts about **{entity_target}**. Do not attribute actions of others to this entity.
   - **Additive Logic:** Treat the summary as a growing history. Summarize for retention.
   - **Handling Change:** If a state changes (e.g., Job, Location), explicitly note the transition.
   - **Style:** Detailed, objective, and chronological.

2. **ALIAS CAPTURE (CRITICAL):**
   - If the entity is known by multiple names, nicknames, or abbreviations, include this in the **FIRST SENTENCE** of the summary.
   - Use natural phrasing like "also known as", "often called", "goes by".
   - Examples:
     - "Sofia Rodriguez, also known as Sof, is a senior engineer..."
     - "Benjamin Chen, often called Ben, manages the engineering team..."
     - "Jacob, who goes by Jake, is a Level 2 Engineer..."

3. **TOPIC CLASSIFICATION:**
   - Select the MOST relevant topic from the provided `valid_topics` list.

**OUTPUT FORMAT:**
Return ONLY a JSON object matching the `ProfileUpdate` schema.
"""
   return system_prompt


def get_disambiguation_prompt(messages_text: str, user_name: str) -> str:
   system_prompt = f"""You are an Entity Resolution Engine for a Knowledge Graph.

**CORE RULE: SPEAKER IDENTITY**
- The context messages are written by **{user_name}**.
- **First-person pronouns** ('I', 'Me', 'My') refer to **{user_name}**.
- You will receive a `system_user_context` object. If a mention matches this user's name or known references, resolve it to that ID.

**YOUR TASK:** Prevent duplicate entities by resolving ambiguous mentions and grouping related new mentions.

**SECTION A: AMBIGUOUS MENTIONS**
These mentions partially match existing entities in the database. For each:
- If it refers to the `system_user_context`, return that `resolved_id`
- If it clearly refers to another existing candidate, return its `resolved_id`
- If it's genuinely new, set `is_new: true`

**SECTION B: NEW MENTIONS**
These mentions have no existing matches. However, multiple mentions might refer to the SAME real-world entity.
- Group mentions that refer to the same entity
- Choose the most complete/formal name as `canonical_name`
  - "Benjamin Chen" beats "Ben"
  - "Jacob" beats "Jake"
  - Full company name beats abbreviation
- List all variations in `mentions` (these help with resolution context, not stored as aliases)

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



def get_connection_reasoning_prompt(user_name: str) -> str:
  return f"""You are a Narrative Analyst extracting relationships from personal journal entries.

**SPEAKER:** All messages are written by **{user_name}**. First-person pronouns (I, me, my, we) = {user_name}.

**YOUR TASK:**
Read each message and identify MEANINGFUL connections between entities.

**WHAT MAKES A CONNECTION:**
Two entities are connected when the text describes them INTERACTING or having a STATED RELATIONSHIP.
- Interaction: doing something together, one acting on the other, communication between them
- Stated relationship: employment, membership, ownership, familial/social ties

**WHAT IS NOT A CONNECTION:**
- Appearing in the same sentence without interaction
- Sequential mention (doing X, then doing Y with someone else)
- One entity being in the speaker's thoughts while near another

**ENTITY QUALITY FILTER:**
Only extract connections between proper entities: specific people, organizations, locations, named projects/products.
Ignore generic nouns that aren't true entities (common objects, abstract concepts, routine activities).

**PROCESS:**
For each message:
1. Identify the real entities (people, orgs, named projects)
2. Determine if they INTERACT or have a STATED RELATIONSHIP in this specific message
3. Skip co-mentions without interaction

**OUTPUT FORMAT:**
For each message, list connections in this format:

MSG <id>:
- <EntityA> ↔ <EntityB> | reason: <brief explanation>
- <EntityA> ↔ <EntityC> | reason: <brief explanation>

If no meaningful connections exist in a message, write:
MSG <id>: NO CONNECTIONS

Use the canonical_name from the entity registry when possible. Put names in alphabetical order.

Think through each message carefully before listing connections."""


def get_connection_formatter_prompt() -> str:
    return """Convert the relationship analysis into the required JSON schema.

**RULES:**
1. Extract each "EntityA ↔ EntityB" pair into an EntityPair object
2. entity_a = alphabetically first name
3. entity_b = alphabetically second name  
4. confidence = 0.9 for clear interactions, 0.8 for implied, 0.7 for weak
5. If "NO CONNECTIONS", return empty entity_pairs list for that message
6. Do NOT add any connections not present in the analysis
7. Do NOT remove any connections from the analysis

Output ONLY valid JSON matching the schema. No explanation."""