# Vestige

A self-hosted knowledge graph and memory layer for conversational AI.

Inspired by [Zep’s temporal knowledge graph architecture](https://github.com/getzep/zep), Vestige extracts entities and relationships from unstructured text, maintains evolving profiles, and provides grounded context for LLM conversations. Emphasizes transparency, bounded AI behavior, and cost-effective token usage.

## Features

**Entity Extraction & Disambiguation**  
Identifies people, places, organizations, and concepts from conversational text. Handles the chaos of real messaging: typos, nicknames, inconsistent casing, and resolves them to canonical entities.

**Relationship Tracking**  
Builds a graph of who knows who, what connects to what, with message-level evidence. Relationships have weights and timestamps, so you know what’s strong and what’s stale.

**Topic-Based Access Control**  
Toggle topics active/inactive to restrict what the agent can see. Inactive topics are filtered out at the database level, not just hidden in the UI. Mark topics as “hot” for priority retrieval.

## SS Agents

Sleepy/Simple agents. They wake up during idle periods to clean house, then go back to sleep.

Current SS agents:

- **Profile Refinement** — Entity summaries evolve as new information comes in. Learns that “Marc” and “Marcus” are the same person, and that he switched jobs three months ago.
- **Merge Detection** — Catches duplicates that slip through initial disambiguation. Uses embedding similarity + cross-encoder verification to propose merges with confidence scores.
- **DLQ Replay** — Retries failed batches when transient errors (network blips, timeouts) were the cause. Parks fatal errors for inspection instead of infinite loops.

The scheduler is designed to be extensible. Add your own SS agents for custom background tasks.

## Architecture

Vestige separates **write** (deterministic extraction) from **read** (agentic retrieval).

### Write Path

In *One Piece*, Dr. Vegapunk is the world’s greatest scientist, so brilliant that his brain grew too large for his body. His solution? Split his consciousness into six satellites, each handling a specialized aspect of his genius: logic, evil, good, desire, violence, and wisdom.

Vestige borrows this idea. Rather than throwing one monolithic prompt at extraction and hoping for the best, the write path splits cognitive labor across specialized prompts:

|Satellite  |Role                             |
|-----------|---------------------------------|
|VEGAPUNK-01|Named entity recognition         |
|VEGAPUNK-02|Disambiguation reasoning         |
|VEGAPUNK-03|Disambiguation formatting        |
|VEGAPUNK-04|Connection/relationship reasoning|
|VEGAPUNK-05|Connection formatting            |
|VEGAPUNK-06|Profile refinement               |
|VEGAPUNK-07|Summary merging                  |

Each prompt does one thing well. Reasoning and formatting are deliberately separated: let the LLM think freely, then constrain the output. This keeps accuracy high and structured output reliable.

### Read Path (In Progress)

STELLA serves as the main conversational agent, using a bounded 5-state machine for retrieval rather than free-form ReAct patterns. Tools query the graph; the LLM synthesizes responses with grounded context.