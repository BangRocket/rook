# Cognitive Science Concepts

Understanding the science behind Rook's memory system.

## Overview

Rook applies research from cognitive psychology and neuroscience to create AI memory that behaves more like human memory. Unlike simple vector databases that treat all information equally, Rook implements:

1. **Forgetting curves** - Memories decay naturally over time
2. **Strengthening through use** - Accessed memories become more stable
3. **Prediction error gating** - Surprising information is prioritized
4. **Spreading activation** - Related memories activate each other
5. **Consolidation** - Important memories are stabilized

## FSRS-6 Memory Dynamics

### What is FSRS?

FSRS (Free Spaced Repetition Scheduler) is an algorithm developed for optimal learning, based on the forgetting curve research by Ebbinghaus and subsequent memory research. Rook uses FSRS-6, the latest version.

### Key Concepts

**Stability (S):** How long a memory will persist before decaying. Higher stability means slower forgetting.

**Difficulty (D):** How hard a memory is to retain. Range 0-1, where higher values mean more challenging.

**Retrievability (R):** The probability of successfully retrieving a memory at any given time. Calculated as:

```
R = (1 + t/S)^(-decay)
```

Where:
- `t` = time since last access
- `S` = stability
- `decay` = power-law decay rate (default: 0.9)

### How It Works in Rook

```
New memory created → Initial stability based on content
        ↓
Memory accessed → Stability increases based on grade
        ↓
Time passes → Retrievability decreases (forgetting)
        ↓
Retrievability < threshold → Memory archived (intelligent forgetting)
```

### Grades

When a memory is accessed or reviewed:

| Grade | Meaning | Effect on Stability |
|-------|---------|---------------------|
| Again | Wrong/unhelpful | Decrease significantly |
| Hard | Partially correct | Slight decrease |
| Good | Correct/helpful | Increase |
| Easy | Effortlessly correct | Increase significantly |

### Dual Strength Model

Rook tracks two types of strength:

- **Retrieval Strength:** How easily the memory can be accessed now
- **Storage Strength:** How well the memory is encoded long-term

This mirrors the distinction in human memory between accessibility and availability.

## Prediction Error Gating

### The Science

In neuroscience, prediction error is the difference between what we expect and what we observe. High prediction error signals novelty and triggers learning; low prediction error means the information is redundant.

### Implementation in Rook

When new information arrives, Rook evaluates it against existing memories:

```
New input → Compare to existing memories → Calculate prediction error
                                                    ↓
                                    ┌───────────────┼───────────────┐
                                    ↓               ↓               ↓
                               High error      Medium error     Low error
                                    ↓               ↓               ↓
                                 CREATE          UPDATE           SKIP
```

### Detection Layers

Rook uses multiple layers to detect conflicts and novelty:

1. **Embedding Similarity:** Vector distance to existing memories
2. **Keyword Negation:** Detect explicit contradictions ("is" vs "is not")
3. **Temporal Conflict:** Date/time inconsistencies
4. **LLM Semantic Check:** Deep semantic analysis (configurable)

### Outcomes

| Result | When | Example |
|--------|------|---------|
| **Skip** | Information already known | "Alice is an engineer" when already stored |
| **Create** | Genuinely new information | First mention of a topic |
| **Update** | Additional detail on existing | "Alice is a senior engineer" |
| **Supersede** | Contradicts existing | "Alice now works at BigCorp" |

## Spreading Activation (ACT-R)

### The Science

Spreading activation is a theory from cognitive psychology (ACT-R architecture) that explains how activating one concept in memory spreads activation to related concepts.

### How It Works

```
Query: "coffee"
        ↓
Direct match: "User likes coffee" (high activation)
        ↓
Spread to related:
  → "Morning routine" (medium activation)
  → "Favorite cafe" (medium activation)
  → "Work habits" (low activation)
```

### Implementation

Rook builds a knowledge graph from extracted entities and relationships:

```
[Alice] --works_at--> [Acme Corp]
   |                      |
   └--knows--> [Bob] -----┘
```

When searching, activation spreads through edges:

1. Query matches nodes directly → Base activation
2. Activation spreads to neighbors → Reduced by decay factor
3. Continues until threshold or max depth
4. Final activation combined with vector similarity

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spreading_decay` | 0.5 | Activation reduction per hop |
| `spreading_threshold` | 0.1 | Minimum activation to continue |
| `spreading_max_depth` | 3 | Maximum hops from source |

## Synaptic Tagging & Consolidation

### The Science

Synaptic tagging and capture (STC) explains how memories become permanent:

1. Learning creates a temporary "tag" at synapses
2. Novel/emotional events trigger "plasticity-related proteins" (PRPs)
3. Tagged synapses capture PRPs and become stabilized
4. Untagged or weak memories fade

### Implementation in Rook

**Synaptic Tags:**
- Created when memories are added
- Decay exponentially over time (tau = 60 minutes default)
- Stronger initial tags for surprising/emotional content

**Behavioral Tagging:**
- Novel events boost temporally adjacent memories
- Window: 30 minutes before, 2 hours after the novel event
- Mimics "flashbulb memory" effects

**Consolidation Process:**
```
Tagged memories → Consolidation check (periodic)
        ↓
Tag strength > threshold → Boost storage_strength (+15%)
        ↓
Tag strength < threshold → Memory remains volatile
```

### Phases

| Phase | Time Window | Effect |
|-------|-------------|--------|
| Immediate | 0-6 hours | Tag active, memory volatile |
| Early | 6-24 hours | Initial consolidation |
| Late | 24-72 hours | Deep consolidation |

## Retrieval Modes

Rook offers multiple retrieval modes that combine these cognitive mechanisms:

### Quick Mode
- Vector similarity only
- Fastest, lowest resource usage
- Best for: Simple lookups, high-volume queries

### Standard Mode
- Vector + spreading activation + keyword search
- Balanced performance and quality
- Best for: General use

### Precise Mode
- All methods + reranking
- Highest quality, more resources
- Best for: Critical queries, complex questions

### Cognitive Mode
- Spreading activation + FSRS weighting
- Prioritizes strong, interconnected memories
- Best for: Mimicking human recall patterns

## Classification System

### Categories

Rook classifies memories into cognitive categories:

| Category | Description | Examples |
|----------|-------------|----------|
| Personal | Identity information | Name, age, location |
| Preference | Likes and dislikes | Favorite foods, hobbies |
| Fact | Objective knowledge | "Python is a programming language" |
| Belief | Subjective views | Opinions, values |
| Skill | Abilities | "Knows Rust programming" |
| Experience | Past events | "Visited Paris in 2020" |
| Social | Relationships | "Works with Bob" |
| Work | Professional info | Job, projects |
| Health | Health-related | Allergies, conditions |
| Other | Uncategorized | Misc information |

### Key Memories

Critical information can be marked as "key memories":
- Always included in search results
- Protected from archival/forgetting
- Capped to prevent overwhelming results (default: 15)

## Practical Implications

### For Memory Quality
- Frequent access strengthens memories
- Contradictions are detected and resolved
- Unimportant information naturally fades

### For Search Quality
- Related memories surface together
- Recent and frequently-used memories rank higher
- Context from the knowledge graph enriches results

### For System Design
- Not a replacement for databases (memories can be forgotten)
- Best for: User preferences, learned context, conversation history
- Not for: Critical data that must be preserved exactly

## Further Reading

- [FSRS Algorithm](https://github.com/open-spaced-repetition/fsrs-rs)
- [ACT-R Cognitive Architecture](http://act-r.psy.cmu.edu/)
- [Synaptic Tagging and Capture](https://en.wikipedia.org/wiki/Synaptic_tagging)
- [Prediction Error in Learning](https://en.wikipedia.org/wiki/Prediction_error)
