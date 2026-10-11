---
name: restaurant-uniqueness-explorer
description: >-
  Orchestrates parallel per-restaurant research subagents (1 agent per restaurant)
  to evaluate shortlisted restaurants for regional uniqueness, culinary
  distinctiveness, and "What Would I Order?" dish curation. Use when the user
  wants to compare shortlisted restaurants, decide if a restaurant is worth
  visiting, or discover the unique, non-standard regional dishes to order at
  specific restaurants (either passed in chat or pulled from the BigQuery
  fsa_master table).
---

# Restaurant Uniqueness & "What Would I Order?" Explorer

Dispatches **1 dedicated research subagent per restaurant** in parallel to evaluate a shortlist of restaurants. Each subagent performs deep English + native-language menu and culinary-science research under the hood, then outputs an **ultra-concise, telegraph-style report** weighted **30% 6-Pillar Background Check** and **70% "What Would I Order?" Blueprint**:
1. **6-Pillar Check**: 6 one-line bullets verifying the *Healthy Host & Explorer* pillars + a 3-tier Worthiness Verdict (~30% of output).
2. **What Would I Order?**: 5 one-line unique dish recommendations (filtered of common UK baseline dishes) + a 4-line *Feast for 2 (F&A)* order + a 1-line *Skip* list (~70% of output).

---

## Workflow

### Step 1: Resolve the Restaurant Shortlist
1. **Explicit List in Chat**: Resolve any Google Maps short links and enrich against BigQuery (`filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master`) via [fetch_bq_shortlist.py](file:///usr/local/google/home/filipegracio/.gemini/config/skills/restaurant-uniqueness-explorer/scripts/fetch_bq_shortlist.py) (`--names`).
2. **Shortlist from BigQuery**: Run [fetch_bq_shortlist.py](file:///usr/local/google/home/filipegracio/.gemini/config/skills/restaurant-uniqueness-explorer/scripts/fetch_bq_shortlist.py) with `--outcode`, `--borough`, `--search`, `--min-match-score`, or `--unrated-only`.

### Step 2: Launch 1 Subagent Per Restaurant in Parallel
Populate [subagent_prompt_template.md](file:///usr/local/google/home/filipegracio/.gemini/config/skills/restaurant-uniqueness-explorer/references/subagent_prompt_template.md) and launch **all per-restaurant subagents in a single `invoke_subagent` call** (`TypeName: "self"`, `Model: "inherit"`). Do NOT poll; end your turn and wait for subagent replies.

### Step 3: Ultra-Concise Output Standard (30% / 70% Weighting)
> **Research deeply, write ultra-concisely**: Subagents must investigate menus, delivery apps, diaspora reviews, native-language dish names, and culinary science during search, then compress the output into tight 1-line bullets under clean headings (never write "Part A/B" or percentage labels in the headings):

* **Executive Comparison Table**: 1 row per restaurant + 1-line decision rule per venue.
* **`### 6-Pillar Check`**: 6 one-line bullets (`1. Value`, `2. Community`, `3. Linguistic/Insider`, `4. Geography`, `5. Uncompromisingness`, `6. Integrity`, max ~20 words per bullet).
* **`### What Would I Order? (Uniqueness: XX/100)`**:
  - **Top 5 Unique Dishes**: **1 single bullet per dish** (`English Name` (`Native Script` — £Price): *Why it's rare + flavor/health note in 20–25 words*).
  - **Feast for 2 (F&A) Blueprint**: 4 one-line bullets (`Fast Openers`, `Centerpiece`, `Fibre & Starch`, `Tip & Finish`).
  - **Skip**: 1 single comma-separated line of menu traps to ignore.

### Step 4: Deliver Commentable `.md` Artifact
Write the synthesized comparison to `<appDataDir>/brain/<conversation-id>/restaurant_uniqueness_guide.md` (`UserFacing: true`, `RequestFeedback: false`). Keep BigQuery read-only unless explicitly asked to update ratings/notes.
