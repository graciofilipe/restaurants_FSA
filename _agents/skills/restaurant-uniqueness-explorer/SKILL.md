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

# Restaurant uniqueness and "what would I order?" explorer

Dispatches **1 dedicated research subagent per restaurant** in parallel to evaluate a shortlist of restaurants. Each subagent performs deep English and native-language menu and culinary-science research under the hood, then outputs an **ultra-concise, telegraph-style report** weighted across three sections:
1. **6-Pillar Check**: 6 one-line bullets verifying the *Healthy Host and Explorer* pillars plus a 3-tier Worthiness Verdict (~25% of output).
2. **What to skip or modify (top guardrail)**: Placed at the top of the ordering guide to block generic high-street staples available anywhere, over-ordered regional defaults, and hidden frying or sugary-glaze traps (~15% of output).
3. **What would I order? (the "unique and health-leaning" balance)**: First identify what is genuinely unique, regional, special, different, and unusual at the restaurant; then highlight the **health-leaning options and preparations** of those regional specialties across **(1) Unique and health-leaning fish and poultry highlights** and **(2) Unique vegetarian, plant-protein, and regional egg highlights** (including the healthiest version of foundational regional ferments). Format each dish with **Dish name**, **What it is** (max 2 sentences), and **What makes it unique** (1 sentence) (~60% of output).

---

## Workflow

### Step 1: Resolve the restaurant shortlist
1. **Explicit list in chat**: Resolve any Google Maps short links and enrich against BigQuery (`filipegracio-ai-learning.filipegracio_fsa_restaurants.fsa_master`) via [fetch_bq_shortlist.py](file:///usr/local/google/home/filipegracio/.gemini/config/skills/restaurant-uniqueness-explorer/scripts/fetch_bq_shortlist.py) (`--names`).
2. **Shortlist from BigQuery**: Run [fetch_bq_shortlist.py](file:///usr/local/google/home/filipegracio/.gemini/config/skills/restaurant-uniqueness-explorer/scripts/fetch_bq_shortlist.py) with `--outcode`, `--borough`, `--search`, `--min-match-score`, or `--unrated-only`.

### Step 2: Launch 1 subagent per restaurant in parallel
Populate [subagent_prompt_template.md](file:///usr/local/google/home/filipegracio/.gemini/config/skills/restaurant-uniqueness-explorer/references/subagent_prompt_template.md) and launch **all per-restaurant subagents in a single `invoke_subagent` call** (`TypeName: "self"`, `Model: "inherit"`). Do NOT poll; end your turn and wait for subagent replies.

### Step 3: Ultra-concise output standard
> **Research deeply, write ultra-concisely**: Subagents must investigate menus, delivery apps, diaspora reviews, native-language dish names, and culinary science during search, then compress the output into tight bullets under clean headings (never write "Part A/B" or percentage labels in the headings):

* **Executive comparison table**: 1 row per restaurant plus 1-line decision rule per venue.
* **`### 6-Pillar check`**: 6 one-line bullets (`1. Value`, `2. Community`, `3. Linguistic/Insider`, `4. Geography`, `5. Uncompromisingness`, `6. Integrity`, max ~20 words per bullet).
* **`### What to skip or modify (do not waste your appetite on generic dishes)`**: Placed **before** the recommended dishes, grouped into: (1) generic UK high-street staples available anywhere, (2) over-ordered regional defaults, (3) high-sugar glazes and double-fried starch fritters, and (4) hidden Stage 2 deep-fryer traps where a healthier regional dish or unfried preparation should be chosen instead.
* **`### What would I order? (Uniqueness: XX/100)`**:
  - **1. Unique and health-leaning fish and poultry highlights**: Rare regional dishes that showcase complex spice pastes, aromatics, and traditional techniques while leaning toward healthier cooking methods (steamed, poached, grilled, broth-simmered, or spice-braised without heavy frying or sugary glazes).
  - **2. Unique vegetarian, plant-based, and regional egg highlights**: Unusual regional vegetable dishes, slow-braised eggs, and the healthiest version of the cuisine's signature ferments or plant proteins (such as braised or simmered tempeh, tofu, or legumes—never dropping a foundational regional ferment just because its name appears on other menus).
  - **Dish format**: Each dish entry states **Dish name** (with price, region, and cooking/dietary tags), **What it is** (maximum 2 sentences), and **What makes it unique** (1 single sentence). Keep summary guides focused and free of foreign-language script blocks unless requested.

### Step 4: Deliver commentable `.md` artifact (and optional 1-page PDF for calendar invites)
Write the synthesized guide to `<appDataDir>/brain/<conversation-id>/restaurant_uniqueness_guide.md` (`UserFacing: true`, `RequestFeedback: false`). Keep BigQuery read-only unless explicitly asked to update ratings or notes. When requested as a PDF or for an upcoming calendar dinner invite, render a 1-page PDF (with **What to skip or modify** at the top, followed by the two unique and health-leaning dish sections), upload and share it on Google Drive (`--notify=false`) with attendees, and attach it to the calendar event.
