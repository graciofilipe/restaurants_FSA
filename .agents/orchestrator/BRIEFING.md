# Orchestrator Briefing

## 🔒 My Identity
- **Role**: Project Orchestrator
- **TypeName**: orchestrator
- **Mission**: Complete user request to add async BQML training to Streamlit app and verify it.

## 🔒 Key Constraints
- I DO NOT write/edit source code directly. I ONLY use file-editing tools for .md metadata in `.agents/` directory.
- Require workers to run `pytest` and `blaze build/test`.
- Run forensic auditor at the end. Audit is a BINARY VETO.
- Ensure Hand-off protocols are strictly followed.

## 🔒 My Workflow
- Pattern: Project Pattern / SWE
- Iteration Config: 3 Explorers, 1 Worker, 2 Reviewers, 2 Challengers, 1 Auditor
- Sub-orchestrators: None required. Scope fits single cycle.
- Current Milestones:
  - 1: Streamlit Async BQML Trigger (Reviewing/Auditing pending)

## Succession Status
- Spawn count: 6 / 16
- Pending subagents:
  - 226327f5-2fa0-4bcc-a0c3-295569b81be1 (Reviewer 1)
  - d1bd3257-8d00-4c00-86e3-30271a3a2c52 (Reviewer 2)
  - 68a4e1a3-ae4c-4539-9582-e5cdf56078b1 (Challenger 1)
  - c19fd53b-b9a1-4474-ab2c-abd9e401190d (Challenger 2)
  - 81597ccd-8ec6-4b1c-9721-58db894a1ede (Auditor)

## Team Roster
- Agent ID: ec7659ea-a6a0-4923-bbe4-4a34a032427d | Archetype: worker | Task: Commit tests | Status: completed
- Agent ID: 226327f5-2fa0-4bcc-a0c3-295569b81be1 | Archetype: reviewer | Task: Review | Status: in-progress
- Agent ID: d1bd3257-8d00-4c00-86e3-30271a3a2c52 | Archetype: reviewer | Task: Review | Status: in-progress
- Agent ID: 68a4e1a3-ae4c-4539-9582-e5cdf56078b1 | Archetype: challenger | Task: Empirically verify | Status: in-progress
- Agent ID: c19fd53b-b9a1-4474-ab2c-abd9e401190d | Archetype: challenger | Task: Empirically verify | Status: in-progress
- Agent ID: 81597ccd-8ec6-4b1c-9721-58db894a1ede | Archetype: auditor | Task: Forensic Audit | Status: in-progress
