# Specification: Prompt Updates and Agent Enhancement

## Overview
This track involves updating prompts for Gemini bulk analysis and the Maps agent to improve output structure and readability. It also includes refactoring the `get_agent_insight` function to pass additional address details to the agent, providing richer context for its research.

## Functional Requirements

### 1. Gemini Bulk Analysis Prompt Update
- **Goal:** Place the "Final Verdict" at the very top of the generated report.
- **Scope:** Apply this change to all Gemini bulk analysis reports.
- **Behavior:** The analysis should still perform careful research *before* generating the verdict, but the final output string must present the verdict first.

### 2. Maps Agent Instruction Update
- **Goal:** Standardize the agent's output format.
- **Format:** The agent must output its findings in **JSON format**.
- **Content:** The JSON should capture the key insights (cuisine, rating, review summary, etc.) in a structured way to facilitate downstream parsing and display.

### 3. `get_agent_insight` Enhancement
- **Goal:** Provide more location context to the agent.
- **Input:** Update the function signature or internal logic to extract `AddressLine2` and `LocalAuthorityName` from the restaurant data object.
- **Usage:** Include these new fields in the prompt sent to the Maps agent alongside `AddressLine1` and `PostCode`.
- **Error Handling:** These fields are **optional**. If `AddressLine2` or `LocalAuthorityName` are missing or empty in the source data, pass an empty string or appropriate placeholder to the agent (do not fail).

## Non-Functional Requirements
- **Backward Compatibility:** Ensure the changes to `get_agent_insight` do not break existing calls to this function (though the internal prompt construction will change).
- **Code Quality:** Maintain existing error handling and logging patterns.

## Acceptance Criteria
- [ ] Gemini bulk analysis reports display the final verdict at the top.
- [ ] Maps agent responses are consistently returned in valid JSON format.
- [ ] The `get_agent_insight` function correctly extracts and uses `AddressLine2` and `LocalAuthorityName` when available.
- [ ] The system handles missing address fields gracefully without errors.
