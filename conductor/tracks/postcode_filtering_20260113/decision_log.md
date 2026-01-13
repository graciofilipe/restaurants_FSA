# Decision Log: Postcode Extraction Method

**Date:** 2026-01-13
**Track:** postcode_filtering_20260113

## Context
We need to extract the "outcode" (first part) of the postcode from the `PostCode` string column in the dataframe. Two methods were proposed:
1. **Whitespace Split:** `postcode.split(' ')[0]`
2. **Regex Extraction:** `re.match(r'^([A-Z]{1,2}[0-9]{1,2}[A-Z]?)', postcode)`

## Benchmark Results
- **Iterations:** 100,000 passes over 12 test cases.
- **Method A (Split):** ~0.56s
- **Method B (Regex):** ~1.24s

## Observations
- **Performance:** Method A is approximately **2.2x faster** than Method B.
- **Accuracy:** Both methods correctly distinguish between "SE1" and "SE14".
- **Edge Cases:** 
    - Method A returns the full string if no space is present (e.g., "INVALID" -> "INVALID").
    - Method B returns `None` if the pattern doesn't match.

## Decision
**Selected Method: Method A (Whitespace Split)**

**Reasoning:**
1. **Performance:** The significant speed advantage is crucial for runtime data processing in Streamlit, especially as the dataset grows.
2. **Simplicity:** The logic is standard and easy to read (`str.split`).
3. **Data Validity:** FSA data is generally well-formatted. If "dirty" data (no space) exists, having it appear as a distinct filter option (the full string) is acceptable behavior for a discovery tool, allowing the user to see/filter those odd rows.

## Implementation Details
We will use the Pandas vectorized string accessor `.str.split(' ').str[0]` for maximum performance on the full DataFrame.
