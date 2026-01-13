import time
import re

def extract_via_split(postcode):
    """Method A: Whitespace Split"""
    if not isinstance(postcode, str):
        return None
    return postcode.split(' ')[0]

def extract_via_regex(postcode):
    """Method B: Regex Extraction"""
    if not isinstance(postcode, str):
        return None
    match = re.match(r'^([A-Z]{1,2}[0-9]{1,2}[A-Z]?)', postcode)
    if match:
        return match.group(1)
    return None

def benchmark():
    # Sample dataset with edge cases
    test_data = [
        "SE1 7PB",      # Standard
        "SW16 5RT",     # Longer outcode
        "E1 6AN",       # Short outcode
        "SE14 5QR",     # Similar prefix to SE1
        "WC2N 5DU",     # London central code
        "M1 1AA",       # Manchester short
        "B33 8TH",      # Birmingham
        "CR2 6XH",      # Croydon
        "W1A 1AA",      # Special code
        "INVALID",      # No space (Edge case for split)
        "",             # Empty
        None            # None
    ]
    
    # Expected results for verification (excluding None/Invalid for strict comparison if needed)
    # We mainly care that SE1 and SE14 are distinct.
    
    iterations = 100000
    
    print(f"Benchmarking {iterations} iterations over {len(test_data)} test cases...\n")
    
    # --- Test Method A: Split ---
    start_time = time.time()
    for _ in range(iterations):
        for pc in test_data:
            extract_via_split(pc)
    end_time = time.time()
    split_duration = end_time - start_time
    
    print(f"Method A (Split) Duration: {split_duration:.4f} seconds")
    
    # --- Test Method B: Regex ---
    start_time = time.time()
    for _ in range(iterations):
        for pc in test_data:
            extract_via_regex(pc)
    end_time = time.time()
    regex_duration = end_time - start_time
    
    print(f"Method B (Regex) Duration: {regex_duration:.4f} seconds")
    
    # --- Verify Accuracy / Behavior Differences ---
    print("\n--- Accuracy & Behavior Check ---")
    print(f"{ 'Input':<15} | { 'Split Result':<15} | { 'Regex Result':<15}")
    print("-" * 50)
    
    for pc in test_data:
        res_split = extract_via_split(pc)
        res_regex = extract_via_regex(pc)
        print(f"{str(pc):<15} | {str(res_split):<15} | {str(res_regex):<15}")

    print("\n--- Recommendations ---")
    if split_duration < regex_duration:
        print("Recommendation: Method A (Split) is faster.")
    else:
        print("Recommendation: Method B (Regex) is faster.")

    # Check specific edge case "INVALID" (missing space)
    invalid_split = extract_via_split("INVALID")
    invalid_regex = extract_via_regex("INVALID")
    
    if invalid_split == "INVALID" and invalid_regex != "INVALID":
        print("Note: 'Split' returns full string if no space found. Regex captures pattern or None.")

if __name__ == "__main__":
    benchmark()
