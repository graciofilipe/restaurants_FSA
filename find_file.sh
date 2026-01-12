#!/bin/bash

# Check arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <directory_path> <partial_filename>"
    echo "Example: $0 /home/user config"
    exit 1
fi

SEARCH_DIR="$1"
SEARCH_TERM="$2"

echo "--- Searching for files containing '$SEARCH_TERM' (case-insensitive) in '$SEARCH_DIR' ---"

# Check directory existence
if [ ! -d "$SEARCH_DIR" ]; then
    echo "Error: Directory '$SEARCH_DIR' does not exist."
    exit 1
fi

# Run find
# -iname "*$SEARCH_TERM*": Adds wildcards before and after the term to find substrings
find "$SEARCH_DIR" -type f -iname "*$SEARCH_TERM*" 2>/dev/null

echo "--- Search Complete ---"