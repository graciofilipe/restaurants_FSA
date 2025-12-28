# Product Guide: FSA API Explorer & Restaurant Discovery Tool

## Initial Concept
A personal Streamlit application designed to automate the discovery of newly opened restaurants using data from the Food Standards Agency (FSA) API. The tool identifies establishments that are not yet in the user's "master list," facilitating early discovery of interesting dining spots before they become widely popular.

## Target Users
- **Primary User:** A food enthusiast and developer (personal use) seeking to find new restaurants.
- **Goal:** To identify and research new establishments as soon as they appear in official records.
- **Access:** Open-access application with no mandatory authentication.

## Core Value Proposition
- **Automated Discovery:** Eliminates manual searching by automatically fetching and comparing new FSA data against a historical baseline.
- **Early Access:** Highlights new openings immediately, allowing the user to visit before places get too popular.
- **Efficient Research:** Streamlines the transition from "discovery" to "research" by providing necessary details for Google Maps and review lookups.

## Key Features
- **Delta Analysis:** Automatically compares fetched API data with the stored BigQuery "master list" to isolate and display *only* new restaurants.
- **Data Persistence:** Uses Google BigQuery to maintain a permanent record of all seen restaurants, ensuring accurate "new vs. old" comparison.
- **Research Aids:** Provides key details (and potential direct links) to external tools like Google Maps to facilitate manual review of new finds.
- **Integrated Google Maps Links:** Automatically generates clickable search links for every newly discovered restaurant to speed up the research process.
