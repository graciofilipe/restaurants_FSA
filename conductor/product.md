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
- **Scheduled Delta Analysis:** A weekly Cloud Job automatically fetches new FSA data and compares it with the stored BigQuery "master list" to isolate new restaurants.
- **Data Persistence:** Uses Google BigQuery to maintain a permanent record of all seen restaurants, ensuring accurate "new vs. old" comparison.
- **Research Aids:** Provides key details to facilitate manual review of new finds on external tools like Google Maps.
- **AI Agent & Insights:** A dedicated Agent workflow that can research specific restaurants using Google Maps data.
- **Automated Insights Display:** Automatically displays generated insights (cuisine type, ratings, review summaries) in the UI immediately after generation.
- **Bulk Status Update:** Allows users to select multiple restaurants and update their review status (e.g., 'accepted', 'rejected') in batch.
