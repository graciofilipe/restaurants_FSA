# Food Standards Agency API Explorer

This application allows users to fetch data from the Food Standards Agency (FSA) API based on geographical coordinates.

## Functionality

- Users can input **longitude** and **latitude** values.
- The application makes a request to the FSA API using these coordinates to find relevant food establishment data.
- If the API request is successful, users can **download the resulting data in JSON format**.
- If the API request fails, an appropriate error message is displayed.

## Installation

1.  Clone this repository or download the source code.
2.  Ensure you have Python installed (version 3.7+ recommended).
3.  Install the necessary dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

Once the dependencies are installed, you can run the Streamlit application using the following command:

```bash
streamlit run st_app.py
```

This will typically open the application in your default web browser.

## Authentication & Setup

This application uses Firebase Authentication (Google Sign-In).

### Configuration
1.  Set up a Firebase project and enable Google Sign-In.
2.  Add your Firebase configuration to `.streamlit/secrets.toml`.

### Authorized Domains
If deploying to Cloud Run (or any other host), you must add your domain to the **Authorized Domains** list in the Firebase Console:
1.  Go to **Authentication** > **Settings** > **Authorized Domains**.
2.  Add your application's domain (e.g., `your-service-xyz.run.app`).

*Note: If you skip this, users will see a "Configuration Error" regarding unauthorized domains when trying to log in.*

## AI Agent Prototype (Maps Grounding)

A prototype AI Agent capable of answering questions about restaurants using Google Maps Grounding is available.

### Usage
To interact with the agent via CLI:
```bash
python scripts/prototype_maps_agent.py
```

### Requirements
- Google Cloud Project with Vertex AI API enabled.
- Google Maps Grounding enabled in the project/agent configuration.
- `google-adk` installed.
