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
