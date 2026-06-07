#!/bin/bash

echo "Enabling Places API..."
gcloud services enable places-backend.googleapis.com

echo "Creating API key..."
gcloud services api-keys create   --display-name="Maps Places API Key for FSA app"   --api-target=service=places-backend.googleapis.com

echo ""
echo "========================================================"
echo "Please copy the keyString from the output above and set it"
echo "as your GOOGLE_MAPS_API_KEY environment variable:"
echo "export GOOGLE_MAPS_API_KEY="your_key_here""
echo "========================================================"
