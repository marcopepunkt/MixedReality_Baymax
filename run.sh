#!/bin/bash

# Default values
DEFAULT_GEMINI_API_KEY=""
DEFAULT_MAPS_API_KEY=""
DEFAULT_IP="192.168.1.245"


# Parse arguments
GEMINI_API_KEY=$DEFAULT_GEMINI_API_KEY
MAPS_API_KEY=$DEFAULT_MAPS_API_KEY
IP=$DEFAULT_IP

# Run the Python script with the parsed or default values
python3 ./scripts/baymax.py --GEMINI_API_KEY "$GEMINI_API_KEY" --MAPS_API_KEY "$MAPS_API_KEY" --IP "$IP"