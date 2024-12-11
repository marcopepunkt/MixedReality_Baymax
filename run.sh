#!/bin/bash

# Default values
DEFAULT_API_KEY="AIzaSyD9yl2md73tLb9XkkC56m4T4KVH8yFmsVg"
DEFAULT_IP="192.168.1.245"

# Parse arguments
API_KEY=$DEFAULT_API_KEY
IP=$DEFAULT_IP

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --API_KEY) API_KEY="$2"; shift ;;
        --IP) IP="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the Python script with the parsed or default values
python3 ./scripts/baymax.py --API_KEY "$API_KEY" --IP "$IP"