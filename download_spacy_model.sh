#!/bin/bash
# Script to download the spacy English model

echo "Downloading spacy English model..."
python -m spacy download en_core_web_sm

echo "Spacy model downloaded successfully!"