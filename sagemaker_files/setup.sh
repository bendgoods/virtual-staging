#!/bin/sh

# Pull changes from Github 
# git stash
git pull origin main

# Create model .tar.gz file
mkdir code
cp sagemaker_files/inference.py code/inference.py
cp -r models/ code/
cp requirements.txt code/
tar -czvf model.tar.gz code/ model-v1.0.pt

# Remove unwanted files
rm -r code
