#!/bin/bash

# Download the GloVe embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip

# Extract the GloVe embeddings
unzip glove.6B.zip -d data/raw

# Move the GloVe embeddings to the external folder
mv data/raw/glove.6B data/external/

# Delete the zip file
rm glove.6B.zip