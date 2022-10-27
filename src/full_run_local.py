# Single python script that runs the full training loop and INLP locally, decomposes the BERT embeddings and evaluates
# the results

# Libraries
import numpy as np
import pandas as pd
import pytorch as torch
import pickle

from load_bert import DataProcessing
from LinearClassifier import LinearClassifier, INLPTraining

#####
# Load and pre-process the data



#####
# Run the INLP loop


#####
# Define projection matrices and decompose embeddings


#####
# Run evaluation tasks on decomposed embeddings