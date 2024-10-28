# @title Load Data

from peptide_dataset import PeptideDataset

input_file = "training_data/cycpeptmpdb_cleaned_training_set.csv" # Set this to the path of the dataset

dataset = PeptideDataset(input_file)
dataset.clean()

print(f"Number of peptides in the dataset: {len(dataset)}")
