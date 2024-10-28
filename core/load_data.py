# @title Load Data

from peptide_dataset import PeptideDataset

input_file = "/content/drive/MyDrive/Aa_Lokey_Lab/cyclic_peptide_training_set.csv"

dataset = PeptideDataset(input_file)
dataset.clean()

print(f"Number of peptides in the dataset: {len(dataset)}")
