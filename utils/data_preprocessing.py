import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


# ## Data Preprocessing Helper Function

def read_aa_csv(file_path):
    # Read CSV with columns: PDB_ID, Chain_ID, Amino_Acid_Sequence
    df = pd.read_csv(file_path, names=["PDB_ID", "Chain_ID", "Amino_Acid_Sequence"])
    return df


def read_coordinates_excel(alpha_c_dir, pdb_id, chain_id):
    # Read Excel file of coordinates for the given pdb_id and chain_id
    file_path = Path(f"{alpha_c_dir}/{pdb_id}_{chain_id}.xlsx")
    if not file_path.exists():
        return np.array([])
    df = pd.read_excel(file_path, header=None)
    coords = df.to_numpy(dtype=float)
    return coords


def plot_sequence_length_distribution(df, column_name="Sequence_Length", bins=10):
    # Plot histogram of sequence lengths using defined bins
    bin_edges = range(0, 1024, bins)
    plt.figure(figsize=(8, 6))
    n, bins, patches = plt.hist(df[column_name], bins=bin_edges, edgecolor='black')
    plt.xticks(bins)
    plt.title("Distribution of Amino Acid Sequence Length")
    plt.xlabel("Sequence Length")
    plt.ylabel("Count")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def check_file_exist(df, alpha_c_path):
    # Filter rows where the coordinate file exists
    keep_flags = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        pdb_id = row["PDB_ID"]
        chain_id = str(row["Chain_ID"])
        file_path = Path(f"{alpha_c_path}/{pdb_id}_{chain_id}.xlsx")
        keep_flags.append(file_path.exists())
    df_filtered = df[keep_flags].copy()
    print(f"Removed {len(df) - len(df_filtered)} rows due to missing coordinate files.")
    return df_filtered


def remove_unknown(df):
    # Remove rows with unknown or rare amino acids (X or U)
    original_count = len(df)
    df_filtered = df[~df["Amino_Acid_Sequence"].str.contains("X|U")].copy()
    print(f"Removed {original_count - len(df_filtered)} rows with unknown amino acids.")
    return df_filtered


def remove_duplicate_sequences(df):
    # Remove duplicate sequences based on Amino_Acid_Sequence
    original_count = len(df)
    df_unique = df.drop_duplicates(subset=["Amino_Acid_Sequence"])
    print(f"Removed {original_count - len(df_unique)} duplicate sequences.")
    return df_unique


def filter_unique_pdb(df):
    # Keep one random row per unique PDB_ID
    filtered_df = df.groupby("PDB_ID", group_keys=False).sample(n=1)
    print(f"Filtered to {len(filtered_df)} unique PDB IDs.")
    return filtered_df


# Functions for distance matrix computation and sequence encoding

def normalize_distance_matrix(dist_mat):
    # Normalize the distance matrix by its maximum value
    max_val = dist_mat.max()
    if max_val > 1e-8:
        dist_mat = dist_mat / max_val
    return dist_mat


def compute_distance_matrix(coords, L=64):
    coords = np.array(coords, dtype=np.float32)
    n = len(coords)
    if n == 0:
        return np.zeros((L, L), dtype=np.float32)
    full_dist_mat = np.zeros((n, n), dtype=np.float32)
    # Compute the distance matrix from coordinates
    for i in range(n):
        for j in range(n):
            full_dist_mat[i, j] = 0.0 if i == j else np.linalg.norm(coords[i] - coords[j])

    # normalize the distance matrix
    full_dist_mat = normalize_distance_matrix(full_dist_mat)
    if n >= L:
        # intercept first L sequence from origin sequences
        dist_mat = full_dist_mat[:L, :L]
    else:
        # padding 0 for sequence < L which should never be reached
        dist_mat = np.zeros((L, L), dtype=np.float32)
        dist_mat[:n, :n] = full_dist_mat
    return dist_mat


def get_unique_amino_acids(sequences):
    # Get a set of unique amino acids from the list of sequences
    unique_aas = set()
    for seq in sequences:
        unique_aas.update(seq)
    return unique_aas


def create_amino_acid_mapping(unique_aas):
    # Create a mapping from amino acid to index
    sorted_aas = sorted(list(unique_aas))
    aa_to_idx = {aa: i for i, aa in enumerate(sorted_aas)}
    return aa_to_idx


def one_hot_encode_sequence(seq, max_len, aa_to_idx, num_aa):
    # Convert sequence to one-hot encoding of fixed length max_len
    arr = np.zeros((max_len, num_aa), dtype=np.float32)
    for i, aa in enumerate(seq):
        if i >= max_len:
            break
        idx = aa_to_idx.get(aa, -1)
        if idx >= 0:
            arr[i, idx] = 1.0
    return arr


def generate_random_symmetric_matrix(num_aa):
    # Generate a random symmetric matrix of shape (num_aa, num_aa)
    random_matrix = torch.rand(num_aa, num_aa)
    transform_matrix = (random_matrix + random_matrix.t()) / 2
    return transform_matrix
