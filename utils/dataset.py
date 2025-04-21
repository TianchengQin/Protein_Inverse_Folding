import torch
import numpy as np
from torch.utils.data import Dataset
from utils import data_preprocessing
from utils.config import device


class ProteinDataset(Dataset):
    def __init__(self, df, alpha_c_dir, aa_to_idx, num_aa, max_len=64, transform_matrix=None):
        super().__init__()
        self.records = []
        self.aa_to_idx = aa_to_idx
        self.num_aa = num_aa
        self.max_len = max_len
        self.alpha_c_dir = alpha_c_dir
        self.transform_matrix = transform_matrix

        # Iterate through each row in the DataFrame to build the dataset
        from tqdm import tqdm
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Building Dataset"):
            pdb_id = row["PDB_ID"]
            chain_id = str(row["Chain_ID"])
            seq = row["Amino_Acid_Sequence"]
            # Read protein coordinates from the Excel file
            coords = data_preprocessing.read_coordinates_excel(alpha_c_dir, pdb_id, chain_id)
            if len(coords) == 0:
                continue
            # Compute the distance matrix with a fixed maximum length
            dist_mat = data_preprocessing.compute_distance_matrix(coords, L=max_len)
            # One-hot encode the amino acid sequence
            one_hot_seq = data_preprocessing.one_hot_encode_sequence(seq, max_len, aa_to_idx, num_aa)
            # Append the processed record to the list
            self.records.append({
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "seq_arr": one_hot_seq,
                "dist_arr": dist_mat
            })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        # Convert the distance matrix to a tensor with shape (1, max_len, max_len)
        dist_input = torch.from_numpy(item["dist_arr"]).unsqueeze(0).float().to(device)
        # Convert the one-hot encoded sequence to a tensor with shape (1, max_len, num_aa)
        seq_input = torch.from_numpy(item["seq_arr"]).unsqueeze(0).float().to(device)
        # Create target tensor by taking the argmax across the one-hot dimension (shape: max_len)
        seq_target = torch.from_numpy(np.argmax(item["seq_arr"], axis=1)).long().to(device)
        # If a transformation matrix is provided, apply it to the sequence input
        if self.transform_matrix is not None:
            seq_input = torch.matmul(seq_input, self.transform_matrix)
        # Return a dictionary containing the inputs and metadata
        return {
            "dist_input": dist_input,
            "seq_input": seq_input,
            "seq_target": seq_target,
            "pdb_id": item["pdb_id"],
            "chain_id": item["chain_id"]
        }
