import argparse
import torch
from torch.utils.data import DataLoader, random_split

# Import configuration
from utils.config import device
from utils import evaluation
from utils.data_preprocessing import (read_aa_csv, plot_sequence_length_distribution, check_file_exist,
                                      remove_unknown, remove_duplicate_sequences, filter_unique_pdb,
                                      get_unique_amino_acids, create_amino_acid_mapping,
                                      generate_random_symmetric_matrix)
from utils.dataset import ProteinDataset

# Import training functions from train modules
from train.train_distance_transformer import train_distance_transformer
from train.train_dual_ae import train_SequenceAutoencoder_and_heteroencoder
from train.train_protein_cnnlstm import train_protein_cnnlstm


# Import model classes
from models.distance_transformer import MultiScaleDistanceTransformer
from models.cnn_sequence_ae import SequenceAutoencoder
from models.heteroencoder import Heteroencoder
from models.protein_cnnlstm import ProteinCNNLSTM



def main(args):
    # Define dataset parameters
    csv_path = args.csv_path  # CSV file path containing protein sequences
    alpha_c_dir = args.alpha_c_dir  # Directory with protein coordinate Excel files
    max_len = 64

    print("Loading CSV data...")
    df = read_aa_csv(csv_path)
    # Compute sequence lengths
    df["Sequence_Length"] = df["Amino_Acid_Sequence"].apply(len)

    # Optionally, plot sequence length distribution
    # plot_sequence_length_distribution(df, column_name="Sequence_Length", bins=64)

    # Filter DataFrame based on sequence length
    df_filtered = df[(df["Sequence_Length"] >= max_len) & (df["Sequence_Length"] <= max_len * 2)].copy()
    # Remove rows without coordinate files
    df_filtered = check_file_exist(df_filtered, alpha_c_dir)
    # Remove rows with unknown amino acids
    df_filtered = remove_unknown(df_filtered)
    # Filter for unique PDB IDs
    df_filtered = filter_unique_pdb(df_filtered)
    # Remove duplicate sequences
    df_filtered = remove_duplicate_sequences(df_filtered)
    # If sample number is provided and less than dataset size, sample records
    if args.sample and args.sample < len(df_filtered):
        df_filtered = df_filtered.sample(args.sample, random_state=2025)

    # Build amino acid mapping
    all_seqs = df_filtered["Amino_Acid_Sequence"].tolist()
    unique_aas = get_unique_amino_acids(all_seqs)
    aa_to_idx = create_amino_acid_mapping(unique_aas)
    idx_to_aa = {v: k for k, v in aa_to_idx.items()}
    num_aa = len(aa_to_idx)
    print("Number of distinct amino acids:", num_aa)

    # Optionally define a transformation matrix (set to None by default)
    transform_matrix = None
    # Uncomment below line to use a random symmetric transformation matrix
    # transform_matrix = generate_random_symmetric_matrix(num_aa).to(device)

    # Build dataset
    full_dataset = ProteinDataset(df_filtered, alpha_c_dir, aa_to_idx, num_aa, max_len=max_len,
                                  transform_matrix=transform_matrix)
    print("Dataset size:", len(full_dataset))

    # Split the dataset: 80% train, 10% validation, 10% test
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size
    train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size],
                                                generator=torch.Generator().manual_seed(2025))
    print(f"Train: {len(train_set)}, Validation: {len(val_set)}, Test: {len(test_set)}")

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    # Model selection and training
    model_choice = args.model.lower()
    if model_choice == "multiscaledistancetransformer":
        print("Training DistanceTransformer model...")
        model = MultiScaleDistanceTransformer(max_len=max_len, num_aa=num_aa, latent_dim=256, d_model=256, dropout_rate=0.2).to(
            device)
        model, train_losses, val_losses = train_distance_transformer(
            model, train_loader, val_loader, num_epochs=args.num_epochs, lr=args.lr, weight_decay=1e-4, patience=300)
        # Evaluate model
        avg_sw, sw_scores, sw_examples = evaluation.evaluate_model_sw(model, test_loader, idx_to_aa)
        print(f"DistanceTransformer Average SW Score: {avg_sw:.4f}")
        for ex in sw_examples:
            print(ex)
        evaluation.plot_learning_curve(train_losses, val_losses, model_name="DistanceTransformer")
        evaluation.plot_sw_distribution(sw_scores)

    elif model_choice == "sequenceautoencoder":
        print("Training Dual AE (SequenceAutoencoder + Heteroencoder) model...")
        seq_vae = SequenceAutoencoder(latent_dim=256, dropout_rate=0.2).to(device)
        het_vae = Heteroencoder(latent_dim=256, dropout_rate=0.2).to(device)
        seq_vae, het_vae, train_losses, val_losses = train_SequenceAutoencoder_and_heteroencoder(
            seq_vae, het_vae, train_loader, val_loader, num_epochs=args.num_epochs, lr=args.lr, lambda_mapping=1,
             patience=500)
        # Evaluate model
        avg_sw, sw_scores, sw_examples = evaluation.evaluate_duel_model_sw(seq_vae, het_vae, test_loader, idx_to_aa)
        print(f"Dual VAE Average SW Score: {avg_sw:.4f}")
        for ex in sw_examples:
            print(ex)
        evaluation.plot_learning_curve(train_losses, val_losses, model_name="DualVAE")
        evaluation.plot_sw_distribution(sw_scores)

    elif model_choice == "proteincnnlstm":
        print("Training ProteinCNNLSTM model...")
        model = ProteinCNNLSTM(num_aa=num_aa, latent_dim=256, max_len=max_len, dropout_rate=0.3).to(device)
        model, train_losses, val_losses = train_protein_cnnlstm(
            model, train_loader, val_loader, num_epochs=args.num_epochs, lr=args.lr, patience=200)
        # Evaluate model
        avg_sw, sw_scores, sw_examples = evaluation.evaluate_model_sw(model, test_loader, idx_to_aa)
        print(f"ProteinCNNLSTM Average SW Score: {avg_sw:.4f}")
        for ex in sw_examples:
            print(ex)
        evaluation.plot_learning_curve(train_losses, val_losses, model_name="ProteinCNNLSTM")
        evaluation.plot_sw_distribution(sw_scores)


    else:
        print(
            "Unknown model choice. Please select from: MultiScaleDistanceTransformer, SequenceAutoencoder, ProteinCNNLSTM.")

    # Clear GPU memory
    # evaluation.clear_gpu_memory()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CSI6900 Protein Design Project")
    parser.add_argument("--csv_path", type=str, default="dataset/large/AA.csv",
                        help="Path to the CSV file containing protein sequences")
    parser.add_argument("--alpha_c_dir", type=str, default="dataset/large/PDBalphaC",
                        help="Directory containing protein coordinate Excel files")
    parser.add_argument("--sample", type=int, default=0,
                        help="Number of samples to use (0 for using all)")
    parser.add_argument("--model", type=str, default="MultiScaleDistanceTransformer",
                        help="Model to train and evaluate: MultiScaleDistanceTransformer, SequenceAutoencoder, ProteinCNNLSTM")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50000, help="Number of training epochs")
    args = parser.parse_args()
    main(args)
