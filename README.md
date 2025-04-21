# CSI6900 Project - Designing Amino Acid Sequences for Target Protein Structures Using Deep Learning


## Overview

The primary goal of the project is to develop a system that generates amino acid sequences for specific target protein structures. It includes four deep learning models:

- **Multi-Scale DistanceTransformer**  
  A multi-scale convolutional model with residual connections and a transformer encoder to extract and process features from distance matrices.

- **SequenceAutoencoder + Heteroencoder**  
  **SequenceAutoencoder**  to process protein sequence one-hot encodings. This model is paired with a **Heteroencoder** that maps distance matrix features to the latent space.

- **ProteinCNNLSTM**  
  A hybrid model that uses ResidualCNN-based feature extraction with an LSTM decoder for generating protein sequences.


To evaluate model performance, the project employs a scoring metric based on the Smith-Waterman sequence alignment algorithm.

## Project Structure
```
CSI6900/ 
├── main.py # Main entry script with command-line argument controls for training and evaluation
├── main.ipynb # Jupyter Notebook version for interactive development and experimentation 
├── requirements.txt # List of Python dependencies 
├── models/ # Contains model definitions 
│ ├── init.py 
│ ├── common.py 
│ ├── distance_transformer.py 
│ ├── cnn_sequence_vae.py 
│ ├── hetero_vae.py 
│ └── protein_cnnlstm.py 
├── train/ # Training scripts for individual models 
│ ├── init.py 
│ ├── train_distance_transformer.py 
│ ├── train_dual_vae.py 
│ └── train_protein_cnnlstm.py 
└── utils/ # Utility functions: data preprocessing, dataset construction, evaluation, and configurations 
  ├── init.py 
  ├── config.py 
  ├── data_preprocessing.py 
  ├── dataset.py 
  └── evaluation.py
```

## Dependencies

Before running the project, install the required Python packages using:

```bash
pip install -r requirements.txt
```
The dependencies include:
- torch~=2.5.1+cu124
- matplotlib~=3.10.0
- numpy~=2.0.1
- pandas~=2.2.3
- tqdm~=4.66.5
- seaborn~=0.13.2
- scikit-learn~=1.6.1

## Running the Project
### Jupyter Notebook
The project provides an interactive Jupyter Notebook (`main.ipynb`) for experimentation and visualization. To start the notebook, run:
```bash
jupyter notebook main.ipynb
```
### Python Script
The main.py script is a standard Python entry point with command-line arguments that allow you to control:
1. **Data Paths:**
   - `--csv_path`: Path to the CSV file containing protein sequences (default: `"dataset/large/AA.csv"`).
   - `--alpha_c_dir`: Directory containing protein coordinate Excel files (default: `"dataset/large/PDBalphaC"`).
2. **Data Sampling:**
   - `--sample`: Number of samples to use (default: `0`, which means using the entire dataset).
3. **Model Selection:**
   - `--model`: Model to train and evaluate.(The default is `DistanceTransformer`.) Options include:
     - `MultiScaleDistanceTransformer`
     - `SequenceAutoencoder`
     - `ProteinCNNLSTM` 

4. **Hyperparameters:**
   - `--lr`: Learning rate (default: `1e-3`).
   - `--batch_size`: Batch size (default: `128`).
   - `--num_epochs`: Number of training epochs (default: `50000` for DistanceTransformer; adjust this value accordingly for other models).

To run the script, use a command similar to:

```bash
python main.py --csv_path dataset/large/AA.csv --alpha_c_dir dataset/large/PDBalphaC --model DistanceTransformer --batch_size 128 --lr 1e-3 --num_epochs 50000
```

Modify the parameters as needed for different experiments.

You can change the global setting at ./utils/config.py
## Contact
Tiancheng Qin - [tqin021@uottawa.ca](tqin021@uottawa.ca)
