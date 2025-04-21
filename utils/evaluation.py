import numpy as np
import torch
import matplotlib.pyplot as plt
from utils.config import device


def smith_waterman(seq1, seq2, match=1, mismatch=-1, gap=-1):
    """
    smith_waterman code adapted from slavianap's Github: https://github.com/slavianap/Smith-Waterman-Algorithm/blob/master/Script.py
    """
    # initialize score and tracing matrices
    m, n = len(seq1), len(seq2)
    score_matrix = np.zeros((m + 1, n + 1))
    tracing_matrix = np.zeros((m + 1, n + 1), dtype=int)
    max_score = -1
    max_pos = (0, 0)

    # fill matrices
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = match if seq1[i - 1] == seq2[j - 1] else mismatch
            diag = score_matrix[i - 1, j - 1] + s
            up = score_matrix[i - 1, j] + gap
            left = score_matrix[i, j - 1] + gap
            best = max(0, diag, up, left)
            score_matrix[i, j] = best
            if best == 0:
                tracing_matrix[i, j] = 0
            elif best == diag:
                tracing_matrix[i, j] = 1
            elif best == up:
                tracing_matrix[i, j] = 2
            else:
                tracing_matrix[i, j] = 3
            if best >= max_score:
                max_score = best
                max_pos = (i, j)

    # traceback to build aligned sequences
    aligned_seq1 = ""
    aligned_seq2 = ""
    i, j = max_pos
    while score_matrix[i, j] > 0:
        if tracing_matrix[i, j] == 1:
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            i -= 1
            j -= 1
        elif tracing_matrix[i, j] == 2:
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = '-' + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = '-' + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            j -= 1
    return score_matrix, max_score, aligned_seq1, aligned_seq2


def evaluate_model_sw(model, loader, idx_to_aa, samples=10):
    model.eval()
    sw_scores = []
    examples = []
    with torch.no_grad():
        for batch in loader:
            dist_input = batch["dist_input"].to(device)
            seq_target = batch["seq_target"].to(device)
            # get output from model
            logits = model(dist_input)
            # check if contains some other output like mu, var from VAE
            if isinstance(logits, tuple):
                logits = logits[0]
            # argmax logit into indices
            pred_indices = torch.argmax(logits, dim=-1)
            batch_size = pred_indices.size(0)
            for i in range(batch_size):
                # transform indices into sequence and apply smith-waterman algorithm
                pred_seq = "".join([idx_to_aa[int(idx)] for idx in pred_indices[i].cpu().numpy()])
                true_seq = "".join([idx_to_aa[int(idx)] for idx in seq_target[i].cpu().numpy()])
                _, score, aligned_pred, aligned_true = smith_waterman(pred_seq, true_seq)
                sw_scores.append(score)
                # append first 10 as the examples
                if len(examples) < samples:
                    examples.append({
                        "pred_seq": pred_seq,
                        "true_seq": true_seq,
                        "aligned_pred": aligned_pred,
                        "aligned_true": aligned_true,
                        "sw_score": score
                    })
    avg_sw = np.mean(sw_scores)
    return avg_sw, sw_scores, examples


def evaluate_duel_model_sw(seq_model, hetero_model, loader, idx_to_aa, samples=10, transform_matrix=None):
    seq_model.eval()
    hetero_model.eval()
    sw_scores = []
    examples = []

    with torch.no_grad():
        for batch in loader:
            dist_input = batch["dist_input"].to(device)
            seq_input = batch["seq_input"].to(device)
            seq_target = batch["seq_target"].to(device)
            # get output from model
            z_het = hetero_model(dist_input)
            _, z_gt = seq_model(seq_input)

            batch_size = seq_input.size(0)
            z_het = seq_model.fc_dec(z_het)
            z_gt = seq_model.fc_dec(z_gt)
            z_het = z_het.view(batch_size, 128, 8, 3)
            z_gt = z_gt.view(batch_size, 128, 8, 3)
            recon_seq = seq_model.decoder(z_het)
            recon_gt = seq_model.decoder(z_gt)
            if transform_matrix is not None:
                transform_matrix_inverse = torch.pinverse(transform_matrix).to(device)
                recon_seq = torch.matmul(recon_seq.squeeze(1), transform_matrix_inverse)
                recon_gt = torch.matmul(recon_gt.squeeze(1), transform_matrix_inverse)
            # argmax logit into indices
            pred_indices = torch.argmax(recon_seq, dim=-1)
            gt_indices = torch.argmax(recon_gt, dim=-1)

            for i in range(batch_size):
                # transform indices into sequence and apply smith-waterman algorithm
                pred_seq = "".join([idx_to_aa[int(idx)] for idx in pred_indices[i].cpu().numpy()])
                true_seq = "".join([idx_to_aa[int(idx)] for idx in seq_target[i].cpu().numpy()])
                _, score, aligned_pred, aligned_true = smith_waterman(pred_seq, true_seq)
                sw_scores.append(score)
                # append first 10 as the examples
                if len(examples) < samples:
                    examples.append({
                        "pred_seq": pred_seq,
                        "true_seq": true_seq,
                        "aligned_pred": aligned_pred,
                        "aligned_true": aligned_true,
                        "sw_score": score
                    })
    avg_sw = np.mean(sw_scores)
    return avg_sw, sw_scores, examples


def plot_learning_curve(train_losses, val_losses, model_name="Model"):
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", linestyle='-')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss", linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("./images/" + model_name + ".pdf", format="pdf", bbox_inches="tight")
    plt.show()


def plot_sw_distribution(smith_scores, model_name="model"):
    print(f"Collected {len(smith_scores)} Smith-Waterman scores.")
    plt.figure(figsize=(8, 6))
    plt.hist(smith_scores, bins=20, edgecolor='black')
    plt.xlabel("SW Score")
    plt.ylabel("Frequency")
    plt.title(f"{model_name} Smith-Waterman Score Distribution on Test Set")
    plt.savefig("./images/" + model_name + "_sw.pdf", format="pdf", bbox_inches="tight")
    plt.show()


def clear_gpu_memory():
    import gc
    gc.collect()
    torch.cuda.empty_cache()
