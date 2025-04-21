import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.cnn_sequence_ae import SequenceAutoencoder
from models.heteroencoder import Heteroencoder


def cosine_similarity_loss(z1, z2):
    cos_sim = nn.functional.cosine_similarity(z1, z2, dim=1)
    return 1 - cos_sim.mean()


def train_SequenceAutoencoder_and_heteroencoder(seq_ae, heteroencoder, train_loader, val_loader, num_epochs=100,
                                                lr=1e-3, weight_decay=1e-5, lambda_mapping=1.0, patience=10):
    criterion = nn.MSELoss()
    optimizer = AdamW(list(seq_ae.parameters()) + list(heteroencoder.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20)
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_state_seq = None
    best_state_het = None
    train_losses = []
    val_losses = []
    annel_epoch = 5000
    for epoch in range(1, num_epochs + 1):
        seq_ae.train()
        heteroencoder.train()
        total_train_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()
            seq_input = batch['seq_input']
            dist_input = batch['dist_input']

            recon_seq, latent_seq = seq_ae(seq_input)
            loss_ae = criterion(recon_seq, seq_input)

            latent_het = heteroencoder(dist_input)
            loss_mapping = cosine_similarity_loss(latent_het, latent_seq)

            lambda_mapping_current = lambda_mapping * min(1, epoch / annel_epoch)
            total_loss = loss_ae + lambda_mapping_current * loss_mapping

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(seq_ae.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(heteroencoder.parameters(), 1.0)
            optimizer.step()

            total_train_loss += total_loss.item() * seq_input.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        seq_ae.eval()
        heteroencoder.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                seq_input = batch['seq_input']
                dist_input = batch['dist_input']

                recon_seq, latent_seq = seq_ae(seq_input)
                loss_ae = criterion(recon_seq, seq_input)
                latent_het = heteroencoder(dist_input)

                loss_mapping = cosine_similarity_loss(latent_het, latent_seq)

                loss = loss_ae + loss_mapping

                total_val_loss += loss.item() * seq_input.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)
        print(
            f"[Epoch {epoch + 1}/{num_epochs}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | AE: {loss_ae.item():.6f} | Map: {loss_mapping.item():.6f} ")
        if epoch > 100:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
                best_state_seq = seq_ae.state_dict()
                best_state_het = heteroencoder.state_dict()
                torch.save({
                    'epoch': epoch,
                    'seq_model_state_dict': best_state_seq,
                    'het_model_state_dict': best_state_het,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                }, 'best_model_dual_model.pth')


            else:
                early_stop_counter += 1
                if early_stop_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch + 1}")
                    break

    if best_state_seq is not None and best_state_het is not None:
        seq_ae.load_state_dict(best_state_seq)
        heteroencoder.load_state_dict(best_state_het)
    return seq_ae, heteroencoder, train_losses, val_losses