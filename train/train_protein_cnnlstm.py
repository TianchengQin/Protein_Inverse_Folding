import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from models.protein_cnnlstm import ProteinCNNLSTM


def train_protein_cnnlstm(model, train_loader, val_loader, num_epochs=10000, lr=1e-3, weight_decay=1e-4, grad_clip=1.0,
                          patience=500):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)
    best_val_loss = float("inf")
    early_stop_counter = 0
    best_model_state = None
    train_losses = []
    val_losses = []

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            dist_input = batch["dist_input"]
            seq_target = batch["seq_target"]

            logits = model(dist_input)

            loss = criterion(logits.view(-1, logits.size(-1)), seq_target.view(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_loss += loss.item() * dist_input.size(0)
        avg_train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                dist_input = batch["dist_input"]
                seq_target = batch["seq_target"]
                logits = model(dist_input)
                loss = criterion(logits.view(-1, logits.size(-1)), seq_target.view(-1))
                val_loss += loss.item() * dist_input.size(0)
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch}: LR = {current_lr:.2e}, Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_model_state,
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
            }, 'best_model_cnnlstm.pth')
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    print(f"Model 3 Training complete. Best Val Loss: {best_val_loss:.6f}")
    return model, train_losses, val_losses
