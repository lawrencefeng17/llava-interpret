import torch
from torch import nn, optim
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
import pickle
import numpy as np
import os
from pathlib import Path

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.mps.is_available() else "cpu")

hidden_activatons_folder = "/raid/lawrence/hidden_states"
class HiddenStatesDataset(Dataset):
    """Dataset class for loading hidden states from .npz files"""
    def __init__(self, data_dir, batch_tokens=32):
        """
        Args:
            data_dir (str): Directory containing .npz files
            batch_tokens (int): Number of tokens to include in each batch
        """
        self.data_dir = Path(data_dir)
        self.npz_files = sorted(list(self.data_dir.glob("*.npz")))
        self.batch_tokens = batch_tokens
        
        # Load first file to get dimensions
        sample = np.load(self.npz_files[0])
        first_key = list(sample.keys())[0]
        self.hidden_states_shape = sample[first_key].shape
        
        # Calculate total number of possible batches
        self.samples_per_file = self.hidden_states_shape[0]
        self.num_tokens = self.hidden_states_shape[1]
        self.hidden_dim = self.hidden_states_shape[2]
        self.total_batches = len(self.npz_files) * self.samples_per_file * (self.num_tokens // batch_tokens)

    def __len__(self):
        return self.total_batches

    def __getitem__(self, idx):
        """Load and return a batch of token hidden states"""
        # Calculate which file, sample, and token range to load
        file_idx = idx // (self.samples_per_file * (self.num_tokens // self.batch_tokens))
        remaining_idx = idx % (self.samples_per_file * (self.num_tokens // self.batch_tokens))
        sample_idx = remaining_idx // (self.num_tokens // self.batch_tokens)
        token_start = (remaining_idx % (self.num_tokens // self.batch_tokens)) * self.batch_tokens
        
        # Load the specific file and extract the batch
        npz_path = self.npz_files[file_idx]
        with np.load(npz_path) as data:
            first_key = list(data.keys())[0]
            hidden_states = data[first_key][sample_idx, token_start:token_start + self.batch_tokens, :]
            
        # Convert to torch tensor and ensure correct shape
        hidden_states = torch.from_numpy(hidden_states).float()
        # Reshape to (batch_size, hidden_dim) where batch_size = batch_tokens
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        
        return hidden_states

def create_dataloader(data_dir, batch_tokens=32, num_workers=4, batch_size=128):
    """
    Create a DataLoader for the hidden states dataset
    
    Args:
        data_dir (str): Directory containing .npz files
        batch_tokens (int): Number of tokens to include in each batch
        num_workers (int): Number of worker processes for data loading
        batch_size (int): Number of batches to combine
    
    Returns:
        DataLoader: PyTorch DataLoader object
    """
    dataset = HiddenStatesDataset(data_dir, batch_tokens)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader, dataset.hidden_dim
  
class SparseAutoencoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Linear(embedding_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, embedding_dim)
        
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.xavier_normal_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def forward(self, x):
        encoded = self.encoder(x).relu()
        decoded = self.decoder(encoded)
        return encoded, decoded

    def loss(self, x):
        encoded, decoded = self(x)
        reconstruction_loss = nn.functional.mse_loss(x, decoded)
        # Fix: Use matrix multiplication between encoded activations and decoder weights norm
        decoder_norms = torch.linalg.norm(self.decoder.weight, dim=0)
        activation_penalty = (encoded * decoder_norms).mean()
        return reconstruction_loss, activation_penalty

def train_sparse_autoencoder(
    model,
    optimizer,
    dataloader,
    num_epochs,
    device,
    penalty_coeff,
    log_interval=100,
    use_wandb=True
):
    """
    Train the sparse autoencoder
    
    Args:
        model: SparseAutoencoder instance
        optimizer: PyTorch optimizer
        dataloader: DataLoader for hidden states
        num_epochs: Number of training epochs
        device: torch device
        penalty_coeff: Coefficient for activation penalty
        log_interval: How often to log metrics
        use_wandb: Whether to log metrics to wandb
    """
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        epoch_recon_loss = 0
        epoch_penalty_loss = 0
        epoch_total_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, activations in enumerate(progress_bar):
            activations = activations.to(device)
            
            # Calculate losses
            reconstruction_loss, activation_penalty = model.loss(activations)
            total_loss = reconstruction_loss + penalty_coeff * activation_penalty
            
            # Check for NaN/infinite loss
            if not torch.isfinite(total_loss):
                raise ValueError(f"Loss is not finite! Reconstruction: {reconstruction_loss.item()}, "
                               f"Penalty: {activation_penalty.item()}")
            
            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            # Optional: Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update metrics
            epoch_recon_loss += reconstruction_loss.item()
            epoch_penalty_loss += activation_penalty.item()
            epoch_total_loss += total_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'recon_loss': reconstruction_loss.item(),
                'penalty': (penalty_coeff * activation_penalty).item()
            })
            
            # Log metrics
            if use_wandb and batch_idx % log_interval == 0:
                wandb.log({
                    'reconstruction_loss': reconstruction_loss.item(),
                    'activation_penalty': activation_penalty.item(),
                    'total_loss': total_loss.item(),
                    'epoch': epoch,
                    'global_step': global_step
                })
            
            global_step += 1
        
        # Compute epoch averages
        num_batches = len(dataloader)
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_penalty_loss = epoch_penalty_loss / num_batches
        avg_total_loss = epoch_total_loss / num_batches
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Reconstruction Loss: {avg_recon_loss:.6f}")
        print(f"Average Activation Penalty: {avg_penalty_loss:.6f}")
        print(f"Average Total Loss: {avg_total_loss:.6f}")
        
        if use_wandb:
            wandb.log({
                'epoch_avg_reconstruction_loss': avg_recon_loss,
                'epoch_avg_activation_penalty': avg_penalty_loss,
                'epoch_avg_total_loss': avg_total_loss,
                'epoch': epoch
            })
    
    return model

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder on hidden states")
    parser.add_argument("--key", type=str, help="Wandb API key")
    args = parser.parse_args()
    
    wandb.login(key=args.key)
    wandb.login()
    
    # Initialize wandb
    wandb.init(
        project="sparse-autoencoder",
        config={
            "embedding_dim": 4096,
            "expansion_factor": 8,
            "hidden_dim": 4096 * 8,
            "penalty_coeff": 0.05,
            "learning_rate": 1e-4,
            "batch_tokens": 32,
            "batch_size": 128,
            "num_epochs": 10
        }
    )
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader, embedding_dim = create_dataloader(
        data_dir="/raid/lawrence/hidden_states/",
        batch_tokens=32,
        num_workers=4,
        batch_size=128
    )
    
    # Create model
    expansion_factor = 8
    hidden_dim = embedding_dim * expansion_factor
    model = SparseAutoencoder(embedding_dim, hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Train
    model = train_sparse_autoencoder(
        model=model,
        optimizer=optimizer,
        dataloader=dataloader,
        num_epochs=10,
        device=device,
        penalty_coeff=0.05
    )
    
    wandb.finish()

