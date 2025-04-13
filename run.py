#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trajectory prediction using Transformer-GMM
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle

from utils.util import set_seeds
from utils.trajectory_dataset import TrajectoryDataset, get_data_loaders
from models.transformer_GMM import AttentionGMM

def create_predictor(past_trajectory, future_trajectory, device, normalize, checkpoint_file, win_size):
    """
    Initializes and returns the Transformer-GMM model
    
    Args:
        past_trajectory (int): Number of past timesteps used for prediction
        future_trajectory (int): Number of future timesteps to predict
        device (str): Device on which the model should run ('cuda' or 'cpu')
        normalize (bool): Whether to normalize input features
        checkpoint_file (str): Path to a checkpoint file if loading a pre-trained model
        win_size (int): Window size for the model

    Returns:
        AttentionGMM model instance
    """
    return AttentionGMM(saving_checkpoint_path=checkpoint_file,
                        past_trajectory=past_trajectory, 
                        future_trajectory=future_trajectory, 
                        device=device, 
                        normalize=normalize,
                        win_size=win_size)

def inspect_dataset(data_folder):
    """
    Performs a detailed inspection of the dataset structure
    """
    print("\n" + "="*50)
    print("DETAILED DATASET INSPECTION")
    print("="*50)
    
    # List all files in the data folder
    files = os.listdir(data_folder)
    print(f"Files in dataset folder: {files}")
    
    # Load the metadata if it exists
    if "metadata.pkl" in files:
        with open(os.path.join(data_folder, "metadata.pkl"), 'rb') as f:
            metadata = pickle.load(f)
        print(f"Metadata: {metadata}")
    
    # Check each split file
    for split in ['train.pkl', 'val.pkl', 'test.pkl']:
        if split in files:
            file_path = os.path.join(data_folder, split)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            print(f"{split}: {file_size:.2f} MB")
            
            # Load a few samples to inspect
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                
                print(f"  Number of samples: {len(data)}")
                if len(data) > 0:
                    sample = data[0]
                    print(f"  First sample structure: {type(sample)}, length: {len(sample)}")
                    print(f"  Observation shape: {np.array(sample[0]).shape}")
                    print(f"  Target shape: {np.array(sample[1]).shape}")
                    
                    # Check the data range
                    all_values = []
                    for i in range(min(5, len(data))):
                        obs, tgt = data[i]
                        all_values.extend(np.array(obs).flatten())
                        all_values.extend(np.array(tgt).flatten())
                    
                    all_values = np.array(all_values)
                    print(f"  Data range (first 5 samples): {np.min(all_values):.4f} to {np.max(all_values):.4f}")
            except Exception as e:
                print(f"  Error inspecting {split}: {str(e)}")
        # Create data loaders
    data_loaders = get_data_loaders(
            data_folder=args.data_folder,
            batch_size=args.batch_size,
            include_velocity=True,
            normalize=args.normalize,
            num_workers=args.num_workers,
            verbose=args.verbose,
            calculate_warmup = True
        )

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run Transformer-GMM predictor on Waymo trajectory dataset')
    p.add_argument('--data_folder', type=str, required=True, help='Path to the trajectory dataset')
    p.add_argument('--past_trajectory', type=int, default=15, help='Number of past timesteps')
    p.add_argument('--future_trajectory', type=int, default=30, help='Number of future timesteps to predict')
    p.add_argument('--window_size', type=int, default=1, help='Sliding window size')
    p.add_argument('--setting', type=str, default='insspect', choices=['train', 'evaluate', 'inspect'], help='Execution mode')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file')
    p.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    p.add_argument('--normalize', type=bool, default=False, help='Normalize data')
    p.add_argument('--batch_size', type=int, default=32, help='Batch size')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model')
    p.add_argument('--seed', type=int, default=42, help='Seed for reproducibility')
    p.add_argument('--save_path', type=str, default='checkpoints', help='Path to save model checkpoints')
    p.add_argument('--verbose', type=bool, default=True, help='Print detailed information')

    args = p.parse_args()
    
    # Set random seed for reproducibility
    set_seeds(args.seed)
    

    # Special mode for dataset inspection
    if args.setting == 'inspect':
        inspect_dataset(args.data_folder)
        exit(0)

    # Create data loaders
    data_loaders = get_data_loaders(
            data_folder=args.data_folder,
            batch_size=args.batch_size,
            include_velocity=True,
            normalize=args.normalize,
            num_workers=args.num_workers,
            verbose=args.verbose,
            calculate_warmup = True
        )
    

    train_loader = data_loaders['train']
    val_loader = data_loaders['val']
    test_loader = data_loaders['test']
    
    
    # Print model configuration
    print("\n" + "="*50)
    print("MODEL CONFIGURATION")
    print("="*50)
    print(f"Past trajectory steps: {args.past_trajectory}")
    print(f"Future trajectory steps: {args.future_trajectory}")
    print(f"Window size: {args.window_size}")
    print(f"Device: {args.device}")
    print(f"Normalize: {args.normalize}")
    print(f"Batch size: {args.batch_size}")
    if args.checkpoint:
        print(f"Loading from checkpoint: {args.checkpoint}")

    # Create predictor model
    predictor = create_predictor(
        past_trajectory=args.past_trajectory, 
        future_trajectory=args.future_trajectory, 
        device=args.device,
        normalize=args.normalize,
        checkpoint_file=args.checkpoint,
        win_size=args.window_size
    )
    
    # Train or evaluate the model
    if args.setting == 'train':
        # Create save directory if it doesn't exist
        os.makedirs(args.save_path, exist_ok=True)
        
        # Train the model
        predictor.train(train_loader, val_loader, save_path=args.save_path)
        
        # Evaluate on test set
        predictor.evaluate(test_loader)
        
    elif args.setting == 'evaluate':
        assert args.checkpoint is not None, "Checkpoint file is required for evaluation"
        
        # Evaluate the model
        metrics = predictor.evaluate(test_loader)
        print(f"Test metrics: {metrics}")