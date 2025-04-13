# utils/trajectory_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
import os
import math

class TrajectoryDataset(Dataset):
    def __init__(self, data_folder, setting, include_velocity=True, normalize=False, verbose=True):
        """
        Custom dataset for trajectory prediction from Waymo data
        
        Args:
            data_folder (str): Folder containing the dataset files
            setting (str): Dataset split (train, val, test)
            include_velocity (bool): Whether to include velocity features
            normalize (bool): Whether to normalize the data
            verbose (bool): Whether to print detailed information
        """
        self.include_velocity = include_velocity
        self.normalize = normalize
        self.verbose = verbose

        # Load data
        file = f"{data_folder}/{setting}.pkl"
        with open(file, 'rb') as f:
            self.data = pickle.load(f)
        
        # Get example shapes
        if len(self.data) > 0:
            first_sample = self.data[0]
            obs_shape = np.array(first_sample[0]).shape
            tgt_shape = np.array(first_sample[1]).shape
            
            print(f"Loaded {setting} dataset with {len(self.data)} samples")
            print(f"Observation shape: {obs_shape}, Target shape: {tgt_shape}")
            print(f"Sample data range: min={self._get_data_range(first_sample)[0]:.4f}, max={self._get_data_range(first_sample)[1]:.4f}")
            
            if self.verbose:
                # Check first few samples
                for i in range(min(3, len(self.data))):
                    obs, tgt = self.data[i]
                    print(f"Sample {i} - Obs: {np.array(obs).shape}, Target: {np.array(tgt).shape}")
        else:
            print(f"WARNING: Empty dataset for {setting}")
            
        # Calculate dataset statistics if normalizing
        self.mean, self.std = self._calculate_statistics()
        print(f"Dataset statistics - Mean: {self.mean}, Std: {self.std}")

    def _get_data_range(self, sample):
        """Get min and max values from a sample"""
        obs, tgt = sample
        all_values = np.concatenate([np.array(obs).flatten(), np.array(tgt).flatten()])
        return np.min(all_values), np.max(all_values)

    def _calculate_statistics(self):
        """
        Calculate mean and standard deviation of the dataset.
        If normalization is disabled, return zero mean and unit std with correct shape.
        """
        all_sequences = []
        
        for observation, target in self.data:
            obs_tensor = torch.tensor(observation, dtype=torch.float32)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            
            if self.include_velocity:
                obs_vel = obs_tensor[1:] - obs_tensor[:-1]
                obs_vel = torch.cat([obs_vel[[0]], obs_vel], dim=0)
                obs_tensor = torch.cat([obs_tensor, obs_vel], dim=1)
                
                first_target_vel = target_tensor[0:1] - obs_tensor[-1:, :2]
                rest_target_vel = target_tensor[1:] - target_tensor[:-1]
                target_vel = torch.cat([first_target_vel, rest_target_vel], dim=0)
                target_tensor = torch.cat([target_tensor, target_vel], dim=1)
            
            all_sequences.append(obs_tensor)
            all_sequences.append(target_tensor)
        
        all_data = torch.cat(all_sequences, dim=0)
        
        if self.verbose:
            print(f"All data shape for statistics: {all_data.shape}")
            print(f"Data range before normalization: {torch.min(all_data).item():.4f} to {torch.max(all_data).item():.4f}")
        
        if self.normalize:
            mean = torch.mean(all_data, dim=0)
            std = torch.std(all_data, dim=0)
            std[std < 1e-6] = 1.0
        else:
            mean = torch.zeros(all_data.shape[1], dtype=torch.float32)
            std = torch.ones(all_data.shape[1], dtype=torch.float32)
        
        return mean, std


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves the observation and target for a given index.
        """
        observation, target = self.data[idx]
        
        # Convert positions to tensors
        obs_tensor = torch.tensor(observation, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        if self.include_velocity:
            # Compute velocities for observation
            obs_vel = obs_tensor[1:] - obs_tensor[:-1]
            obs_vel = torch.cat([obs_vel[[0]], obs_vel], dim=0)
            obs_tensor = torch.cat([obs_tensor, obs_vel], dim=1)

            # Calculate velocities for target
            first_target_vel = target_tensor[0:1] - obs_tensor[-1:, :2]
            rest_target_vel = target_tensor[1:] - target_tensor[:-1]
            target_vel = torch.cat([first_target_vel, rest_target_vel], dim=0)
            target_tensor = torch.cat([target_tensor, target_vel], dim=1)
        
        # Normalize data if requested
        if self.normalize:
            feature_dim = 4 if self.include_velocity else 2
            obs_tensor = (obs_tensor - self.mean[:feature_dim]) / self.std[:feature_dim]
            target_tensor = (target_tensor - self.mean[:feature_dim]) / self.std[:feature_dim]
            
        return obs_tensor, target_tensor
    
    def denormalize(self, tensor):
        """
        Denormalize a tensor using the dataset statistics
        """
        if not self.normalize:
            return tensor
            
        feature_dim = 4 if self.include_velocity else 2
        return tensor * self.std[:feature_dim] + self.mean[:feature_dim]

    def analyze_velocity_statistics(self):
        """
        Analyze velocity data and return statistics
        
        Returns:
            dict: Statistics about velocity components
        """
        if not self.include_velocity:
            print("Velocity analysis skipped - velocity features not included")
            return None
            
        print("\nAnalyzing velocity statistics...")
        
        velocity_stats = {
            'vx': {'values': [], 'min': float('inf'), 'max': float('-inf')},
            'vy': {'values': [], 'min': float('inf'), 'max': float('-inf')}
        }
        
        # Use a small subset for efficiency if dataset is large
        sample_size = min(len(self), 1000)
        indices = np.random.choice(len(self), sample_size, replace=False)
        
        for idx in indices:
            obs_tensor, _ = self.__getitem__(idx)
            
            # Extract velocity columns (last 2)
            if obs_tensor.shape[1] >= 4:
                vx = obs_tensor[:, 2].cpu().numpy()  # velocity x
                vy = obs_tensor[:, 3].cpu().numpy()  # velocity y
                
                velocity_stats['vx']['values'].extend(vx)
                velocity_stats['vy']['values'].extend(vy)
                
                velocity_stats['vx']['min'] = min(velocity_stats['vx']['min'], np.min(vx))
                velocity_stats['vx']['max'] = max(velocity_stats['vx']['max'], np.max(vx))
                velocity_stats['vy']['min'] = min(velocity_stats['vy']['min'], np.min(vy))
                velocity_stats['vy']['max'] = max(velocity_stats['vy']['max'], np.max(vy))
        
        # Calculate statistics
        for component in ['vx', 'vy']:
            values = np.array(velocity_stats[component]['values'])
            velocity_stats[component]['mean'] = float(np.mean(values))
            velocity_stats[component]['std'] = float(np.std(values))
            velocity_stats[component]['median'] = float(np.median(values))
            velocity_stats[component]['abs_mean'] = float(np.mean(np.abs(values)))
        
        # Print results
        print(f"Velocity X: min={velocity_stats['vx']['min']:.4f}, max={velocity_stats['vx']['max']:.4f}, mean={velocity_stats['vx']['mean']:.4f}, std={velocity_stats['vx']['std']:.4f}")
        print(f"Velocity Y: min={velocity_stats['vy']['min']:.4f}, max={velocity_stats['vy']['max']:.4f}, mean={velocity_stats['vy']['mean']:.4f}, std={velocity_stats['vy']['std']:.4f}")
        
        return velocity_stats


def get_data_loaders(data_folder, batch_size=32, include_velocity=True, normalize=True, num_workers=4, verbose=True, calculate_warmup=True):
    """
    Create data loaders for training, validation and testing
    
    Args:
        data_folder (str): Folder containing the dataset files
        batch_size (int): Batch size for the data loaders
        include_velocity (bool): Whether to include velocity features
        normalize (bool): Whether to normalize the data
        num_workers (int): Number of workers for data loading
        verbose (bool): Whether to print detailed information
        calculate_warmup (bool): Whether to calculate warmup steps
        
    Returns:
        dict: Dictionary containing train, val, test data loaders and warmup information
    """
    print("\n" + "="*50)
    print(f"LOADING DATASETS FROM: {data_folder}")
    print("="*50)
    
    train_dataset = TrajectoryDataset(data_folder, 'train', include_velocity, normalize, verbose)
    
    # Try to load validation set, use test set if not available
    try:
        val_dataset = TrajectoryDataset(data_folder, 'val', include_velocity, normalize, verbose)
    except FileNotFoundError:
        print("Validation set not found, using test set for validation")
        val_dataset = TrajectoryDataset(data_folder, 'test', include_velocity, normalize, verbose)
    
    test_dataset = TrajectoryDataset(data_folder, 'test', include_velocity, normalize, verbose)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Get a batch from each loader to verify shapes
    print("\n" + "="*50)
    print("VERIFYING DATALOADER SHAPES")
    print("="*50)
    
    if verbose:
        for name, loader in [("Train", train_loader), ("Val", val_loader), ("Test", test_loader)]:
            try:
                sample_batch = next(iter(loader))
                obs_batch, target_batch = sample_batch
                print(f"{name} loader: {len(loader)} batches, batch shapes: Obs {obs_batch.shape}, Target {target_batch.shape}")
                
                # Report on velocity channels if included
                if include_velocity:
                    print(f"  Position channels range: {torch.min(obs_batch[:,:,:2]).item():.4f} to {torch.max(obs_batch[:,:,:2]).item():.4f}")
                    print(f"  Velocity channels range: {torch.min(obs_batch[:,:,2:]).item():.4f} to {torch.max(obs_batch[:,:,2:]).item():.4f}")
            except:
                print(f"Could not get sample batch from {name} loader")
    
    # Calculate warmup steps if requested
    warmup_info = None
    if calculate_warmup:
        warmup_info = calculate_warmup_steps(train_loader)
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'warmup_info': warmup_info
    }

def calculate_warmup_steps(train_loader, total_epochs=100):
    """
    Calculate recommended warmup steps for transformer training
    
    Args:
        train_loader: Training data loader
        total_epochs: Total number of training epochs
    
    Returns:
        dict: Warmup step recommendations
    """
    print("\n" + "="*50)
    print("CALCULATING WARMUP STEPS")
    print("="*50)
    
    # Count steps per epoch
    steps_per_epoch = len(train_loader)
    total_training_steps = steps_per_epoch * total_epochs
    
    # Calculate recommended warmup steps using different methods
    warmup_10_percent = int(0.1 * total_training_steps)
    warmup_3_epochs = int(3 * steps_per_epoch)
    warmup_sqrt = int(math.sqrt(total_training_steps))
    
    # Select recommended method based on dataset size
    if total_training_steps < 5000:
        recommended = warmup_10_percent
        method = "10% of total steps (small dataset)"
    elif total_training_steps > 50000:
        recommended = min(warmup_3_epochs, warmup_sqrt)
        method = "min(3 epochs, sqrt) (large dataset)"
    else:
        recommended = warmup_3_epochs
        method = "3 epochs (medium dataset)"
    
    warmup_info = {
        'steps_per_epoch': steps_per_epoch,
        'total_training_steps': total_training_steps,
        'warmup_10_percent': warmup_10_percent,
        'warmup_3_epochs': warmup_3_epochs,
        'warmup_sqrt': warmup_sqrt,
        'recommended_warmup_steps': recommended,
        'recommendation_method': method
    }
    
    # Print results
    print(f"Dataset size info:")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total training steps ({total_epochs} epochs): {total_training_steps}")
    
    print("\nWarmup step calculations:")
    print(f"  10% of total steps: {warmup_10_percent}")
    print(f"  3 epochs: {warmup_3_epochs}")
    print(f"  Square root of total steps: {warmup_sqrt}")
    
    print(f"\nRecommended warmup steps: {recommended} ({method})")
    
    # Learning rate schedule recommendation
    print("\nSuggested learning rate schedule:")
    print(f"  - Start with learning rate: 1e-4")
    print(f"  - Linear warmup for {recommended} steps")
    print(f"  - First decay at step {int(0.6 * total_training_steps)} (60% of training)")
    print(f"  - Second decay at step {int(0.8 * total_training_steps)} (80% of training)")
    
    return warmup_info

