import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import os
import glob
import pickle
import numpy as np
from tqdm import tqdm

class WaymoToTrajectoryDataset:
    def __init__(self, observation_steps=20, prediction_steps=30, window_stride=1, sample_rate=10):
        """
        Convert Waymo Open Dataset to a trajectory prediction dataset format
        
        Args:
            observation_steps (int): Number of timesteps for observation sequence
            prediction_steps (int): Number of timesteps for prediction sequence
            window_stride (int): Stride size for sliding window (default: 1)
            sample_rate (int): Sampling rate in Hz (default: 10, original data rate)
                               Can be set to lower values (e.g., 5) to downsample
        """
        self.observation_steps = observation_steps
        self.prediction_steps = prediction_steps
        self.window_stride = window_stride
        self.sample_rate = sample_rate
        self.sample_interval = None
        
        # If sample_rate is less than 10Hz (original data rate), calculate sample interval
        if sample_rate < 10:
            self.sample_interval = int(10 / sample_rate)
            print(f"Downsampling from 10Hz to {sample_rate}Hz (keeping every {self.sample_interval}th point)")
        else:
            print(f"Using original sampling rate (10Hz)")
            self.sample_interval = 1
        
    # def extract_trajectories(self, tfrecord_folder, file_pattern="*.tfrecord", 
    #                          max_files=None, min_trajectory_len=None):
    #     """
    #     Extract vehicle trajectories from Waymo TFRecord files
        
    #     Args:
    #         tfrecord_folder (str): Path to folder containing TFRecord files
    #         file_pattern (str): Pattern to match TFRecord files
    #         max_files (int): Maximum number of files to process, None for all
    #         min_trajectory_len (int): Minimum trajectory length to include
            
    #     Returns:
    #         list: Processed trajectory data in [observation, target] format with time information
    #     """
    #     # Find all TFRecord files in the folder
    #     tfrecord_files = glob.glob(os.path.join(tfrecord_folder, file_pattern))
        
    #     if not tfrecord_files:
    #         print(f"No files matching pattern '{file_pattern}' found in {tfrecord_folder}")
    #         return []
        
    #     # Limit files if needed
    #     if max_files is not None:
    #         tfrecord_files = tfrecord_files[:max_files]
            
    #     print(f"Processing {len(tfrecord_files)} files from {tfrecord_folder}")
    #     print(f"Using window stride: {self.window_stride}")
        
    #     # Initialize dataset
    #     dataset = []
    #     stats = {
    #         "total_scenarios": 0,
    #         "total_vehicles": 0,
    #         "valid_trajectories": 0,
    #         "object_types": {},
    #         "avg_time_step": 0,
    #         "time_step_samples": 0
    #     }
        
    #     # Process each file
    #     for file_idx, tfrecord_path in enumerate(tfrecord_files):
    #         print(f"\nProcessing file {file_idx+1}/{len(tfrecord_files)}: {os.path.basename(tfrecord_path)}")
            
    #         try:
    #             # Create a dataset from the TFRecord file
    #             tf_dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
                
    #             # Iterate through the records (scenarios)
    #             for data in tqdm(tf_dataset):
    #                 scenario = scenario_pb2.Scenario()
    #                 scenario.ParseFromString(data.numpy())
                    
    #                 stats["total_scenarios"] += 1
                    
    #                 # Skip if no timestamps
    #                 if not scenario.timestamps_seconds:
    #                     continue
                        
    #                 # Get timestamps for this scenario
    #                 timestamps = list(scenario.timestamps_seconds)
                    
    #                 # Calculate average time step for statistics
    #                 if len(timestamps) > 1:
    #                     time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    #                     avg_time_diff = sum(time_diffs) / len(time_diffs)
    #                     stats["avg_time_step"] += sum(time_diffs)
    #                     stats["time_step_samples"] += len(time_diffs)
                        
    #                 # Extract vehicle trajectories
    #                 for track in scenario.tracks:
    #                     # Track object type
    #                     object_type = track.object_type
    #                     object_type_name = scenario_pb2.Track.ObjectType.Name(object_type)
                        
    #                     # Count object types
    #                     if object_type_name not in stats["object_types"]:
    #                         stats["object_types"][object_type_name] = 0
    #                     stats["object_types"][object_type_name] += 1
                        
    #                     # Only process vehicles
    #                     if object_type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
    #                         stats["total_vehicles"] += 1
                            
    #                         # Extract trajectory with timestamps
    #                         full_trajectory = []
    #                         for i, state in enumerate(track.states):
    #                             if state.valid and i < len(timestamps):
    #                                 # Store [x, y, timestamp]
    #                                 full_trajectory.append([state.center_x, state.center_y])
                            
    #                         # Apply downsampling if needed
    #                         if self.sample_interval > 1:
    #                             trajectory = full_trajectory[::self.sample_interval]
    #                         else:
    #                             trajectory = full_trajectory
                            
    #                         # Skip if trajectory is too short
    #                         total_steps = self.observation_steps + self.prediction_steps
    #                         if min_trajectory_len is not None and len(trajectory) < min_trajectory_len:
    #                             continue
    #                         if len(trajectory) < total_steps:
    #                             continue
                                
    #                         # Create observation/target pairs with sliding window
    #                         for i in range(0, len(trajectory) - total_steps + 1, self.window_stride):
    #                             observation = trajectory[i:i+self.observation_steps]
    #                             target = trajectory[i+self.observation_steps:i+total_steps]
                                
    #                             # Add to dataset
    #                             dataset.append([observation, target])
    #                             stats["valid_trajectories"] += 1
                
    #         except Exception as e:
    #             print(f"Error processing file {tfrecord_path}: {str(e)}")
        
    #     # Calculate average time step
    #     if stats["time_step_samples"] > 0:
    #         stats["avg_time_step"] /= stats["time_step_samples"]
        
    #     print(f"\nDataset Statistics:")
    #     print(f"Total scenarios processed: {stats['total_scenarios']}")
    #     print(f"Total vehicles found: {stats['total_vehicles']}")
    #     print(f"Valid trajectories extracted: {stats['valid_trajectories']}")
    #     print(f"Average time step between data points: {stats['avg_time_step']:.4f} seconds")
    #     print(f"Object types:")
    #     for obj_type, count in stats["object_types"].items():
    #         print(f"  - {obj_type}: {count}")
            
    #     return dataset

    def extract_trajectories(self, tfrecord_folder, file_pattern="*.tfrecord", 
                            max_files=None, min_trajectory_len=None):
        """
        Extract vehicle trajectories from Waymo TFRecord files
        
        Args:
            tfrecord_folder (str): Path to folder containing TFRecord files
            file_pattern (str): Pattern to match TFRecord files
            max_files (int): Maximum number of files to process, None for all
            min_trajectory_len (int): Minimum trajectory length to include
            
        Returns:
            list: Processed trajectory data in [observation, target] format
            dict: Time step statistics
        """
        # Find all TFRecord files in the folder
        tfrecord_files = glob.glob(os.path.join(tfrecord_folder, file_pattern))
        
        if not tfrecord_files:
            print(f"No files matching pattern '{file_pattern}' found in {tfrecord_folder}")
            return [], {}
        
        # Limit files if needed
        if max_files is not None:
            tfrecord_files = tfrecord_files[:max_files]
            
        print(f"Processing {len(tfrecord_files)} files from {tfrecord_folder}")
        print(f"Using window stride: {self.window_stride}")
        
        # Initialize dataset and time differences collection
        dataset = []
        time_diffs = []
        
        stats = {
            "total_scenarios": 0,
            "total_vehicles": 0,
            "valid_trajectories": 0,
            "object_types": {},
            "avg_time_step": 0,
            "time_step_samples": 0
        }
        
        # Process each file
        for file_idx, tfrecord_path in enumerate(tfrecord_files):
            print(f"\nProcessing file {file_idx+1}/{len(tfrecord_files)}: {os.path.basename(tfrecord_path)}")
            
            try:
                # Create a dataset from the TFRecord file
                tf_dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
                
                # Iterate through the records (scenarios)
                for data in tqdm(tf_dataset):
                    scenario = scenario_pb2.Scenario()
                    scenario.ParseFromString(data.numpy())
                    
                    stats["total_scenarios"] += 1
                    
                    # Skip if no timestamps
                    if not scenario.timestamps_seconds:
                        continue
                        
                    # Get timestamps for this scenario
                    timestamps = list(scenario.timestamps_seconds)
                    
                    # Calculate average time step for statistics
                    if len(timestamps) > 1:
                        scenario_time_diffs = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
                        time_diffs.extend(scenario_time_diffs)
                        avg_time_diff = sum(scenario_time_diffs) / len(scenario_time_diffs)
                        stats["avg_time_step"] += sum(scenario_time_diffs)
                        stats["time_step_samples"] += len(scenario_time_diffs)
                        
                    # Extract vehicle trajectories
                    for track in scenario.tracks:
                        # Track object type
                        object_type = track.object_type
                        object_type_name = scenario_pb2.Track.ObjectType.Name(object_type)
                        
                        # Count object types
                        if object_type_name not in stats["object_types"]:
                            stats["object_types"][object_type_name] = 0
                        stats["object_types"][object_type_name] += 1
                        
                        # Only process vehicles
                        if object_type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
                            stats["total_vehicles"] += 1
                            
                            # Extract trajectory coordinates and timestamps separately
                            full_trajectory = []
                            full_timestamps = []
                            
                            for i, state in enumerate(track.states):
                                if state.valid and i < len(timestamps):
                                    # Store [x, y] for trajectory
                                    full_trajectory.append([state.center_x, state.center_y])
                                    # Store timestamp separately
                                    full_timestamps.append(timestamps[i])
                            
                            # Apply downsampling if needed
                            if self.sample_interval > 1:
                                trajectory = full_trajectory[::self.sample_interval]
                                sampled_timestamps = full_timestamps[::self.sample_interval]
                            else:
                                trajectory = full_trajectory
                                sampled_timestamps = full_timestamps
                            
                            # Skip if trajectory is too short
                            total_steps = self.observation_steps + self.prediction_steps
                            if min_trajectory_len is not None and len(trajectory) < min_trajectory_len:
                                continue
                            if len(trajectory) < total_steps:
                                continue
                                
                            # Create observation/target pairs with sliding window
                            for i in range(0, len(trajectory) - total_steps + 1, self.window_stride):
                                observation = trajectory[i:i+self.observation_steps]
                                target = trajectory[i+self.observation_steps:i+total_steps]
                                
                                # Calculate time differences for this window
                                window_timestamps = sampled_timestamps[i:i+total_steps]
                                for j in range(len(window_timestamps)-1):
                                    time_diffs.append(window_timestamps[j+1] - window_timestamps[j])
                                
                                # Add to dataset
                                dataset.append([observation, target])
                                stats["valid_trajectories"] += 1
                
            except Exception as e:
                print(f"Error processing file {tfrecord_path}: {str(e)}")
        
        # Calculate average time step
        if stats["time_step_samples"] > 0:
            stats["avg_time_step"] /= stats["time_step_samples"]
        
        # Calculate time statistics
        time_stats = {}
        if time_diffs:
            time_stats = {
                "min": min(time_diffs),
                "max": max(time_diffs),
                "mean": sum(time_diffs) / len(time_diffs),
                "median": sorted(time_diffs)[len(time_diffs)//2],
                "count": len(time_diffs)
            }
        else:
            time_stats = {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "count": 0
            }
        
        print(f"\nDataset Statistics:")
        print(f"Total scenarios processed: {stats['total_scenarios']}")
        print(f"Total vehicles found: {stats['total_vehicles']}")
        print(f"Valid trajectories extracted: {stats['valid_trajectories']}")
        print(f"Time step statistics:")
        if time_stats["count"] > 0:
            print(f"  Min: {time_stats['min']:.4f} sec")
            print(f"  Max: {time_stats['max']:.4f} sec")
            print(f"  Mean: {time_stats['mean']:.4f} sec")
            print(f"  Median: {time_stats['median']:.4f} sec")
            print(f"  Samples: {time_stats['count']}")
        else:
            print("  No valid time step information available")
        print(f"Object types:")
        for obj_type, count in stats["object_types"].items():
            print(f"  - {obj_type}: {count}")
            
        # Store time statistics in self for later use
        self.time_stats = time_stats
        
        return dataset    
    def compute_time_steps(self, dataset):
        """
        Return the time statistics computed during trajectory extraction
        
        Args:
            dataset (list): Dataset (not used, kept for API compatibility)
            
        Returns:
            dict: Dictionary with time step statistics
        """
        # Return the pre-computed time statistics
        if hasattr(self, 'time_stats'):
            return self.time_stats
        else:
            return {
                "min": None,
                "max": None,
                "mean": None,
                "median": None,
                "count": 0,
                "note": "Time statistics not computed during extraction phase"
            }
    
    def split_dataset(self, dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, shuffle=True):
        """
        Split dataset into train, validation and test sets
        
        Args:
            dataset (list): Dataset to split
            train_ratio (float): Ratio of training data
            val_ratio (float): Ratio of validation data
            test_ratio (float): Ratio of test data
            shuffle (bool): Whether to shuffle the dataset before splitting
            
        Returns:
            dict: Dictionary containing train, val, test splits
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
        
        # Copy dataset to avoid modifying the original
        dataset_copy = dataset.copy()
        
        # Shuffle if needed
        if shuffle:
            np.random.shuffle(dataset_copy)
            
        # Calculate split indices
        n = len(dataset_copy)
        train_idx = int(n * train_ratio)
        val_idx = train_idx + int(n * val_ratio)
        
        # Split the dataset
        train_data = dataset_copy[:train_idx]
        val_data = dataset_copy[train_idx:val_idx]
        test_data = dataset_copy[val_idx:]
        
        # Compute time step statistics for each split
        time_stats = {
            "all": self.compute_time_steps(dataset_copy),
            "train": self.compute_time_steps(train_data),
            "val": self.compute_time_steps(val_data),
            "test": self.compute_time_steps(test_data)
        }
        
        print("\nTime step statistics (in seconds):")
        for split_name, stats in time_stats.items():
            if stats["count"] > 0:
                print(f"  {split_name.capitalize()} split: min={stats['min']:.4f}, max={stats['max']:.4f}, "
                      f"mean={stats['mean']:.4f}, median={stats['median']:.4f}, count={stats['count']}")
            else:
                print(f"  {split_name.capitalize()} split: No valid time differences found")
        
        return {
            "train": train_data,
            "val": val_data,
            "test": test_data
        }



    def save_dataset(self, dataset_splits, output_folder):
        """
        Save dataset splits to pickle files
        
        Args:
            dataset_splits (dict): Dictionary containing train, val, test splits
            output_folder (str): Output folder to save the dataset
        """
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Save each split
        for split_name, split_data in dataset_splits.items():
            output_file = os.path.join(output_folder, f"{split_name}.pkl")
            
            with open(output_file, 'wb') as f:
                pickle.dump(split_data, f)
                
            print(f"Saved {split_name} dataset with {len(split_data)} samples to {output_file}")
            
        # Save metadata about the dataset
        metadata = {
            "observation_steps": self.observation_steps,
            "prediction_steps": self.prediction_steps,
            "window_stride": self.window_stride,
            "sample_rate": self.sample_rate,
            "sample_interval": self.sample_interval,
            "original_sample_rate": 10,  # Waymo dataset original rate
            "format": "Each data point is [observation, target] where observation and target are lists of [x, y] values",
            "time_statistics": self.time_stats if hasattr(self, 'time_stats') else None
        }
        
        metadata_file = os.path.join(output_folder, "metadata.pkl")
        with open(metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
            
        print(f"Saved dataset metadata to {metadata_file}")

def main():
    # Parameters
    tfrecord_folder = "waymo_scenarios/"  # Path to folder containing TFRecord files
    file_pattern = "*.tfrecord*"  # Pattern to match TFRecord files
    output_folder = "trajectory_dataset"  # Output folder for processed dataset
    observation_steps = 15  # Number of timesteps for observation
    prediction_steps = 30  # Number of timesteps for prediction
    window_stride = 20  # Stride for sliding window (e.g., 5 means take every 5th window)
    max_files = 50  # Process all files
    min_trajectory_len = 45  # Minimum trajectory length to include
    
    # Create converter
    converter = WaymoToTrajectoryDataset(
        observation_steps=observation_steps,
        prediction_steps=prediction_steps,
        window_stride=window_stride
    )
    
    # Extract trajectories
    dataset = converter.extract_trajectories(
        tfrecord_folder=tfrecord_folder,
        file_pattern=file_pattern,
        max_files=max_files,
        min_trajectory_len=min_trajectory_len
    )
    
    # Split dataset
    dataset_splits = converter.split_dataset(dataset)
    
    # Save dataset
    converter.save_dataset(dataset_splits, output_folder)
    
    print(f"Dataset processing complete!")
    print(f"Total trajectories: {len(dataset)}")
    print(f"Train samples: {len(dataset_splits['train'])}")
    print(f"Validation samples: {len(dataset_splits['val'])}")
    print(f"Test samples: {len(dataset_splits['test'])}")

if __name__ == "__main__":
    main()