import tensorflow as tf
from waymo_open_dataset.protos import scenario_pb2
import os
import glob
from  tqdm import tqdm

def parse_tfrecord_folder_with_tf(tfrecord_folder, file_pattern="*.tfrecord"):
    """Read all .tfrecord files in a folder and extract vehicle trajectories using TensorFlow."""
    
    all_vehicles = []
    file_count = 0
    object_types_count = {}
    
    # Find all TFRecord files in the folder
    tfrecord_files = glob.glob(os.path.join(tfrecord_folder, file_pattern))
    print(tfrecord_files)
    
    if not tfrecord_files:
        print(f"No files matching pattern '{file_pattern}' found in {tfrecord_folder}")
        return all_vehicles, object_types_count
    
    # Process each file in the folder
    for tfrecord_path in tfrecord_files:
        file_count += 1
        print(f"\nProcessing file {file_count}/{len(tfrecord_files)}: {os.path.basename(tfrecord_path)}")
        
        # Create a dataset from the TFRecord file
        dataset = tf.data.TFRecordDataset(tfrecord_path, compression_type="")
        
        file_vehicles = []
        file_object_types = {}
        
        # Iterate through the records
        for data in tqdm(dataset):
            scenario = scenario_pb2.Scenario()
            scenario.ParseFromString(data.numpy())
            
            # Print scenario metadata
            # print(f"Scenario ID: {scenario.scenario_id}")
            # print(f"Scenario timestamp_micros: {scenario.timestamps_seconds}")
            
            # Iterate through the tracks (agents) in the scenario
            for agent in scenario.tracks:
                # Count object types
                object_type = agent.object_type
                object_type_name = scenario_pb2.Track.ObjectType.Name(object_type)
                
                if object_type_name not in file_object_types:
                    file_object_types[object_type_name] = 0
                file_object_types[object_type_name] += 1
                
                if object_type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
                    trajectory = []
                    # Collect the trajectory of each vehicle
                    for state in agent.states:
                        # Check if the state has a valid pose
                        if state.valid:
                            trajectory.append([state.center_x, state.center_y, state.heading])
                    
                    if trajectory:  # Only add if we have valid trajectory points
                        file_vehicles.append(trajectory)
        
        print(f"Object types in this file: {file_object_types}")
        print(f"Found {len(file_vehicles)} vehicles in {os.path.basename(tfrecord_path)}")
        
        # Update global counts
        for obj_type, count in file_object_types.items():
            if obj_type not in object_types_count:
                object_types_count[obj_type] = 0
            object_types_count[obj_type] += count
        
        all_vehicles.extend(file_vehicles)
    
    return all_vehicles, object_types_count

# Alternative version if you want to avoid TensorFlow dependency
def parse_tfrecord_folder_without_tf(tfrecord_folder, file_pattern="*.tfrecord"):
    """Read all .tfrecord files in a folder and extract vehicle trajectories without using TensorFlow."""
    
    import struct
    
    all_vehicles = []
    file_count = 0
    object_types_count = {}
    
    # Find all TFRecord files in the folder
    tfrecord_files = glob.glob(os.path.join(tfrecord_folder, file_pattern))
    
    if not tfrecord_files:
        print(f"No files matching pattern '{file_pattern}' found in {tfrecord_folder}")
        return all_vehicles, object_types_count
    
    # Process each file in the folder
    for tfrecord_path in tfrecord_files:
        file_count += 1
        print(f"Processing file {file_count}/{len(tfrecord_files)}: {os.path.basename(tfrecord_path)}")
        
        file_vehicles = []
        file_object_types = {}
        
        with open(tfrecord_path, 'rb') as f:
            while True:
                # Read the header
                header_str = f.read(8)
                if not header_str:
                    break  # End of file
                
                # Parse the header to get the length of the data
                header = struct.unpack('Q', header_str)[0]
                
                # Read the data
                data_str = f.read(header)
                
                # Read the footer (CRC)
                f.read(4)
                
                # Parse the data
                scenario = scenario_pb2.Scenario()
                scenario.ParseFromString(data_str)
                
                # Print scenario metadata
                print(f"Scenario ID: {scenario.scenario_id}")
                print(f"Scenario timestamp_micros: {scenario.timestamp_micros}")
                
                # Iterate through the tracks (agents) in the scenario
                for agent in scenario.tracks:
                    # Count object types
                    object_type = agent.object_type
                    object_type_name = scenario_pb2.Track.ObjectType.Name(object_type)
                    
                    if object_type_name not in file_object_types:
                        file_object_types[object_type_name] = 0
                    file_object_types[object_type_name] += 1
                    
                    if agent.object_type == scenario_pb2.Track.ObjectType.TYPE_VEHICLE:
                        trajectory = []
                        # Collect the trajectory of each vehicle
                        for state in agent.states:
                            if state.valid:
                                trajectory.append([state.center_x, state.center_y, state.heading])
                        
                        if trajectory:
                            file_vehicles.append(trajectory)
        
        print(f"Object types in this file: {file_object_types}")
        print(f"Found {len(file_vehicles)} vehicles in {os.path.basename(tfrecord_path)}")
        
        # Update global counts
        for obj_type, count in file_object_types.items():
            if obj_type not in object_types_count:
                object_types_count[obj_type] = 0
            object_types_count[obj_type] += count
            
        all_vehicles.extend(file_vehicles)
    
    return all_vehicles, object_types_count

# Example usage:
tfrecord_folder = "waymo_scenarios/"  # Path to folder containing TFRecord files
file_pattern = "*.tfrecord*"  # Pattern to match TFRecord files

# Choose one of the methods:
# Method 1: Using TensorFlow (recommended for Waymo dataset)
vehicles_trajectories, object_types_count = parse_tfrecord_folder_with_tf(tfrecord_folder, file_pattern)

# Method 2: Without TensorFlow
# vehicles_trajectories, object_types_count = parse_tfrecord_folder_without_tf(tfrecord_folder, file_pattern)

# Print summary of object types
print("\nSummary of object types found:")
for obj_type, count in object_types_count.items():
    print(f"  - {obj_type}: {count}")

# Print summary of vehicles
print(f"\nTotal number of vehicles found: {len(vehicles_trajectories)}")

# Print sample of the first vehicle trajectory for inspection
if vehicles_trajectories:
    print(f"\nFirst vehicle trajectory (x, y, heading) - first 5 points:")
    for i, point in enumerate(vehicles_trajectories[0][:5]):
        print(f"Point {i}: {point}")
else:
    print("No vehicles found in the TFRecord files")

# Print information about the scenario_pb2.Track.ObjectType enum
print("\nAvailable object types in Waymo dataset:")
for name, value in scenario_pb2.Track.ObjectType.items():
    print(f"  - {name}: {value}")

# Print other information from the Track class to understand its structure
track_field_info = [field.name for field in scenario_pb2.Track.DESCRIPTOR.fields]
print("\nFields available in Track class:")
for field in track_field_info:
    print(f"  - {field}")

# Print state field information
state_field_info = [field.name for field in scenario_pb2.ObjectState.DESCRIPTOR.fields]
print("\nFields available in ObjectState class:")
for field in state_field_info:
    print(f"  - {field}")