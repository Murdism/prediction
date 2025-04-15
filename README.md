# prediction
trajectory prediction



## Quick Train

- To train or evaluate model on a dataset use run.py 
```bash
python run.py --data_folder trajectory_dataset --setting train
```
- To train on your own by importing the model use the following code
```bash
from models.transformer_GMM import AttentionGMM

predictor = AttentionGMM(saving_checkpoint_path=checkpoint_file,
                        past_trajectory=past_trajectory, 
                        future_trajectory=future_trajectory, 
                        device=device, 
                        normalize=normalize,
                        win_size=win_size)

os.makedirs(args.save_path, exist_ok=True)
        
# Train the model
predictor.train(train_loader, val_loader, save_path=args.save_path)

# Evaluate on test set
predictor.evaluate(test_loader)
```
### TrajectoryDataset
Used to create the dataloaders for training and testing




### WaymoToTrajectoryDataset
Used to create a trajectory prediction dataset from waymo tf records and saves it as a pickle



### TrajectoryHandler

A lightweight GPU-based utility that stores recent (x, y) trajectories for detected objects, to be used by the predictor.

**Inputs:**
- `vision_msgs/Detection3DArray` messages
  - Each `Detection3D` must include:
    - `bbox.center.position.x` and `.y` — used as (x, y) coordinates.
    - At least one `ObjectHypothesisWithPose`, where `hypothesis.class_id` represents the object ID (as a stringified integer, e.g., `"1"`).

**Outputs:**
- Returns a `(num_objects, max_length, 2)` PyTorch tensor containing the recent (x, y) positions for the requested object IDs (converted to integers).

This module ensures temporal consistency for each tracked object, making it easier to provide sequential input to a trajectory prediction model.

