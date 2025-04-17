

import os
import argparse
from models.transformer_GMM import AttentionGMM
import torch


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run Transformer-GMM predictor on Waymo trajectory dataset')
    p.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint file')
    p.add_argument('--num_workers', type=int, default=4, help='Number of workers for dataloader')
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the model')


    args = p.parse_args()


    predictor = AttentionGMM(saving_checkpoint_path=args.checkpoint,device=args.device, mode='predict')

    # get detection_msg from ROS topic


    predictor.predict(detection_msg)