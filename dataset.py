import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class PointCloudDataset(Dataset):
    def __init__(self, df, num_points=4096, augmentation=None, r_prob=0.25):
        self.df = df
        self.num_points = num_points
        self.augmentation = augmentation
        self.r_prob = r_prob

    def __len__(self):
        return len(self.df) // self.num_points
    
    def random_rotate(self, points):
        ''' randomly rotates point cloud about vertical axis.'''
        phi = np.random.uniform(-np.pi, np.pi)
        theta = np.random.uniform(-np.pi, np.pi)
        psi = np.random.uniform(-np.pi, np.pi)

        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

        rot_y = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        rot_z = np.array([
            [np.cos(psi), -np.sin(psi), 0],
            [np.sin(psi), np.cos(psi), 0],
            [0, 0, 1]
        ])

        return np.matmul(points, rot_z)
    
    def __getitem__(self, idx):
        start_idx = idx * self.num_points
        end_idx = start_idx + self.num_points
        batch_df = self.df.iloc[start_idx:end_idx]

        coords = batch_df[['x', 'y', 'z']].values
        #Normalize Points
        point_cloud = (coords - np.mean(coords, axis=0)) / np.std(coords, axis=0)
        
        labels = batch_df['scalar_Label'].values
        
        if self.augmentation != 'False':
            # Random Gaussian Noise
            point_cloud += np.random.normal(0., 0.01, point_cloud.shape)
            # Random rotate 
            if np.random.uniform(0, 1) > 1 - self.r_prob:
                point_cloud[:, :3] = self.random_rotate(point_cloud[:, :3])

            point_cloud += np.random.normal(0, 0.02, size=point_cloud.shape)

        return torch.tensor(point_cloud, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)