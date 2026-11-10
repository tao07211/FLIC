import utils
import sampling
import numpy as np
import SimpleITK as sitk
from torch.utils import data

def build_coordinate_val(D, angle):
    angle_rad = np.deg2rad(angle)
    trans_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad),  np.cos(angle_rad)]
        ]
    )
    x = np.linspace(-1, 1, int(2 * D)).reshape(-1, 1)
    y = np.linspace(-1, 1, int(2 * D)).reshape(-1, 1)
    x, y = np.meshgrid(x, y, indexing='xy')            
    xy = np.stack([x, y], -1).reshape(-1, 2)
    xy = xy @ trans_matrix.T
    xy = xy.reshape(1, 2 * D, 2 * D, 2)
    return xy

class TestData(data.Dataset):
    def __init__(self, theta, D, fan_pose):
        angles = np.linspace(0, 180, theta + 1)[:-1]
        fan_angle = fan_pose
        self.rays = []
        for angle in angles:
            xy = utils.parallel_coordinate(len(fan_angle), D)
            self.rays.append(utils.build_coordinate_parallel(xy, angle))

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, item):
        return self.rays[item]               

class ValData(data.Dataset):
    def __init__(self, D, angle):
        self.rays = build_coordinate_val(D, angle)

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, item):
        return self.rays[item]

class TrainData(data.Dataset):
    def __init__(self, theta, sin_data, sample_N, fan_pose, D):
        self.sample_N = sample_N
        self.D = D

        angles = np.linspace(0, 180, theta + 1)[:-1]
        fan_angle = fan_pose
        sin = sin_data

        rays_list  = []
        proj_list  = []
        for angle in angles:
            proj_list.append(sin_data[len(proj_list), :])
            xy = utils.parallel_coordinate(len(fan_angle), D)                     
            rays_list.append(utils.build_coordinate_parallel(xy, angle))

        self.projections_lines = np.array(proj_list)                 
        self.rays_geom = np.array(rays_list)                                         

        self.rays = self.rays_geom.copy()                                   

        na, L, N, _ = self.rays.shape
        self.weights = np.ones((na, L, N), dtype=np.float64)              

    def resample(self, prior, eta, N_b=1):

        grid_size = self.rays_geom.shape[2]               
        N         = self.rays_geom.shape[2]                           

        new_rays, new_weights = sampling.resample_from_prior(
            prior      = prior,
            rays_geom  = self.rays_geom,
            grid_size  = grid_size,
            N          = N,
            eta        = eta,
            N_b        = N_b,
        )
        self.rays    = new_rays                                            
        self.weights = new_weights                                      

    def __len__(self):
        return len(self.projections_lines)

    def __getitem__(self, item):
        projection_l = self.projections_lines[item]         
        ray          = self.rays[item]                                  
        weights      = self.weights[item]                            

        sample_indices = np.random.choice(
            len(projection_l), self.sample_N, replace=False)

        return (ray[sample_indices].astype(np.float32),
                weights[sample_indices].astype(np.float32),
                projection_l[sample_indices].astype(np.float32))