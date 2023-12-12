#!/usr/bin/env python3

import rospy
from std_msgs.msg import String
import numpy as np
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from nav_msgs.msg import OccupancyGrid

#----------------------------------MAP DETAILS-------------------------------------

map_width = 10 # in meters
map_height = 10 # in meters
map_resolution = 0.1 # in meters
x_min = 5
y_min = -5
z_cut = 2.40


#------------------------------------NERF CONFIGURATION-----------------------------------------
#See if the nn can be placed in the GPU
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

map_pub = rospy.Publisher('/nerf_occupancy_grid', OccupancyGrid, queue_size=10)

class NeuralNetwork(nn.Module): # Se crea la red haciendo subclassing del nn.Module (estructura igual al paper del iSDF)
    def __init__(self): # Se inicializan las capas en el __init__
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_softplus_stack = nn.Sequential(
            nn.Linear(3, 64),
            nn.Dropout(0.5),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Dropout(0.5),
            nn.Softplus(),
            nn.Linear(64, 1),
        )

    def forward(self, x): # Se implementan las operaciones sobre los datos de entrada en el forward
        x = self.flatten(x)
        logits = self.linear_softplus_stack(x)
        return logits

NeRF = NeuralNetwork().to(device) # Instanciamos la red neuronal y la movemos a la GPU
NeRF.load_state_dict(torch.load('nerf_model.pth'))
NeRF.eval()  # Set the model to evaluation mode


def RViz_NeRF():
    # Create a new OccupancyGrid message
    occupancy_grid = OccupancyGrid()

    # Set the map metadata
    cell_width = int(map_width // map_resolution)
    cell_height = int(map_height // map_resolution)
    occupancy_grid.info.width = cell_width  # Width of the map in cells
    occupancy_grid.info.height = cell_height  # Height of the map in cells
    occupancy_grid.info.resolution = map_resolution  # Size of each cell in meters
    occupancy_grid.info.origin.position.x = -1.0  # Origin (lower-left corner) x-coordinate
    occupancy_grid.info.origin.position.y = -1.0  # Origin (lower-left corner) y-coordinate
    print(f"Cell width= {cell_width} // Cell height= {cell_height} // Resolution = {map_resolution}")

    # Generate occupancy values (0 to 100) for each cell
    grid_data=[]
    # Iterate over rows
    for row in range(cell_height):
        # Iterate over columns in each row
        for col in range(cell_width):
            value1 = x_min + (col+0.5)*map_resolution
            value2 = y_min + (row+0.5)*map_resolution
            value3 = z_cut
            input_tensor = torch.tensor([[value1, value2, value3]], dtype=torch.float32)
            output = NeRF(input_tensor.to(device))
            output_item = output.item()
            print(output_item)
            if output_item > 100: output_item = 100
            if output_item < 0: output_item = 0

            # Add your occupancy value (0 or 100) to the data list
            grid_data.append(output_item)

    
    # Rescale data (between 0 and 1000)

    grid_data_min = np.min(grid_data)
    grid_data_max = np.max(grid_data)
    print(f"Max = {grid_data_max} // Min = {grid_data_min}")

    # Rescale the array to have values between 0 and 100
    scaled_data = 100 * (grid_data - grid_data_min) / (grid_data_max - grid_data_min)
    scaled_data = scaled_data.astype(int)

    print(scaled_data)

    occupancy_grid.data = scaled_data.tolist()
    occupancy_grid.header.stamp = rospy.Time.now()
    occupancy_grid.header.frame_id = 'nerf_occupancy_grid'
    map_pub.publish(occupancy_grid)
    print("Grid message sent!")


def main():
    # Initialize the ROS node
    rospy.init_node('rviz_nerf_representation', anonymous=True)


    # Launch the representation
    RViz_NeRF()

    # Keep the script running
    rospy.spin()
    


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass