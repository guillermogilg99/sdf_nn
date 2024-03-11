# ----------------Includes y variables globales
import rospy
from std_msgs.msg import String
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Imu
import sensor_msgs.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from numpy import random
from scipy.spatial import cKDTree
import message_filters

import os
import torch
import torch.nn as nn
import torch.optim as optim
from siren_pytorch import SirenNet
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

#Complete point list
total_wall_point_list = np.empty((0,3))

#Lista de puntos en coordenadas globales donde se han tomado muestras
training_positions = np.empty((0,3))

#Puntos para el entrenamiento
training_points = np.empty((0,4))

#Puntos pasados (para recordar)
saved_points = np.empty((0,4))

#Puntos de la pared escogidos (para posterior validación)
taken_wall_points = np.empty((0,4))

#Topic selection variables
num_topics = 0


#Initialize variables
Q = np.array([1, 0, 0, 0])
Q_last = np.array([1, 0, 0, 0])
x_pos = 0
y_pos = 0
z_pos = 0
x_last = -1000
y_last = -1000
z_last = -1000
x_ini = np.nan
y_ini = np.nan
z_ini = np.nan

# ----------------Voxel map and functions definition 
voxel_size = 0.01 # voxel size (m)
voxel_map_dim = 10 # voxel map radius (m) 

voxel_grid_dim = voxel_map_dim / voxel_size # voxel map radius (in voxel numbers)
map_total_dim = 2*voxel_grid_dim + 1 # voxel map dimension (in voxel numbers)
voxel_grid = np.full((map_total_dim,map_total_dim, map_total_dim), -1000)

def check_line_of_sight(x_ini,y_ini,z_ini,x_last,y_last,z_last): # Returns 1 if line of sight exists, 0 otherwise

    global voxel_grid

    # Calculte difference between points
    dx = x_last - x_ini
    dy = y_last - y_ini
    dz = z_last - z_ini

    # Determne increments for each axis
    sx = 1 if dx > 0 else -1
    sy = 1 if dy > 0 else -1
    sz = 1 if dz > 0 else -1

    dx = abs(dx)
    dy = abs(dy)
    dz = abs(dz)

    # Initialize internal counters
    cont_x = 1
    cont_y = 1
    cont_z = 1

    x_act = x_ini
    y_act = y_ini
    z_act = z_ini

    while (x_act, y_act, z_act) != (x_last, y_last, z_last):
        x_coef = cont_x/(dx*2)
        y_coef = cont_y/(dy*2)
        z_coef = cont_z/(dz*2)

        if (x_coef <= y_coef and x_coef <= z_coef):
            x_act += sx
            cont_x += 2
        if (y_coef <= x_coef and y_coef <= z_coef):
            y_act += sy
            cont_y += 2
        if (z_coef <= y_coef and z_coef <= x_coef):
            z_act += sz
            cont_z += 2

        if (voxel_grid[x_act][y_act][z_act] == 0):
            return 0
    
    return 1

def update_sdf_point(target_point_x, target_point_y, target_point_z, x_drone, y_drone, z_drone, discrete_wall_coordinates):
    # Comparar cada punto con los obstáculos nuevos y actualizarlo si el valor absoluto es menor con las nuevas paredes
    global voxel_grid
    sdf_prov_distance = np.nan # Inicializo la variable para guardar el sdf
    for row in discrete_wall_coordinates: # Por cada nuevo obstáculo detectado
        dist_to_obstacle = np.sqrt((target_point_x - row[0])**2 + (target_point_y - row[1])**2 + (target_point_z - row[2])**2) * voxel_size
        if (sdf_prov_distance == np.nan or dist_to_obstacle < sdf_prov_distance): # Si es el primer punto o el nuevo punto está más cerca que los anteriores
            sdf_prov_distance = dist_to_obstacle # Actualizo el sdf estimado

    # Una vez terminado se comprueba el signo
    if(voxel_grid[target_point_x][target_point_y][target_point_z] <= 0): # Si el sdf anterior es negativo
        los_result = check_line_of_sight(x_drone,y_drone,z_drone,target_point_x, target_point_y, target_point_z) # Checkea Line of Sight
        if(los_result == 1): # Si hay línea de visión
            voxel_grid[target_point_x][target_point_y][target_point_z] = sdf_prov_distance # El nuevo sdf es positivo
        else:                 # Si no
            voxel_grid[target_point_x][target_point_y][target_point_z] = -sdf_prov_distance # El nuevo sdf es negativo
    else:
        voxel_grid[target_point_x][target_point_y][target_point_z] = sdf_prov_distance # Si ya era positivo antes, debe seguir siéndolo
    return


#DEPRECATED def update_sdf_point(target_point_x, target_point_y, target_point_z, x_drone, y_drone, z_drone):
#    # Comparar cada punto con los obstáculos nuevos y actualizarlo si el valor absoluto es menor con las nuevas paredes
#    global voxel_grid
#    for radius in range(1,voxel_grid_dim):
#        indices =  np.empty((0,3))
#        for i in range(-radius, radius + 1):
#            for j in range(-radius, radius + 1):
#                search_pos = np.array([target_point_x + i, target_point_y + j, target_point_z + radius])
#                indices = np.append(indices, [search_pos], axis=0)
#                search_pos = np.array([target_point_x + i, target_point_y + j, target_point_z - radius])
#                indices = np.append(indices, [search_pos], axis=0)
#        for i in range(-radius, radius + 1):
#            for k in range (-radius + 1, radius):
#                search_pos = np.array([target_point_x + i, target_point_y + radius, target_point_z + k])
#                indices = np.append(indices, [search_pos], axis=0)
#                search_pos = np.array([target_point_x + i, target_point_y - radius, target_point_z + k])
#                indices = np.append(indices, [search_pos], axis=0)
#        for j in range(-radius + 1, radius):
#            for k in range (-radius + 1, radius):
#                search_pos = np.array([target_point_x + radius, target_point_y + j, target_point_z + k])
#                indices = np.append(indices, [search_pos], axis=0)
#                search_pos = np.array([target_point_x - radius, target_point_y + j, target_point_z + k])
#                indices = np.append(indices, [search_pos], axis=0)
#        
#        sdf_prov_distance = 0   # Inicializo la variable donde guarda el sdf estimado
#        for row in indices:        # Para cada punto de la lista
#            if(0 <= row[0] < map_total_dim and 0 <= row[1] < map_total_dim and 0 <= row[2] < map_total_dim): # Si la casilla está en el grid (hay puntos de la lista que no tienene por qué estar en el grid delimitado)
#                if(voxel_grid[row[0]][row[1]][row[2]] == 0):                                              # Si es un obstáculo
#                    dist_to_obstacle = np.sqrt((target_point_x - row[0])**2 + (target_point_y - row[1])**2 + (target_point_z - row[2])**2) * voxel_size #   Calcula la distancia al obstáculo
#                    if(sdf_prov_distance == 0 or dist_to_obstacle < sdf_prov_distance): # Si es el primer obstáculo que encuentra o la distancia al mismo es la menor hasta ahora
#                        sdf_prov_distance = dist_to_obstacle                            # Guarda la distancia como módulo del sdf que se va a introducir en ese punto
#        
#        if(sdf_prov_distance != 0): # Cuando ha encontrado al menos una pared (p)
#            if(voxel_grid[target_point_x][target_point_y][target_point_z] < 0): # Si el sdf anterior es negativo
#                los_result = check_line_of_sight(x_drone,y_drone,z_drone,target_point_x, target_point_y, target_point_z) # Checkea LOS
#                if(los_result == 1): # Si hay línea de visión
#                    voxel_grid[target_point_x][target_point_y][target_point_z] = sdf_prov_distance # El nuevo sdf es positivo
#                else:                 # Si no
#                    voxel_grid[target_point_x][target_point_y][target_point_z] = -sdf_prov_distance # El nuevo sdf es negativo
#            else:
#                voxel_grid[target_point_x][target_point_y][target_point_z] = sdf_prov_distance # Si ya era positivo antes, debe seguir siéndolo
#            return
                
def update_sdf(x_drone,y_drone,z_drone, discrete_wall_coordinates):
    global voxel_grid
    for i in range(voxel_grid.shape[0]):
        for j in range(voxel_grid.shape[1]):
            for k in range(voxel_grid.shape[2]):
                if(voxel_grid[i][j][k] != 0):
                    update_sdf_point(i,j,k,voxel_grid,x_drone,y_drone,z_drone, discrete_wall_coordinates)

# ----------------Declaración de la red ==> SIREN, 4 hidden layers con 256 neuronas, periodic (sinusoidal) activations, linear output layer, custom loss, custom initial weights
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class SIREN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256, hidden_layers=4, output_dim=1, omega_0=10):
        super(SIREN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.omega_0 = omega_0

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.hidden_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.init_weights()

    def init_weights(self):
        for layer in self.layers:
            n = layer.in_features
            std = (6 / n) ** 0.5 / self.omega_0
            torch.nn.init.uniform_(layer.weight, -std, std)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = x * self.omega_0
        for layer in self.layers[:-1]:
            x = torch.sin(layer(x))
        x = self.layers[-1](x)
        return x

class CustomLoss(nn.Module): # Declare custom loss function (LossTotal = lambda_SDF*LossSDF + lambda_eikonal*LossEikonal)
    def __init__(self, lambda_SDF, lambda_eikonal):
        super(CustomLoss, self).__init__()
        self.lambda_SDF = lambda_SDF
        self.lambda_eikonal = lambda_eikonal

    def forward(self, output, target, grad_output):
        # LossSDF (Difference between estimation and real value)
        loss_sdf = torch.mean(torch.abs(output - target))

        # LossEikonal (Difference between norm of the estimated gradient and 1)
        grad_norm = torch.norm(grad_output, p=2, dim=1)
        loss_eikonal = torch.mean(torch.abs(grad_norm - 1))

        # Combine both losses with constants lambda_SDF and lambda_eikonal
        total_loss = self.lambda_SDF * loss_sdf + self.lambda_eikonal * loss_eikonal
        return total_loss
    
# Create the model

siren_model = SIREN(input_dim=3, hidden_dim=256, hidden_layers=4, output_dim=1, omega_0=10).to(device)
siren_model.train()
print(siren_model)


# ----------------Obtener puntos del LiDAR y posición 3D del robot

def RotQuad(Q): #Puede estar al revés (comprobar)
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
    
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
    
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
    
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def PC_POS_callback_2topics(PC_msg, POS_msg):
    print("Synchronized message")
    #print("POS Seq: ", POS_msg.header.seq)
    #print("POS Stamp: ", POS_msg.header.stamp)
    #print("PC Seq: ", PC_msg.header.seq)
    #print("PC Stamp: ", PC_msg.header.stamp)
    global x_pos, y_pos, z_pos, Q
    x_pos = POS_msg.pose.position.x
    y_pos = POS_msg.pose.position.y
    z_pos = POS_msg.pose.position.z
    q0 = POS_msg.pose.orientation.w #Pendiente de revisar si las coordenadas son en este orden
    q1 = POS_msg.pose.orientation.x
    q2 = POS_msg.pose.orientation.y
    q3 = POS_msg.pose.orientation.z
    Q = np.array([q0, q1, q2, q3])
    SIREN_Trainer(PC_msg)

def SIREN_Trainer(PC_msg):
    #----------------Parameters------------------------------------------------------------------
    dist_between_iter = 0.1 # Meters between two training iterations
    #ang_between_iter = 20 # Degrees between two training iterations
    lidar_lim_max = 25 # Radius of the circle, centered in the drone, where LiDAR points are taken into account (meters)
    lidar_lim_min = 0.3
    max_rays = 500  # Max number of rays to be considered
    max_wall_points = 500 # Max number of wall points to be considered for training
    points_per_ray = 5 # Points per LiDAR ray
    num_past_points = 500 # Past points to be added to the training
    num_epochs = 10 # NN hyperparameter
    batch_size = 64 # NN hyperparameter
    lambda_SDF = 5 # SDF loss weight
    lambda_eikonal = 2 # Eikonal loss weight
    learning_rate = 4e-4
    weight_decay = 0.012
    #num_val_points = 100 # Validation points to be considered between epochs

    global x_pos, y_pos, z_pos, x_last, y_last, z_last, Q, Q_last, total_wall_point_list, training_points, taken_wall_points, training_positions, saved_points

    #print(f"q0 = {Q[0]}, q1 = {Q[1]}, q2 = {Q[2]}, q3 = {Q[3]} || q0last = {Q_last[0]}, q1last = {Q_last[1]}, q2last = {Q_last[2]}, q3last = {Q_last[3]}")
    acos_arg = np.abs(Q[0]*Q_last[0]+Q[1]*Q_last[1]+Q[2]*Q_last[2]+Q[3]*Q_last[3])
    if(acos_arg > 1):
        acos_arg = 1
    elif(acos_arg < -1):
        acos_arg = -1
    #print(f"Argumento del arcocoseno: {asin_arg}")
    drone_pos = np.array([x_pos, y_pos, z_pos])
    if training_positions.shape[0] == 0:
        dist_to_closest_tp = dist_between_iter + 1 # Asures that the training is made if the list is empty
    else:
        tpkdtree = cKDTree(training_positions)
        dist_to_closest_tp, _ = tpkdtree.query(drone_pos) # Checks distance to the closest training position
    if(dist_to_closest_tp > dist_between_iter):
    #if (np.sqrt((x_pos-x_last)**2 + (y_pos-y_last)**2 + (z_pos-z_last)**2) > dist_between_iter or 180/np.pi*np.arccos(acos_arg) > ang_between_iter):
        training_positions = np.append(training_positions, [drone_pos], axis=0) # Add this position to the list of positions where training was performed
        print("x =", x_pos)
        print("y =", y_pos)
        print("z =", z_pos)
        dist_dif = np.sqrt((x_pos-x_last)**2 + (y_pos-y_last)**2 + (z_pos-z_last)**2)
        ang_dif = 180/np.pi*np.arccos(acos_arg)
        print("distance dif =", dist_dif)
        print("angle dif =", ang_dif)
        R = RotQuad(Q) #Matriz de rotación del dron
        x_last = x_pos
        y_last = y_pos
        z_last = z_pos
        Q_last = Q
        pointcount = 0 #Contador de puntos detectados por el LiDAR
        wall_coordinates = np.empty((0,3)) #Para guardar los puntos con SDF=0 (paredes de objetos)
        for p in pc2.read_points(PC_msg, field_names = ("x", "y", "z"), skip_nans=True):
            #Move to global coordinates
            if((p[0]**2 + p[1]**2 + p[2]**2) < lidar_lim_max**2 and (p[0]**2 + p[1]**2 + p[2]**2) > lidar_lim_min**2): # Si está dentro del radio deseado, lo guarda como puntos de la pared
                plocal = np.array([p[0], p[1], p[2]])
                pglobal = drone_pos + R @ plocal
                wall_coordinates =np.append(wall_coordinates, [pglobal], axis=0)
                total_wall_point_list = np.append(total_wall_point_list, [pglobal], axis=0)
                #print(" x : %f  y: %f  z: %f" %(pglobal[0],pglobal[1],pglobal[2]))
                pointcount = pointcount + 1
        print("pointcount =", pointcount)

        training_points = np.empty((0,4)) # Reset the training points



    # ----------------Actualizar Voxfield con los puntos nuevos y obtener samples para el entrenamiento-----------------------
    N = 10000 # samples close to surface (dist < 5 cm)
    M = 30000 # samples away from surface
    NM = 40000 # Total samples (temporal solution)
    

    # Initialize initial position if not already done
    if (x_ini == np.nan and y_ini == np.nan and z_ini == np.nan):
        x_ini = x_pos
        y_ini = y_pos
        z_ini = z_pos

    # Update voxel grid with new points
    discrete_wall_coordinates = np.empty((0,3))
    for row in wall_coordinates:
        x_wall_p = np.rint((row[0] - x_ini)/voxel_size) + voxel_grid_dim # Convert to grid coordinates
        y_wall_p = np.rint((row[1] - y_ini)/voxel_size) + voxel_grid_dim
        z_wall_p = np.rint((row[2] - z_ini)/voxel_size) + voxel_grid_dim
        if (voxel_grid[x_wall_p][y_wall_p][z_wall_p] != 0): # If its a new obstacle (not detected before)
            new_discrete_wall_point = np.array([x_wall_p, y_wall_p, z_wall_p])  
            discrete_wall_coordinates = np.append(discrete_wall_coordinates,[new_discrete_wall_point], axis=0) # Store it in this list of points
            voxel_grid[x_wall_p][y_wall_p][z_wall_p] = 0 # And set the sdf value to 0
    
    update_sdf(voxel_grid, x_pos, y_pos, z_pos, discrete_wall_coordinates) # Update the sdf with the new points

    # Sampling of the voxel grid
    
    for samp in range(NM):
        x_samp = np.random.randint(0,map_total_dim - 1)
        y_samp = np.random.randint(0,map_total_dim - 1)
        z_samp = np.random.randint(0,map_total_dim - 1)
        x_coord = (x_samp - voxel_grid_dim) * voxel_size + x_ini
        y_coord = (y_samp - voxel_grid_dim) * voxel_size + y_ini
        z_coord = (z_samp - voxel_grid_dim) * voxel_size + z_ini
        new_tp = np.array([x_coord, y_coord, z_coord, voxel_grid[x_samp][y_samp][z_samp]])
        training_points = np.append(training_points,[new_tp], axis=0)

    # ----------------Obtener puntos locales a través de la estimación del SDF por fuerza bruta-------------------------------
    truncation_dist = 0.2 # max distance to consider local points
    S = 1000 # rays to be considered
    Q = 20 # points per ray to be considered

    #Create the kdTree for the next step
    pkdtree = cKDTree(total_wall_point_list)

    rnd_ray = np.random.choice(range(0, pointcount), S, replace=False) # Takes random samples of available rays
    for k in rnd_ray: #For each ray
        wall_point = np.array([wall_coordinates[k][0], wall_coordinates[k][1], wall_coordinates[k][2]]) # This is the wall point of that ray
        for l in range(Q): #For each point per ray
            dist_to_wall = random.uniform(-truncation_dist, truncation_dist)
            void_point = wall_point + ((wall_point-drone_pos)/np.linalg.norm(wall_point-drone_pos))*dist_to_wall # Coge puntos aleatorios a lo largo del rayo
            p_sdf_estimado, _ = pkdtree.query(void_point)
            if dist_to_wall > 0: # Si el punto está dentro de la pared, se cambia el signo del sdfestimado
                p_sdf_estimado = -p_sdf_estimado
            new_tp = np.array([void_point[0], void_point[1], void_point[2], p_sdf_estimado])
            if truncation_dist >= np.abs(p_sdf_estimado): # If within bounds, add to the training point list
                training_points = np.append(training_points,[new_tp], axis=0)

    # ----------------Entrenamiento de la SIREN--------------------------------------------------------------------------------
    criterion = CustomLoss(lambda_SDF, lambda_eikonal)
    optimizer = optim.Adam(siren_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Load dataset (X_input, y_output) and move them to the GPU (if able). Then convert to float32 (expected by the NN)
    X_input_batch = torch.tensor(training_points[:, :3])
    y_output_batch = torch.tensor(training_points[:,3])
    print(X_input_batch)
    print(y_output_batch)
    X_input_batch = X_input_batch.to(device)
    y_output_batch = y_output_batch.to(device)
    X_input_batch = X_input_batch.to(torch.float32)
    y_output_batch = y_output_batch.to(torch.float32)
    print("X_input_batch dtype:", X_input_batch.dtype)
    print("y_output_batch dtype:", y_output_batch.dtype)

    # Create the DataLoader
    point_dataset = TensorDataset(X_input_batch, y_output_batch)
    train_loader = DataLoader(point_dataset, batch_size=batch_size, shuffle=True)


    # Train the model
    for epoch in range(num_epochs):
        running_loss = 0.0
        #val_mse_total = 0.0


        for batch in train_loader:
            inputs, targets = batch
            targets = targets.view(-1,1)
            optimizer.zero_grad()
            outputs = siren_model(inputs)
        
            # Compute gradients of the outputs w.r.t. the inputs
            inputs.requires_grad = True
            grad_output = torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True)[0]

            loss = criterion(outputs, targets, grad_output)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        # Check validation each epoch
        #for k in range(num_val_points):
        #    val_input_tensor = torch.tensor([[val_point_list[k][0], val_point_list[k][1], val_point_list[k][2]]], dtype=torch.float32)
        #    val_output = NeRF(val_input_tensor.to(device))
        #    val_output_item = val_output.item()
        #    val_mse_total = val_mse_total + (val_output_item-val_pont_kdtree_sdf[k][0])**2

        #val_mse_total = val_mse_total/num_val_points

        #print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader)} ValLoss: {val_mse_total}')
        print(f'Epoch [{epoch + 1}/{num_epochs}] Loss: {running_loss / len(train_loader)}')

    # Save the trained model if needed
    torch.save(siren_model.state_dict(), 'siren_model.pth')
    print("Saved SIREN model")



# -----------------Main y establecimiento del nodo
def main():
# Initialize the ROS node
    rospy.init_node('nerf_sdf', anonymous=True)

    #Input topic selection
    topic_selector = 0


    if topic_selector == 0:
        topic_name_PC = "/velodyne_points"
        topic_name_POS = "/ground_truth_to_tf/pose"
        PC_sub = message_filters.Subscriber(topic_name_PC, PointCloud2)
        POS_sub = message_filters.Subscriber(topic_name_POS, PoseStamped)
        ts = message_filters.ApproximateTimeSynchronizer([PC_sub, POS_sub], queue_size=1000000, slop=0.1)
        ts.registerCallback(PC_POS_callback_2topics)

    elif topic_selector == 1:
        topic_name_PC = "/os1_cloud_node1/points"
        topic_name_POS = "/leica/pose/relative"
        PC_sub = message_filters.Subscriber(topic_name_PC, PointCloud2)
        POS_sub = message_filters.Subscriber(topic_name_POS, PoseStamped)
        ts = message_filters.ApproximateTimeSynchronizer([PC_sub, POS_sub], queue_size=10000000, slop=0.1)
        ts.registerCallback(PC_POS_callback_2topics)
    else:
        print("Error: topic_selector value is not supported")

    # Keep the script running
    rospy.spin()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass