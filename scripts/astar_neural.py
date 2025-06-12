import sys
from rospkg import RosPack
import os

rospack = RosPack()
package_path = rospack.get_path('camel')
sys.path.append(os.path.join(package_path,'scripts/neural-astar/src/')) # Add the path to the neural-astar module
from torchvision import transforms as T

import numpy as np
import torch
import math
from neural_astar.planner import VanillaAstar
import torch.utils.data as data
device = "cuda" if torch.cuda.is_available() else "cpu"

vanilla_astar = VanillaAstar().to(device)

def create_map(start, goal,array):
    #print(array.shape)
    cmap = np.zeros((4,1, array.shape[0],array.shape[1])).astype(np.float32)
    smap = np.zeros((4,1,array.shape[0],array.shape[1])).astype(np.float32)
    for i in smap:
        i[0][start[0],start[1]] = 1

    gmap = np.zeros((4,1,array.shape[0],array.shape[1])).astype(np.float32)
    for i in gmap:
        i[0][goal[0],goal[1]] = 1
    start_map = torch.from_numpy(smap)
    goal_map = torch.from_numpy(gmap)
    #cmap = torch.abs((array - torch.min(array))/(torch.max(array)- torch.min(array))-1)
    cmap = torch.from_numpy(array)
    map = torch.zeros((4,1,array.shape[0],array.shape[1]),device = 'cpu')
    for i in range(4):
        map[i] = cmap
    vanilla_astar.eval()
    va_outputs = vanilla_astar(map.to(device), start_map.to(device), goal_map.to(device))

    history = torch.where(va_outputs.histories[0][0]>0.1)
    x = history[1].tolist()
    y = history[0].tolist()
    x_, y_ = zip(*sorted(zip(x, y)))
    exp_coords = [[x_[-i],y_[-i]] for i in range(1,len(x_)+1)]
    path = []
    print(exp_coords)

    for i in range(len(exp_coords)-1):
        cur_coord = exp_coords[i]
        cur_x = cur_coord[0]
        if exp_coords[i+1][0] <= cur_x:# and math.sqrt((exp_coords[i+1][0]-exp_coords[i][0])**2+(exp_coords[i+1][1]-exp_coords[i][1])**2)<1.4:
            path.append([cur_coord[1],cur_coord[0]])
            continue
    path.append([start[0],start[1]])
    path = np.asarray(path,dtype=np.uint16)
    #print(path)
    return path, np.asarray(exp_coords)