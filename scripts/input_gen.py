import numpy as np
import cv2
import torch
from torchvision import transforms as T
import sys
torch.set_grad_enabled(False)
import time
import math
import open3d
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm
import dijkstra 
import cv2
from rospkg import RosPack
from PIL import Image
from scipy.ndimage import generic_filter
import osm_processing
Image.MAX_IMAGE_PIXELS = None

def pc2voxel(pc_color,cmap_res):
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(pc_color[:, 0:3])
    lol = np.zeros((pc_color.shape[0],3))
    lol[:,0],lol[:,1],lol[:,2] = pc_color[:,3]/65535*255,pc_color[:, 3]/65535*255,pc_color[:, 3]/65535*255
    pc.colors = open3d.utility.Vector3dVector(lol/255.)
    try:
        downpc = pc.voxel_down_sample(voxel_size = 0.25*cmap_res)
        downpc.estimate_normals(search_param = open3d.geometry.KDTreeSearchParamHybrid(radius = cmap_res,max_nn = 30))
    except RuntimeError:
        print('voxel not processed')
        downpc = 'False'
    else:
        downpc = pc
        downpc.estimate_normals(search_param = open3d.geometry.KDTreeSearchParamHybrid(radius = cmap_res,max_nn = 30))
    #open3d.visualization.draw_geometries([pc])
    return downpc

def height_profile(point_org, semantic,cmap_res,epsg_code,osm_file, dem = False):
        point = point_org.copy()
        gps_org = (np.min(point[:,0]),np.min(point[:,1]))
        gps_extend = (np.max(point[:,0]),np.max(point[:,1]))
        point[:,2] = point[:,2]
        point[:,0] = point[:,0]- np.min(point[:,0])
        point[:,1] = point[:,1]- np.min(point[:,1])

        #outlier filter
        # z_values = point[:,2]
        # z_mean = np.mean(z_values)
        # z_std = np.std(z_values)
        # zmin = z_mean - 3*z_std
        # zmax = z_mean + 3*z_std
        # point = point[(z_values >= zmin) & (z_values <= zmax)]

        x_res = np.max(point[:,0]) - np.min(point[:,0])
        y_res = np.max(point[:,1]) - np.min(point[:,1])
        sx = int(x_res/cmap_res) 
        sy = int(y_res/cmap_res) 
        if sx%2 != 0:
            sx = sx+1
        if sy%2 != 0:
            sy = sy+1
        semantic_image = Image.fromarray(semantic)
        semantic_image = semantic_image.resize((sx,sy),Image.NEAREST)
        semantic = np.array(semantic_image,dtype = np.uint8,copy = True)
        semantic[semantic == 2] = 5
        semantic[semantic == 3] = 7
        semantic[semantic == 4] = 9
        semantic[semantic == 5] = 11
        if osm_file != 'None':
            semantic = osm_processing.osm_reader(osm_file,epsg_code,semantic,cmap_res)
        print('Processing Pointcloud')
        pcd = pc2voxel(point,cmap_res)
        point = []
        print('Normals Estimated')
        # height_map = np.zeros((sx,sy),dtype = np.float32)
        # map_count = np.ones((sx,sy),dtype = np.uint8)
        # proj_map = np.zeros((sx,sy),dtype = np.float32)
        # slope_map = np.zeros((sx,sy),dtype = np.float32)
        # intensity_map = np.zeros((sx,sy),dtype = np.float32)
        if pcd == 'False':
            return 'False'
        else:
            processed_points = np.zeros((len(pcd.points),6),dtype = object) #x,y,z,intensity,normals, projection
            processed_points[:,:3] = np.asarray(pcd.points)
            processed_points[:,3] = np.asarray(pcd.colors)[:,0]
            processed_points[:,4] = np.abs(np.asarray(pcd.normals)[:,2])
            #processed_points[:,5] = np.asarray(pcd.colors)[:,1]/np.max(np.asarray(pcd.colors)[:,1])*3 ## if the pointcloud contains semantic projections

            print("Generating Map")
            y_indices = (processed_points[:, 0] / cmap_res).astype(int)
            x_indices = (processed_points[:, 1] / cmap_res).astype(int)

            # Create a mask for valid (in-bounds) points
            valid = (y_indices >= 0) & (y_indices < sx) & (x_indices >= 0) & (x_indices < sy)
            y_valid = y_indices[valid]
            x_valid = x_indices[valid]

            # Precompute slope values (vectorized acos with clipping to avoid domain errors)
            normals_z = processed_points[valid, 4].astype(np.float64)
            normals_z = np.clip(normals_z, -1.0, 1.0)
            slope_values = np.arccos(normals_z)

            # Use np.add.at for unbuffered accumulation (handles duplicate indices correctly)
            height_map = np.zeros((sx, sy), dtype=np.float32)
            map_count = np.ones((sx, sy), dtype=np.int32)
            intensity_map = np.zeros((sx, sy), dtype=np.float32)
            slope_map = np.zeros((sx, sy), dtype=np.float32)

            np.add.at(height_map, (y_valid, x_valid), processed_points[valid, 2].astype(np.float32))
            np.add.at(map_count, (y_valid, x_valid), 1)
            np.add.at(intensity_map, (y_valid, x_valid), processed_points[valid, 3].astype(np.float32))
            np.add.at(slope_map, (y_valid, x_valid), slope_values.astype(np.float32))

            if dem == True:
                height_map, intensity_map, slope_map, proj_map = height_map/map_count, intensity_map/map_count, slope_map/map_count, semantic  # np.rot90(semantic)#
            else:
                height_map, intensity_map, slope_map, proj_map = np.rot90(height_map/map_count), np.rot90(intensity_map/map_count), np.rot90(slope_map/map_count), semantic  # np.rot90(semantic)#

            intensity_map = np.abs(intensity_map - np.max(intensity_map))
            #identify negative obstacle and add the median offset.
            plain = slope_map < 0.1
            plain_heights = height_map[plain]
            median_height = np.median(plain_heights)
            negative_obs = plain & (height_map < median_height)
            height_map[negative_obs] += median_height

            # intensity_map = np.abs(intensity_map - (np.mean(intensity_map)+2*np.std(intensity_map)))*2
            intensity_map = (intensity_map - np.min(intensity_map))/(np.max(intensity_map)-np.min(intensity_map))
            height_map = (height_map - np.min(height_map))/(np.max(height_map)-np.min(height_map))
            slope_map = (slope_map - np.min(slope_map))/(np.max(slope_map)-np.min(slope_map))
            proj_map = proj_map/11 # since the max value in semantic map is 11, we can normalize it by dividing by 11.
            #intensity_map = (intensity_map - np.min(intensity_map))/(np.max(intensity_map)-np.min(intensity_map))
            #height_map = (height_map - np.min(height_map))/(np.max(height_map)-np.min(height_map))
            #slope_map = (slope_map - np.min(slope_map))/(np.max(slope_map)-np.min(slope_map))
            #proj_map = (proj_map - np.min(proj_map))/(np.max(proj_map)-np.min(proj_map))
            # print('loop time:',time.time()-tim)
            map_count = np.rot90(map_count)
            height_map[map_count==1], intensity_map[map_count==1], slope_map[map_count==1] = height_map.max(), intensity_map.max(), slope_map.max()
            cat = np.zeros((sy,sx,4),dtype = 'float32')
            cat[:,:,0] = intensity_map# org height_map
            cat[:,:,1] = height_map # org intensity_map
            cat[:,:,2] = slope_map # org slope_map
            cat[:,:,3] = proj_map # org proj_map
            # plot_map(height_map)
            # plot_map(slope_map)
            # plot_map(intensity_map)
            # plot_map(proj_map)
            
            return cat, gps_org
    

def plot_map(array):
    current_cmap = plt.cm.Blues
    current_cmap.set_bad(color='red')
    fig, ax = plt.subplots(figsize=(40,28)) #costmap
    ax.matshow(array,cmap=plt.cm.Blues, vmin=np.min(array), vmax = np.max(array)) 
    plt.show()

def plot_pathmap(array):
    current_cmap = plt.cm.Blues
    current_cmap.set_bad(color='red')
    fig, ax = plt.subplots(figsize=(80,80)) #costmap
    ax.matshow(array,cmap=plt.cm.Blues, vmin=0, vmax = 1) 
    plt.show()

def plot_pathimage(path,image):
    plt.scatter(path[:,1],path[:,0],c = 'r',s = 0.1)
    plt.imshow(image)
    plt.show()
