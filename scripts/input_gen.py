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

def height_profile(point_org, semantics,cmap_res,epsg_code,osm_file, dem = False):
        point = point_org.copy()
        semantic = semantics.copy()
        gps_org = (np.min(point[:,0]),np.min(point[:,1]))
        gps_extend = (np.max(point[:,0]),np.max(point[:,1]))
        print('GPS:',gps_org,gps_extend)
        point[:,2] = point[:,2]
        point[:,0] = point[:,0]- np.min(point[:,0])
        point[:,1] = point[:,1]- np.min(point[:,1])

        if point.shape[1] == 3:
            points = np.zeros((point.shape[0],4),dtype = object)
            points[:,:3] = point
            points[:,3] = point[:,2]/np.max(point[:,2])*65535
            point = points
            dem = True
        ##outlier filter
        # z_values = point[:,2]
        # z_mean = np.mean(z_values)
        # z_std = np.std(z_values)
        # zmin = z_mean - 2*z_std
        # zmax = z_mean + 2*z_std
        # point = point[(z_values >= zmin) & (z_values <= zmax)]

        x_res = np.max(point[:,0]) - np.min(point[:,0])
        y_res = np.max(point[:,1]) - np.min(point[:,1])
        sx = int(x_res/cmap_res) + 1    
        sy = int(y_res/cmap_res) + 1
        print(sx,sy)
        if sx%2 != 0:
            sx = sx + 1
        if sy%2 != 0:
            sy = sy + 1
        semantic_image = Image.fromarray(semantic)
        semantic_image = semantic_image.resize((sx,sy),Image.NEAREST)
        semantic = np.array(semantic_image,dtype = np.uint8,copy = True)
        semantic[semantic == 2] = 5
        semantic[semantic == 3] = 7
        semantic[semantic == 4] = 9
        semantic[semantic == 5] = 11
        if osm_file != 'None':
            semantic = osm_processing.osm_reader(osm_file,epsg_code,semantic,cmap_res,gps_org, gps_extend)
        print('Processing Pointcloud')
        pcd = pc2voxel(point,cmap_res)
        point = []
        print('Normals Estimated')
        height_map = np.zeros((sx,sy),dtype = np.float32)
        map_count = np.ones((sx,sy),dtype = np.uint8)
        proj_map = np.zeros((sx,sy),dtype = np.float32)
        slope_map = np.zeros((sx,sy),dtype = np.float32)
        intensity_map = np.zeros((sx,sy),dtype = np.float32)
        if pcd == 'False':
            return 'False'
        else:
            processed_points = np.zeros((len(pcd.points),6),dtype = object) #x,y,z,intensity,normals, projection
            processed_points[:,:3] = np.asarray(pcd.points)
            processed_points[:,3] = np.asarray(pcd.colors)[:,0]
            processed_points[:,4] = np.abs(np.asarray(pcd.normals)[:,2])
            #processed_points[:,5] = np.asarray(pcd.colors)[:,1]/np.max(np.asarray(pcd.colors)[:,1])*3 ## if the pointcloud contains semantic projections

            print("Generating Map")
            tim = time.time()
            for bpoints in tqdm(processed_points):
                
                y_index = int(bpoints[0]/cmap_res)  # negative int to inverse the map
                x_index = int(bpoints[1]/cmap_res)

                height_map[y_index][x_index] = height_map[y_index][x_index] + bpoints[2]
                map_count[y_index][x_index] = map_count[y_index][x_index] + 1
                
                intensity_map[y_index][x_index] = intensity_map[y_index][x_index] + bpoints[3]

                try:
                    slope_map[y_index][x_index] = slope_map[y_index][x_index] + math.acos(bpoints[4])
                except:
                    slope_map[y_index][x_index] = slope_map[y_index][x_index] + 0.785

                # if isinstance(bpoints[5],type(None)):
                #     proj_map[y_index][x_index] = proj_map[y_index][x_index] + 2
                # else:    
                #     proj_map[y_index][x_index] = proj_map[y_index][x_index] + bpoints[5]
            
            map_count[map_count == 0] = 1  # avoid division by zero
            if dem == True:
                height_map, intensity_map, slope_map, proj_map = height_map/map_count, intensity_map/map_count, slope_map/map_count, semantic  # np.rot90(semantic)#
            else:
                height_map, intensity_map, slope_map, proj_map = np.rot90(height_map/map_count), np.rot90(intensity_map/map_count), np.rot90(slope_map/map_count), semantic  # np.rot90(semantic)#
            
            plot_map(height_map)
            # height_map = height_map/height_map.mean()
            # intensity_map = np.abs(intensity_map - np.max(intensity_map))
            # #normalize intensity map
            # proj_map = proj_map/proj_map.mean()
            # intensity_map = (intensity_map - np.min(intensity_map))/(np.max(intensity_map)-np.min(intensity_map))*proj_map.max()
            # slope_map = slope_map*4

            intensity_map = np.abs(intensity_map - (np.mean(intensity_map)+2*np.std(intensity_map)))*2
            intensity_map = (intensity_map - np.min(intensity_map))/(np.max(intensity_map)-np.min(intensity_map))
            height_map = (height_map - np.min(height_map))/(np.max(height_map)-np.min(height_map))
            slope_map = (slope_map - np.min(slope_map))/(np.max(slope_map)-np.min(slope_map))
            proj_map = proj_map/11

            print('loop time:',time.time()-tim)
            map_count = np.rot90(map_count)
            height_map[map_count==1], intensity_map[map_count==1], slope_map[map_count==1] = height_map.max(), intensity_map.max(), slope_map.max()
            cat = np.zeros((sy,sx,4),dtype = 'float32')
            cat[:,:,0] = intensity_map# org height_map
            cat[:,:,1] = height_map # org intensity_map
            cat[:,:,2] = slope_map # org slope_map
            cat[:,:,3] = proj_map # org proj_map
            plot_map(height_map)
            plot_map(slope_map)
            plot_map(intensity_map)
            plot_map(proj_map)
            
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
