#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import dijkstra
import multiprocessing
from itertools import product
import os
import rospy
from time import time
from cmap_gen import costmap_gen
import input_gen
import semantic_seg
from PIL import Image
## if PDAL is installed
#import utils.laz_reader as laz_reader
## if PDAL is not installed
#import utils.laspy_reader as laz_reader
from rospkg import RosPack
import astar_neural
Image.MAX_IMAGE_PIXELS = None
import random
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from pyproj import CRS, Transformer
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import calculate_default_transform, reproject
import geopandas as gpd
from shapely.geometry import Point
from osgeo import gdal,osr
#import simplekml


rospack = RosPack()
package_path = rospack.get_path('camel')

def plot_pathimage(path,image):
    plt.scatter(path[:,1],path[:,0],s = 3)
    plt.imshow(image)
    plt.show()

def occupancy_grid_publisher(map_data, frame_id,cmap_res, publisher = None, gps_org = None):
    occupancy_grid_msg = OccupancyGrid()
    occupancy_grid_msg.header = Header()
    occupancy_grid_msg.header.frame_id = frame_id
    occupancy_grid_msg.header.stamp = rospy.Time.now()
    occupancy_grid_msg.info.resolution = cmap_res*0.308
    occupancy_grid_msg.info.width = map_data.shape[1]
    occupancy_grid_msg.info.height = map_data.shape[0]
    occupancy_grid_msg.info.origin.position.x = gps_org[1]
    occupancy_grid_msg.info.origin.position.y = gps_org[0]
    occupancy_grid_msg.info.origin.position.z = 0.0
    occupancy_grid_msg.info.origin.orientation.x = 0.0
    occupancy_grid_msg.info.origin.orientation.y = 0.0
    occupancy_grid_msg.info.origin.orientation.z = 0.0
    occupancy_grid_msg.info.origin.orientation.w = 1.0
    
    ## flipping costmap for ROS visualization
    map_data = np.flipud(map_data)
    occupancy_data = np.where(map_data == -1.0, -1,(map_data * 100).astype(np.int8)).flatten().tolist()
    occupancy_grid_msg.data = occupancy_data
    publisher.publish(occupancy_grid_msg)

def save_gtiff(image, utm_zone, cmap_res, gps_org):
    """
    Save the given image as a GeoTIFF with a specified UTM zone, 
    resolution, and GPS coordinates for the bottom-left corner.

    :param image: 2D numpy array (grayscale image)
    :param utm_zone: UTM zone (e.g., 32617 for UTM Zone 17N)
    :param cmap_res: Resolution of the image (in meters per pixel)
    :param gps_org: Tuple (longitude, latitude) for the bottom-left corner
    """
    # Calculate the affine transform
    # Bottom-left corner (x, y)
    x_bottom_left, y_bottom_left = gps_org
    
    # Adjust the y-coordinate for the top-left corner (Y increases as you go up)
    y_top_left = y_bottom_left + (cmap_res * image.shape[0])  # The top-left Y is higher than the bottom-left by the height of the image

    # Create the transform (from top-left corner, pixel size)
    transform = rasterio.transform.from_origin(x_bottom_left, y_top_left, cmap_res, cmap_res)

    # Define metadata for the GeoTIFF
    metadata = {
        'driver': 'GTiff',
        'count': 1,  # Single channel image (grayscale)
        'dtype': 'uint8',
        'width': image.shape[1],  # Number of columns (X)
        'height': image.shape[0],  # Number of rows (Y)
        'crs': CRS.from_epsg(utm_zone),  # UTM CRS (e.g., EPSG:32617)
        'transform': transform
    }

    # Write the image to the GeoTIFF
    with rasterio.open('/home/usl/output.tif', 'w', **metadata) as dst:
        dst.write((image * 255).astype(np.uint8), 1)  # Scale image to 0-255 for uint8 type

def save_lat_lon_as_gpkg(array, crs="EPSG:32617", attributes=None):
    """
    Save latitude and longitude arrays as points in a GeoPackage file
    
    Parameters:
    -----------
    lats : array-like
        Latitude values
    lons : array-like
        Longitude values
    output_path : str
        Path to save the GeoPackage file (should end with .gpkg)
    crs : str, default "EPSG:4326"
        Coordinate Reference System for the points
    attributes : dict or None, default None
        Additional attribute columns to add (dict keys become column names)
    """
    # Ensure lats and lons are numpy arrays of the same length
    lats = array[:, 0]
    lons = array[:, 1]
    
    if len(lats) != len(lons):
        raise ValueError("Latitude and longitude arrays must have the same length")
    
    # Create point geometries
    geometry = [Point(lon, lat) for lon, lat in zip(lons, lats)]
    
    # Create a dictionary with default attributes
    data = {
        'latitude': lats,
        'longitude': lons
    }
    
    # Add any additional attributes
    if attributes:
        for key, values in attributes.items():
            # Make sure values match the number of points
            if len(values) == len(lats):
                data[key] = values
            else:
                print(f"Warning: Attribute '{key}' has {len(values)} values but there are {len(lats)} points. Skipping.")
    
    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(data, geometry=geometry, crs=crs)
    
    # Save as GeoPackage
    gdf.to_file('/home/usl/output.gpkg', driver="GPKG")
    # Save as GeoPackage

def reproject_and_crop_single_step(image_path, gps_org, gps_extend, target_epsg=32617):
    """One-step reprojection and cropping using GDAL"""
    src_ds = gdal.Open(image_path)
    if src_ds is None:
        raise ValueError(f"Could not open {image_path}")
    
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(target_epsg)
    dst_wkt = dst_srs.ExportToWkt()
    
    num_bands = src_ds.RasterCount
    min_x, min_y = gps_org
    max_x, max_y = gps_extend
    
    # Set options for both reprojection and cropping in one step
    warp_options = gdal.WarpOptions(
        dstSRS=dst_wkt,
        outputBounds=[min_x, min_y, max_x, max_y],
        resampleAlg=gdal.GRA_NearestNeighbour
    )
    
    # Create an in-memory dataset with reprojection and cropping applied
    dst_ds = gdal.Warp('', src_ds, options=warp_options, format='MEM')
    
    dst_width = dst_ds.RasterXSize
    dst_height = dst_ds.RasterYSize
    
    # Read data from the processed dataset
    if num_bands == 1:
        cropped_data = dst_ds.GetRasterBand(1).ReadAsArray()
    else:
        cropped_data = np.zeros((num_bands, dst_height, dst_width), dtype=np.float32)
        for i in range(1, num_bands + 1):
            cropped_data[i-1] = dst_ds.GetRasterBand(i).ReadAsArray()
        
        if num_bands == 3:  # RGB case
            cropped_data = np.transpose(cropped_data, (1, 2, 0))
    
    # Clean up
    src_ds = None
    dst_ds = None
    
    return cropped_data

def load_waypoints(txt_file,gps_org,cmap_res,input_map):
        if txt_file == 'None':
            height = input_map.shape[0]
            width = input_map.shape[1]
            waypoints = []
            for j in range(5):
                x = random.randint(0, height - 1)
                y = random.randint(0, width - 1)
                waypoints.append((x, y))
                print(f"Waypoint: x={x}, y={y}")
        else:
        # Read waypoints from .txt file
            print("Waypoints from .txt file:")
            waypoints = []
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Assuming waypoints are in the format: x,y
                        x, y = map(float, line.split(','))
                        print(x,y)
                        print(gps_org)
                        waypoints.append((int((x-gps_org[0])/cmap_res),input_map.shape[0] - int((y-gps_org[1])/cmap_res)))
                        print(f"Waypoint: x={x}, y={y}")
        return waypoints

def process_files(image_path, pointcloud_path, cmap_res, txt_file, osm_file = None, epsg_code = None, utm_zone = None):
    #try:
    #load image and generate semantic mask

        if epsg_code == 'None':
        ##load pointcloud and generate input map using pdal
            import utils.laz_reader as laz_reader
            if pointcloud_path[-4:] == '.laz':
                points, epsg_code = laz_reader.reader(pointcloud_path)
            elif pointcloud_path[-4:] == '.las':
                points, epsg_code = laz_reader.reader(pointcloud_path)
            elif pointcloud_path[-4:] == '.npy':
                points = np.load(pointcloud_path)
            elif pointcloud_path[-4:] == '.tif':
                import utils.dem_reader as dem_reader
                points = dem_reader.reader(pointcloud_path)
                epsg_code = '32614'
        else:
            import utils.laspy_reader as laz_reader
            if pointcloud_path[-4:] == '.laz':
                points = laz_reader.reader(pointcloud_path)
            elif pointcloud_path[-4:] == '.las':
                points = laz_reader.reader(pointcloud_path)
            elif pointcloud_path[-4:] == '.npy':
                points = np.load(pointcloud_path)
            elif pointcloud_path[-4:] == '.tif':
                import utils.dem_reader as dem_reader
                points = dem_reader.reader(pointcloud_path)
        # if epsg_code != utm_zone:
        #     input_crs = CRS.from_epsg(epsg_code)
        #     output_crs = CRS.from_epsg(utm_zone)
        #     print(input_crs,output_crs)
        #     transformer = Transformer.from_crs(input_crs,output_crs, always_xy=True)
        #     points[:, 0], points[:, 1] = transformer.transform(points[:, 0], points[:, 1])
        #     epsg_code = utm_zone
        # gps_org = [float(np.min(points[:, 0])), float(np.min(points[:, 1]))]
        # gps_extend = [float(np.max(points[:, 0])), float(np.max(points[:, 1]))]
        # print(gps_org,gps_extend)

        # image = reproject_and_crop_single_step(image_path, gps_org, gps_extend, target_epsg=utm_zone)

        image = Image.open(image_path)
        #image = np.array(image)
        #image = image[:-327,448:,:]
        #image = Image.fromarray(image)
        semantic_mask = semantic_seg.SegmentationInference(os.path.join(package_path,'scripts/weights/segformer/best.pth')).process_image(image)
        input_map, gps_org = input_gen.height_profile(points, semantic_mask, cmap_res, epsg_code, osm_file)

        if txt_file == 'None':
            height = input_map.shape[0]
            width = input_map.shape[1]
            waypoints = []
            for j in range(5):
                x = random.randint(0, height - 1)
                y = random.randint(0, width - 1)
                waypoints.append((x, y))
                print(f"Waypoint: x={x}, y={y}")
        else:
        # Read waypoints from .txt file
            print("Waypoints from .txt file:")
            waypoints = []
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        # Assuming waypoints are in the format: x,y
                        x, y = map(float, line.split(','))
                        waypoints.append((int((x-gps_org[0])/cmap_res), input_map.shape[0] - int((y-gps_org[1])/cmap_res))) #the x remains same but the subtracting y from the height of the map
                        print(f"Waypoint: x={x}, y={y}")

    # except Exception as e:
    #     print(f"Error processing files: {e}")
        return input_map, waypoints, image, gps_org


def path_processing(start, goal, map,i, image,gps_org,cmap_res, epsg_code = 2264, utm_zone = 32617):
    global path_record
    start = [int(start[1]),int(start[0])] # lat, lon converts to y,x in numpy space
    goal = [int(goal[1]),int(goal[0])] # lat, lon converts to y,x in numpy space
    start_time = time()
    grid = dijkstra.Grid(map, start)
    dij = dijkstra.Dijkstra(grid, start, goal)
    dij.find_path()
    path = dij.backtrack_path()
    #path, history = astar_neural.create_map(start,goal,map)
    print('Path Found')
    path = np.array(path)
    print("Execution time:", time() - start_time)
    plt.matshow(map,cmap=plt.cm.Blues, vmin=np.min(map), vmax = np.max(map))
    plt.scatter(path[:,1],path[:,0],c = 'r',s = 5)
    plt.show()
    plot_pathimage(path,image)
    epsg_path = np.zeros((len(path),2))
    epsg_path[:,0] = path[:,1] * cmap_res + gps_org[0] # x coordinate
    epsg_path[:,1] = (map.shape[0] - path[:,0]) * cmap_res + gps_org[1] # y coordinate
    if epsg_code != utm_zone:
        input_crs = CRS.from_epsg(epsg_code)
        output_crs = CRS.from_epsg(utm_zone)
        transformer = Transformer.from_crs(input_crs,output_crs, always_xy=True)
        epsg_path[:, 0], epsg_path[:, 1] = transformer.transform(epsg_path[:, 0], epsg_path[:, 1])
    save_lat_lon_as_gpkg(epsg_path,attributes = {'index':np.arange(len(epsg_path))})

if __name__ == "__main__":
    rospy.init_node('path_generator', anonymous=True)
    cmap_pub = rospy.Publisher('/global_costmap',OccupancyGrid,queue_size=10)
    height_map_pub = rospy.Publisher('/height_map',OccupancyGrid,queue_size=10)
    slope_map_pub = rospy.Publisher('/slope_map',OccupancyGrid,queue_size=10)
    intensity_map_pub = rospy.Publisher('/intensity_map',OccupancyGrid,queue_size=10)
    proj_map_pub = rospy.Publisher('/semantic_map',OccupancyGrid,queue_size=10)

    if len(sys.argv) < 7:
        print("Usage: python script.py <npy_file> <txt_file> ,<name ;(ex:cavasos,irwin)>")
        sys.exit(1)
    
    image_path = sys.argv[1] ## Image path file (.tif/.png files)
    pointcloud_path = sys.argv[2] ## Pointcloud laz file (.laz files)
    txt_file = sys.argv[3] ##waypoints
    site_name = sys.argv[4] ##site name
    cmap_res = int(sys.argv[5]) ##costmap resolution
    epsg_code = sys.argv[6] ##UTM zone
    if epsg_code != 'None':
        epsg_code = int(epsg_code)
    utm_zone = int(sys.argv[7]) ##UTM zon
    osm_file = sys.argv[8] ##OSM file
    costmap = sys.argv[9] ##costmap file

    if costmap != 'None':
        mod_cmap = np.load(costmap,allow_pickle=True)
        try:
            gps_org = [float(sys.argv[9].split('/')[0]),float(sys.argv[9].split('/')[1])] ##gps origin
            waypoints = load_waypoints(txt_file,gps_org,cmap_res,mod_cmap)
            image = Image.open(image_path)
            occupancy_grid_publisher(mod_cmap,'utm_17',cmap_res,cmap_pub,gps_org)
            pairs = [(waypoints[i], waypoints[i + 1], mod_cmap,i, image.resize((mod_cmap.shape[1],mod_cmap.shape[0]),Image.NEAREST),gps_org,cmap_res) for i in range(len(waypoints) - 1)]
            cores = multiprocessing.cpu_count() 
            # with multiprocessing.Pool(4) as p:
            #     p.starmap(path_processing, pairs)
            for i in pairs:
                path_processing(i[0],i[1],i[2],i[3],i[4],i[5],i[6])
        except Exception as e:
            print(f"Error processing files: {e}")
    else:
        input_map , waypoints, image, gps_org = process_files(image_path, pointcloud_path, cmap_res, txt_file, osm_file, epsg_code, utm_zone)
        ## Input map publisher
        occupancy_grid_publisher(input_map[:,:,0],'utm_17',cmap_res,height_map_pub,gps_org)
        occupancy_grid_publisher(input_map[:,:,1],'utm_17',cmap_res,intensity_map_pub,gps_org)
        occupancy_grid_publisher(input_map[:,:,2],'utm_17',cmap_res,slope_map_pub,gps_org)
        occupancy_grid_publisher(input_map[:,:,3],'utm_17',cmap_res,proj_map_pub,gps_org)

        #costmap generation
        mod_cmap = costmap_gen(input_map, (input_map.shape[0],input_map.shape[1]))
        save_gtiff(mod_cmap,utm_zone,cmap_res,gps_org)
        #np.save(os.path.join(package_path,'assets/cost_maps',site_name+'_5_costmap.npy'),mod_cmap,allow_pickle=True)
        #costmap publisher
        occupancy_grid_publisher(mod_cmap,'utm_17',cmap_res,cmap_pub,gps_org)
        pairs = [(waypoints[i], waypoints[i + 1], mod_cmap,i, image.resize((input_map.shape[1],input_map.shape[0]),Image.NEAREST),gps_org,cmap_res) for i in range(len(waypoints) - 1)]
        cores = multiprocessing.cpu_count() 
        # with multiprocessing.Pool(4) as p:
        #     p.starmap(path_processing, pairs)
        for i in pairs:
            path_processing(i[0],i[1],i[2],i[3],i[4],i[5],i[6])
