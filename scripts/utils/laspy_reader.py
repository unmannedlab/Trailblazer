import laspy
import numpy as np
import re
from pyproj import CRS

def reader(file):
    # Open the LAS/LAZ file
    las = laspy.read(file)
    
    # Extract point data
    points = np.zeros((len(las.points), 5))
    points[:, 0] = las.x
    points[:, 1] = las.y
    points[:, 2] = las.z
    points[:, 3] = las.intensity
    
    # Handle classification similar to your PDAL version
    classes = np.array(las.classification)
    tmp = classes.copy()
    classes[tmp == 2] = 1
    classes[tmp == 1] = 2
    classes[tmp == 7] = 1
    points[:, 4] = classes
    return points

