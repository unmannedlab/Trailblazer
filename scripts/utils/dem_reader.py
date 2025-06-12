from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import cv2
import open3d 

def reader(path):
    image = gdal.Open(path)
    rband = image.GetRasterBand(1)
    #gband = image.GetRasterBand(2)
    #bband = image.GetRasterBand(3)
    #band4 = image.GetRasterBand(4)

    band4 = np.asarray(rband.ReadAsArray())
    pos = np.arange(1,band4.shape[1]+1,1)
    x = np.repeat(pos,band4.shape[0])
    y =np.repeat(pos.reshape(-1,1).transpose(), repeats = band4.shape[0], axis=0).flatten()
    z = band4.flatten() - np.min(band4)
    points = np.zeros((x.shape[0],3))
    points[:,0],points[:,1],points[:,2] = x,y,z
    return points