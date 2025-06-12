import pdal
import numpy as np
import open3d
import pdal
import matplotlib.pyplot as plt
import json
import re
 
def reader(file):
  pipeline="""{
    "pipeline": [
      {
          "type": "readers.las",
          "filename": """+'"'+file+'"'+"""
      },
      {
          "type": "filters.sort",
          "dimension": "Z"
      }
    ]
  }"""
  r = pdal.Pipeline(pipeline)
  r.execute()
  metadata = json.loads(r.metadata)
  # Extract CRS information
  crs_info = metadata["metadata"]["readers.las"]["srs"]
  wkt = crs_info.get("wkt", "")

  # Find EPSG code in WKT string
  epsg_match = re.findall(r'AUTHORITY\["EPSG","(\d+)"\]', wkt)
  epsg_code = epsg_match[-1] if epsg_match else None

  arrays = r.arrays[0]
  keys = arrays.dtype.fields.keys()
  classes = arrays['Classification']
  tmp = classes.copy()
  classes[tmp == 2] = 1
  classes[tmp == 1] = 2
  classes[tmp == 7] = 1
  points = np.zeros((len(arrays),5))
  points[:,0], points[:,1], points[:,2], points[:,3], points[:,4]  = arrays['X'], arrays['Y'], arrays['Z'], arrays['Intensity'], classes
  return points, epsg_code