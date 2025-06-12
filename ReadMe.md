# Trailblazer: Learning Off-Road Costmaps for Long-Range Planning


Autonomous navigation in off-road environments is a challenging task, especially for Unmanned Ground Vehicles (UGVs) involved in search and rescue, exploration, and surveillance. **Trailblazer** is a novel framework that automates the generation of costmaps from multi-modal sensor data, such as satellite imagery and LiDAR, to enable efficient and scalable long-range path planning without manual tuning. By leveraging imitation learning and a differentiable A* planner, Trailblazer adapts to diverse terrains and dynamic environments, offering a robust solution for global planning tasks.


---


![Inputs to Trailblazer](/images/input.jpg)
*Fig. 1: Inputs to Trailblazer from the Texas A&M RELLIS test site. Satellite imagery and LiDAR data are used to generate semantic, height, slope, and intensity maps.*


![Trailblazer Architecture](/images/arch.jpg)
*Fig. 2: Trailblazer framework architecture. The encoder-decoder generates costmaps, while the Neural A* planner computes paths from the costmaps.*


![LiDAR vs DEM Costmaps](/images/demvslidar.jpg)
*Fig. 3: Comparison of costmaps generated using LiDAR (4 points/m²) and DEM (1/3 arc-second resolution) data.*

For more details, check out the [Paper](https://arxiv.org/abs/2505.09739)
---


## Getting Started


### Before You Begin
- We are providing ROS 1 package. The repository is tested with ROS Noetic.
- All maps and waypoints should be stored in the `assets` folder.
- Launch files for the RELLIS test site are included.
- The input is preprocessed and saved. The package currently runs the Trailblazer algorithm and generates waypoints in UTM format.
- A `requirements.txt` file is provided for environment setup.


## Manual Installation


Some dependencies need to be installed manually to ensure the proper functioning of the Trailblazer framework. These include:


1. **mmcv**  
2. **mmseg**


### Installation Instructions:
- **mmcv**: Follow the [mmcv GitHub page](https://github.com/open-mmlab/mmcv) for detailed installation steps.
- **mmseg**: Refer to the [mmsegmentation documentation](https://mmsegmentation.readthedocs.io/en/latest/get_started.html) for installation instructions.


---


## Asset Download


The assets required for Camp Roberts are too large to be uploaded to GitHub. You can download them from the following link:  
[Download Assets](https://drive.google.com/drive/folders/1zuBZ8SxEAVgn8I4qtAB87pzkWF05La-j?usp=sharing)


### After Downloading:
- Place the `.tif` file in the `assets/images/rellis` folder.  
- Place the `best.pth` file in the `scripts/weights/segformer` folder.


---


## Running the Package


To run the Trailblazer package, use the following command:


```bash
roslaunch camel rellis.launch
```
 Please run ``roscore`` and ``source devel/setup.bash`` before running the launch file.

## Launch file parameters

 The launch file parameters are as follows:

 1. waypoints_file: Path to the waypoints file. "None" to generate random waypoints.
 2. site_name: Name of the test site; ex: cavasos, irwin, camp roberts.
 3. cmap_resolution: Costmap resolution in meters.'
 4. epsg_code: epsg_code used by the pointcloud. If using PDAl library, use "None".
 5. UTM_zone: UTM zone of the region.
 6. gps_origin: GPS coordinate in UTM format at origin.
 7. image_input: Path to the overhead image.
 8. pointcloud_input: Path to the overhead pointcloud/DEM.
 9. osm_file: Path to OSM file input.

#### One can change the cmap_resolution parameter to generate the costmap at different resolutions.

## Scripts

 The scripts folder contains the following scripts:

 1. camel_pathV2.py: The main script that generates the waypoints.
 2. cmap_gen.py: Generates the costmap from the input.
 3. dijkstra.py: Implements the Dijkstra algorithm.
 4. input_gen.py: Generates the input from the overhead image and pointcloud.
 5. semantic_seg.py: Implements the semantic segmentation.
 6. utils/laz_reader.py: Reads the pointcloud data.
 7. utils/dem_reader.py: Reads Digital Elevation Maps and converts them to pointcloud.

## Future Updates

### The current version supports the RELLIS test site. Additional examples will be added soon.

### Future updates will include support for more test sites and enhanced functionality.

## Citation

```
@misc{viswanath2025trailblazerlearningoffroadcostmaps,
      title={Trailblazer: Learning offroad costmaps for long range planning}, 
      author={Kasi Viswanath and Felix Sanchez and Timothy Overbye and Jason M. Gregory and Srikanth Saripalli},
      year={2025},
      eprint={2505.09739},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.09739}, 
}
```




 



 

