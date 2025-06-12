import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line

def osm_reader(file_path, crs, semantic_mask,cmap_res):
    gdf = gpd.read_file(file_path)
    gdf = gdf.to_crs(crs)
    semantic_mask.setflags(write=True)

    # Extract features of interest
    highway_gdf = gdf[gdf['highway'].notna()]
    creek_gdf = gdf[gdf['waterway'].notna()]

    # Get dimensions of semantic mask
    mask_height, mask_width = semantic_mask.shape

    # Get bounds of the GDF data
    minx, miny, maxx, maxy = gdf.total_bounds

    # Scale factors
    x_scale = mask_width / (maxx - minx)
    y_scale = mask_height / (maxy - miny)

    def update_mask(geometry, class_value,line_width):
        if geometry.geom_type == 'LineString':
            coords = np.array(geometry.coords)
            process_coords(coords, class_value,line_width)
        elif geometry.geom_type == 'MultiLineString':
            for line_geom in geometry.geoms:
                coords = np.array(line_geom.coords)
                process_coords(coords, class_value,line_width)

    def process_coords(coords, class_value, line_width):
        # Scale coordinates to match semantic mask dimensions
        scaled_x = ((coords[:, 0] - minx) * x_scale).astype(int)
        # Flip Y-axis: Invert the scaling for Y-coordinates
        scaled_y = (mask_height - ((coords[:, 1] - miny) * y_scale)).astype(int)

        # Clip to ensure within bounds
        scaled_x = np.clip(scaled_x, 0, mask_width - 1)
        scaled_y = np.clip(scaled_y, 0, mask_height - 1)

        # Draw line on the semantic mask
        for i in range(len(scaled_x) - 1):
            rr, cc = line(scaled_y[i], scaled_x[i], scaled_y[i + 1], scaled_x[i + 1])
            
            # Calculate half width
            half_width = line_width // 2
            
            # Update the line and surrounding pixels
            for idx in range(len(rr)):
                r, c = rr[idx], cc[idx]
                
                # Update the center point
                semantic_mask[r, c] = class_value
                
                # Update points in all directions within the half_width
                for dr in range(-half_width, half_width + 1):
                    for dc in range(-half_width, half_width + 1):
                        # Check if we're within bounds
                        if (0 <= r + dr < mask_height and 0 <= c + dc < mask_width):
                            # Optional: You can use distance formula to make circular brush
                            if dr*dr + dc*dc <= half_width*half_width:  # For circular brush
                                semantic_mask[r + dr, c + dc] = class_value

    # Update semantic mask for waterways (class value 5)
    for _, row in creek_gdf.iterrows():
        update_mask(row.geometry, class_value=8,line_width = int(30/cmap_res))
    
    # Update semantic mask for highways (class value 0)
    for _, row in highway_gdf.iterrows():
        update_mask(row.geometry, class_value=0,line_width = int(30/cmap_res))

    return semantic_mask


if __name__ == "__main__":
    semantic_mask = np.ones((2000, 2000), dtype=np.uint8)
    updated_mask = osm_reader("assets/OSM/NC_site1/Fbragg_osm.gpkg", "EPSG:32614", semantic_mask)

    # Display the resulting mask
    plt.imshow(updated_mask, cmap='gray')
    plt.show()
