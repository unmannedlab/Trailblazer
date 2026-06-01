import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import line
import osmnx as ox
import pandas as pd

def osm_reader(file_path, crs, semantic_mask, cmap_res=1.0):
    # 1) Read OSM/Geo data
    try:
        gdf = gpd.read_file(file_path,layer='lines')
    except Exception:
        # If your OSMnx version errors on layer='lines', remove that kwarg.
        try:
            gdf = ox.features_from_xml(
                file_path,
                tags={"highway": True, "waterway": True},
                layer="lines"
            )
        except Exception:
            # Fallback without 'layer' for versions that don't support it
            print("Warning: Failed to read OSM data.")
            return semantic_mask
    # 2) CRS handling
    try:
        if gdf.crs is None:
            # If source CRS is unknown, assume target (best-effort)
            gdf.set_crs(crs, inplace=True)
        elif str(gdf.crs) != str(crs):
            gdf = gdf.to_crs(crs)
    except Exception:
        # Keep going even if CRS operations fail
        pass

    # 3) Semantic mask writeable
    try:
        semantic_mask.setflags(write=True)
    except Exception:
        pass

    # 4) Ensure tags/columns exist to avoid KeyErrors
    for col in ("highway", "waterway"):
        if col not in gdf.columns:
            gdf[col] = pd.NA

    # 5) Filter line-like geometries only (LineString or MultiLineString)
    gdf = gdf[gdf.geometry.notna()]
    line_like = gdf.geometry.geom_type.isin(["LineString", "MultiLineString"])
    gdf_lines = gdf[line_like]

    highway_gdf = gdf_lines[gdf_lines["highway"].notna()]
    creek_gdf   = gdf_lines[gdf_lines["waterway"].notna()]

    # If nothing to draw, return early
    if highway_gdf.empty and creek_gdf.empty:
        return semantic_mask

    # 6) Compute bounds and scales safely
    minx, miny, maxx, maxy = gdf_lines.total_bounds
    width = maxx - minx
    height = maxy - miny

    mask_height, mask_width = semantic_mask.shape

    # Guard against degenerate bounds
    if width == 0 or height == 0:
        return semantic_mask

    x_scale = mask_width / width
    y_scale = mask_height / height

    # 7) Ensure non-zero widths
    hw_width = max(1, int(round(5 / float(cmap_res))))
    ww_width = max(1, int(round(15 / float(cmap_res))))

    def process_coords(coords, class_value, line_width):
        # Scale coordinates to match semantic mask dimensions
        scaled_x = ((coords[:, 0] - minx) * x_scale).astype(int)
        # Flip Y-axis: map top of mask to maxy
        scaled_y = (mask_height - 1 - ((coords[:, 1] - miny) * y_scale).astype(int))

        # Clip to bounds
        scaled_x = np.clip(scaled_x, 0, mask_width - 1)
        scaled_y = np.clip(scaled_y, 0, mask_height - 1)

        # Draw thickened vertical band around the line
        for i in range(len(scaled_x) - 1):
            rr, cc = line(scaled_y[i], scaled_x[i], scaled_y[i + 1], scaled_x[i + 1])

            half_w = line_width // 2
            for r, c in zip(rr, cc):
                # center pixel
                semantic_mask[r, c] = class_value
                # thicken vertically
                if half_w > 0:
                    r0 = max(0, r - half_w)
                    r1 = min(mask_height - 1, r + half_w)
                    semantic_mask[r0:r1 + 1, c] = class_value

    def update_mask(geometry, class_value, line_width):
        geom_type = geometry.geom_type
        if geom_type == "LineString":
            coords = np.array(geometry.coords)
            if len(coords) >= 2:
                process_coords(coords, class_value, line_width)
        elif geom_type == "MultiLineString":
            for line_geom in geometry.geoms:
                coords = np.array(line_geom.coords)
                if len(coords) >= 2:
                    process_coords(coords, class_value, line_width)

    # 8) Draw highways and waterways
    for _, row in highway_gdf.iterrows():
        update_mask(row.geometry, class_value=1, line_width=hw_width)

    for _, row in creek_gdf.iterrows():
        update_mask(row.geometry, class_value=3, line_width=ww_width)

    return semantic_mask


if __name__ == "__main__":
    semantic_mask = np.ones((2000, 2000), dtype=np.uint8)
    # Provide cmap_res or rely on default
    updated_mask = osm_reader(
        "assets/OSM/NC_site1/Fbragg_osm.gpkg",
        "EPSG:32614",
        semantic_mask,
        cmap_res=1.0
    )

    plt.imshow(updated_mask, cmap="gray")
    plt.show()