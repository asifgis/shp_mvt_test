import math
import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import mapbox_vector_tile
import fiona
from shapely.geometry import shape, box, mapping
from shapely.ops import transform
from pyproj import Transformer

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configuration
SHAPEFILE_PATH = "shp/test_shp.shp"  # Path to your shapefile

# Constants for tile extent
TILE_EXTENT = 4096

# Define projections
WGS84 = 'EPSG:4326'
WEB_MERCATOR = 'EPSG:3857'

# Initialize transformers
transformer_to_3857 = Transformer.from_crs(WGS84, WEB_MERCATOR, always_xy=True)


def tile_to_bbox(z, x, y):
    """
    Convert tile z/x/y to bounding box in Web Mercator (EPSG:3857).

    Args:
        z (int): Zoom level.
        x (int): Tile x coordinate.
        y (int): Tile y coordinate.

    Returns:
        shapely.geometry.box: Bounding box in EPSG:3857.
    """
    n = 2.0 ** z

    lon_deg_left = x / n * 360.0 - 180.0
    lon_deg_right = (x + 1) / n * 360.0 - 180.0

    def lat_deg_from_tile(y, z):
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        return math.degrees(lat_rad)

    lat_deg_top = lat_deg_from_tile(y, z)
    lat_deg_bottom = lat_deg_from_tile(y + 1, z)

    # Create bbox in lon/lat
    bbox_lonlat = box(lon_deg_left, lat_deg_bottom, lon_deg_right, lat_deg_top)

    # Project bbox to Web Mercator
    bbox_mercator = transform(transformer_to_3857.transform, bbox_lonlat)

    return bbox_mercator


def geometry_to_tile_coords(geom, bbox):
    """
    Transform geometry to tile coordinates [0, 4096].

    Args:
        geom (shapely.geometry.base.BaseGeometry): Geometry in EPSG:3857.
        bbox (shapely.geometry.box): Bounding box in EPSG:3857.

    Returns:
        shapely.geometry.base.BaseGeometry: Transformed geometry.
    """
    minx, miny, maxx, maxy = bbox.bounds

    scale_x = TILE_EXTENT / (maxx - minx)
    scale_y = TILE_EXTENT / (maxy - miny)

    def affine_transform_func(x, y):
        tx = (x - minx) * scale_x
        ty = (maxy - y) * scale_y  # Flip Y-axis
        return (tx, ty)

    transformed_geom = transform(affine_transform_func, geom)

    return transformed_geom


def create_pbf_tile(shapefile_path, z, x, y):
    """
    Create a Mapbox Vector Tile (PBF) from a shapefile for a specific tile (z/x/y).

    Args:
        shapefile_path (str): Path to the input shapefile.
        z (int): Zoom level.
        x (int): Tile x coordinate.
        y (int): Tile y coordinate.

    Returns:
        bytes: Encoded PBF tile or None if no features are present.
    """
    # Calculate bbox for the given z/x/y in Web Mercator
    tile_bbox = tile_to_bbox(z, x, y)

    layers = []

    # Read shapefile and extract features within the bbox
    with fiona.open(shapefile_path, 'r') as src:
        layer_name = src.name or 'layer'  # Default to 'layer' if name is None
        features = []

        for feat in src:
            geom = shape(feat['geometry'])

            # Reproject geometry to Web Mercator
            geom_3857 = transform(transformer_to_3857.transform, geom)

            # Check if the geometry intersects the tile bbox
            if not geom_3857.intersects(tile_bbox):
                continue

            # Clip the geometry to the tile bbox
            clipped_geom = geom_3857.intersection(tile_bbox)

            # Skip empty geometries after clipping
            if clipped_geom.is_empty:
                continue

            # Transform the geometry to tile coordinates (0, 4096)
            tile_geom = geometry_to_tile_coords(clipped_geom, tile_bbox)

            # Convert to GeoJSON-like dict
            tile_geom_geojson = mapping(tile_geom)

            # Prepare the feature for the MVT encoding
            features.append({
                "geometry": tile_geom_geojson,
                "properties": feat['properties']
            })

        if features:
            # Add the features to the layer
            layers.append({
                "name": layer_name,
                "features": features
            })

    if not layers:
        return None  # No features found in the tile

    # Encode the layers into a PBF tile
    pbf_tile = mapbox_vector_tile.encode(layers)

    return pbf_tile


@app.get("/tiles/{z}/{x}/{y}.pbf")
async def serve_pbf_tile(z: int, x: int, y: int):
    """
    Endpoint to serve a PBF tile for given z/x/y coordinates.

    Args:
        z (int): Zoom level.
        x (int): Tile x coordinate.
        y (int): Tile y coordinate.

    Returns:
        StreamingResponse: PBF tile if available, else 204 No Content.
    """
    pbf_tile = create_pbf_tile(SHAPEFILE_PATH, z, x, y)

    if pbf_tile is None:
        # Return 204 No Content if no features in tile
        raise HTTPException(status_code=204, detail="No Content")

    return StreamingResponse(
        io.BytesIO(pbf_tile),
        media_type='application/vnd.mapbox-vector-tile',
        headers={"Content-Disposition": f'inline; filename=tile_{z}_{x}_{y}.pbf'}
    )


@app.get("/")
def serve_map():
    """
    Serve the main map HTML file.

    Returns:
        FileResponse: The HTML file to display the map.
    """
    return FileResponse("static/map1.html")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
