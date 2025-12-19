import requests
import numpy as np
from stl import mesh
import math
from tqdm import tqdm

# --- CONFIGURATION & PARAMETERS ---
API_KEY = "API_KEY_HERE Which you can get from https://portal.opentopography.org/user/register"
LAT = 31.57        # Center Latitude
LON = 117.56      # Center Longitude
GEO_WIDTH = 42000     # Real-world width in meters
MODEL_WIDTH = 120    # STL width in mm
BASE_HEIGHT = 3      # Base thickness in mm
Z_SCALE = 12        # Vertical exaggeration

def get_elevation_data(lat, lon, width_m):
    # Adjust for longitudinal shrinkage based on latitude
    lat_deg_range = (width_m / 111111.0) / 2
    lon_deg_range = (width_m / (111111.0 * math.cos(math.radians(lat)))) / 2
    
    params = {
        'demtype': 'SRTMGL1',
        'south': lat - lat_deg_range,
        'north': lat + lat_deg_range,
        'west': lon - lon_deg_range,
        'east': lon + lon_deg_range,
        'outputFormat': 'AAIGrid',
        'API_Key': API_KEY
    }
    
    url = "https://portal.opentopography.org/API/globaldem"
    
    print(f"Connecting to OpenTopography...")
    # Using stream=True to track download progress
    response = requests.get(url, params=params, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code} - {response.text}")

    total_size = int(response.headers.get('content-length', 0))
    content = []
    
    with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading Elevation Data") as pbar:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                content.append(chunk)
                pbar.update(len(chunk))
    
    full_text = b"".join(content).decode('utf-8')
    lines = full_text.splitlines()
    data = np.loadtxt(lines[6:])
    return data

def create_stl(data, model_width, base_h, z_scale):
    rows, cols = data.shape
    cell_size = model_width / max(rows, cols)
    
    min_z = np.min(data)
    # Scale elevation relative to the horizontal resolution (30m SRTM)
    # This keeps the topography proportional to the real world
    z_data = (data - min_z) * (model_width / (cols * 30)) * z_scale
    z_data += base_h

    vertices = []
    # Using tqdm for the mesh generation as well, as it can be CPU intensive
    print("Generating 3D mesh vertices...")
    for r in range(rows):
        for c in range(cols):
            vertices.append([c * cell_size, (rows - r) * cell_size, z_data[r, c]])

    # Bottom floor vertices
    for r in range(rows):
        for c in range(cols):
            vertices.append([c * cell_size, (rows - r) * cell_size, 0])

    vertices = np.array(vertices)
    faces = []
    bottom_start_idx = rows * cols

    def idx(r, c, is_bottom=False):
        return (bottom_start_idx if is_bottom else 0) + (r * cols + c)

    print("Stitching triangular faces...")
    with tqdm(total=(rows-1)*(cols-1), desc="Building Mesh") as pbar:
        for r in range(rows - 1):
            for c in range(cols - 1):
                # Top Surface
                faces.append([idx(r, c), idx(r, c+1), idx(r+1, c)])
                faces.append([idx(r+1, c), idx(r, c+1), idx(r+1, c+1)])
                # Bottom Surface
                faces.append([idx(r, c, True), idx(r+1, c, True), idx(r, c+1, True)])
                faces.append([idx(r+1, c, True), idx(r+1, c+1, True), idx(r, c+1, True)])
                pbar.update(1)

    # Side Walls (Watertight closure)
    for r in range(rows - 1):
        faces.append([idx(r, 0), idx(r+1, 0), idx(r, 0, True)])
        faces.append([idx(r+1, 0), idx(r+1, 0, True), idx(r, 0, True)])
        faces.append([idx(r, cols-1), idx(r, cols-1, True), idx(r+1, cols-1)])
        faces.append([idx(r+1, cols-1), idx(r, cols-1, True), idx(r+1, cols-1, True)])

    for c in range(cols - 1):
        faces.append([idx(0, c), idx(0, c, True), idx(0, c+1)])
        faces.append([idx(0, c+1), idx(0, c, True), idx(0, c+1, True)])
        faces.append([idx(rows-1, c), idx(rows-1, c+1), idx(rows-1, c, True)])
        faces.append([idx(rows-1, c+1), idx(rows-1, c+1, True), idx(rows-1, c, True)])

    terrain_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            terrain_mesh.vectors[i][j] = vertices[f[j], :]

    return terrain_mesh

# --- EXECUTION ---
try:
    elev_data = get_elevation_data(LAT, LON, GEO_WIDTH)
    my_stl = create_stl(elev_data, MODEL_WIDTH, BASE_HEIGHT, Z_SCALE)
    
    output_name = f"terrain_{LAT}_{LON}.stl"
    my_stl.save(output_name)
    print(f"\nSaved: {output_name}")
except Exception as e:
    print(f"\nAn error occurred: {e}")