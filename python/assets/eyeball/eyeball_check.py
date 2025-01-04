# import trimesh
import pathlib
import numpy as np
import open3d as o3d

CWD = pathlib.Path(__file__).parent

# Load eyeball and see if it works
mesh= o3d.io.read_triangle_mesh(str(CWD / 'trimesh_eyeball.obj'), True)
print(mesh)

print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh])