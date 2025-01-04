# import trimesh
import pathlib
import numpy as np
import open3d as o3d

CWD = pathlib.Path(__file__).parent
# sphere_mesh = trimesh.creation.icosphere(subdivisions=2, radius=1)
# sphere_mesh.export(str(CWD / 'eyeball.obj'))

# Load eyeball and see if it works
mesh= o3d.io.read_triangle_mesh(str(CWD / 'trimesh_eyeball.obj'), True)
print(mesh)

print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh])