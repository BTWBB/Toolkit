import trimesh
import numpy as np

# 读取（trimesh）
src = trimesh.load("Path_to_src", process=False)   # 要贴合的 mesh
tgt = trimesh.load("Path_to_tgt", process=False)   # 目标 mesh

src_vertices = src.vertices.copy()   # (N,3)
# trimesh.proximity.closest_point 可以一次性返回每个点的最近表面点
closest_points, distance, triangle_id = trimesh.proximity.closest_point(tgt, src_vertices)
# closest_points: (N,3) -> 每个源顶点在目标表面的最近点
# triangle_id: (N,) -> 最近点所在目标三角面的索引


# 构建保留faces的新网格（顶点被替换）
projected_mesh = trimesh.Trimesh(vertices=closest_points, faces=src.faces, process=False)
print(np.shape(closest_points))
projected_mesh.export("source_projected_trimesh.obj")

D = projected_mesh.vertices
np.savez('v_template.npz', v_template=D)

