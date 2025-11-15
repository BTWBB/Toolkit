from trimesh.remesh import subdivide
import trimesh
import numpy as np
from trimesh import grouping
from trimesh.geometry import faces_to_edges
import pickle
import scipy.sparse as sp
import chumpy as ch

def subdivide_with_origin(vertices,
                          faces,
                          face_index=None,
                          vertex_attributes=None):
    """
    Subdivide a triangular mesh while tracking each new vertex's origin.

    If a vertex is original, its origin pair is (i, i).
    If a vertex is created from an edge (i, j), its origin pair is (i, j).

    Parameters
    ----------
    vertices : (n, 3) float
        Original vertex coordinates
    faces : (m, 3) int
        Face vertex indices
    face_index : (k,) int or None
        Subset of faces to subdivide
    vertex_attributes : dict or None
        Optional per-vertex attribute dict

    Returns
    ----------
    new_vertices : (q, 3) float
        Subdivided vertex coordinates
    new_faces : (p, 3) int
        Updated face indices
    origin_pairs : (q, 2) int
        For each vertex, the pair of source vertex indices
    new_attributes : dict (optional)
        Updated vertex attributes (if provided)
    """
    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    faces_subset = faces[face_index]

    # Find all edges of selected faces
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    unique, inverse = grouping.unique_rows(edges)

    # Compute midpoints for each unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    # Build new faces
    f = np.column_stack([
        faces_subset[:, 0], mid_idx[:, 0], mid_idx[:, 2],
        mid_idx[:, 0], faces_subset[:, 1], mid_idx[:, 1],
        mid_idx[:, 2], mid_idx[:, 1], faces_subset[:, 2],
        mid_idx[:, 0], mid_idx[:, 1], mid_idx[:, 2]
    ]).reshape((-1, 3))

    # Replace old faces and append new ones
    new_faces = np.vstack((faces, f[len(face_index):]))
    new_faces[face_index] = f[:len(face_index)]

    # Combine vertices
    new_vertices = np.vstack((vertices, mid))

    # === Origin pair tracking ===
    n_old = len(vertices)
    origin_pairs = np.zeros((len(new_vertices), 2), dtype=int)

    # Original vertices: (i, i)
    origin_pairs[:n_old, 0] = np.arange(n_old)
    origin_pairs[:n_old, 1] = np.arange(n_old)

    # New midpoints: from edge (i, j)
    origin_pairs[n_old:] = edges[unique]

    # === Optional: attribute interpolation ===
    new_attributes = None
    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            attr_tris = values[faces_subset]
            attr_mid = np.vstack([
                attr_tris[:, g, :].mean(axis=1)
                for g in [[0, 1], [1, 2], [2, 0]]
            ])
            attr_mid = attr_mid[unique]
            new_attributes[key] = np.vstack((values, attr_mid))

        return new_vertices, new_faces, origin_pairs, new_attributes

    return new_vertices, new_faces, origin_pairs

def compute_high_res_J_regressor(J_regressor_low, V_high, edge_pairs):
    """
    根据细分后的顶点对应关系计算新的 J_regressor。

    Parameters
    ----------
    V_low : (N, 3)
        原始网格顶点
    J_regressor_low : (J, N)
        原始 J_regressor，可以是 numpy.ndarray 或 scipy.sparse.csc_matrix
    V_high : (N_high, 3)
        细分后的顶点
    edge_pairs : (N_high, 2)
        每个高分辨率顶点来源的原始顶点索引 (v0, v1)
        - 若为原始顶点，则 (v0, v1) = (i, i)
        - 若为细分中点，则 (v0, v1) = (a, b)
    """
    # 如果是稀疏矩阵，转为稠密再计算
    if sp.issparse(J_regressor_low):
        J_regressor_low = J_regressor_low.toarray()

    n_joints, n_low = J_regressor_low.shape
    n_high = len(V_high)
    J_regressor_high = np.zeros((n_joints, n_high), dtype=J_regressor_low.dtype)

    for i in range(n_high):
        v0, v1 = edge_pairs[i]
        if v0 == v1:
            J_regressor_high[:, i] = J_regressor_low[:, v0]
        else:
            J_regressor_high[:, i] = 0.5 * (J_regressor_low[:, v0] + J_regressor_low[:, v1])

    # 归一化并转回稀疏格式
    sums = J_regressor_high.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1.0  # 避免除零
    J_regressor_high /= sums
    J_regressor_high = sp.csc_matrix(J_regressor_high)
    return J_regressor_high

def compute_high_res_shapedirs(shapedirs_low, edge_pairs):
    """
    Generate high-resolution shapedirs for subdivided mesh.
    
    Parameters
    ----------
    shapedirs_low : (N_low, 3, n_shape) numpy.ndarray
    edge_pairs : (N_high, 2) 每个新顶点来源原顶点索引
    
    Returns
    -------
    shapedirs_high : (N_high, 3, n_shape)
    """
    shapedirs_low = shapedirs_low.r
    N_high = len(edge_pairs)
    n_shape = shapedirs_low.shape[2]
    shapedirs_high = np.zeros((N_high, 3, n_shape), dtype=shapedirs_low.dtype)
    
    for i, (v0, v1) in enumerate(edge_pairs):
        if v0 == v1:
            shapedirs_high[i] = shapedirs_low[v0]
        else:
            shapedirs_high[i] = 0.5 * (shapedirs_low[v0] + shapedirs_low[v1])
    print(np.shape(shapedirs_high))
    shapedirs_high_ch = ch.Ch(shapedirs_high)
    return shapedirs_high_ch

def compute_high_res_posedirs(posedirs_low, edge_pairs):
    """
    Generate high-resolution posedirs for subdivided mesh.
    
    Parameters
    ----------
    posedirs_low : (N_low, 3, 207)
    edge_pairs : (N_high, 2) 每个新顶点来源原顶点索引
    
    Returns
    -------
    posedirs_high : (N_high, 3, 207)
    """
    N_high = len(edge_pairs)
    n_pose = posedirs_low.shape[2]
    posedirs_high = np.zeros((N_high, 3, n_pose), dtype=posedirs_low.dtype)
    
    for i, (v0, v1) in enumerate(edge_pairs):
        if v0 == v1:
            posedirs_high[i] = posedirs_low[v0]
        else:
            posedirs_high[i] = 0.5 * (posedirs_low[v0] + posedirs_low[v1])
    
    return posedirs_high

def compute_high__weights(weights_low, edge_pairs, normalize=True):
    """
    Generate high-resolution skinning weights for subdivided mesh.
    
    Parameters
    ----------
    weights_low : (N_low, 24)
    edge_pairs : (N_high, 2) 每个新顶点来源原顶点索引
    normalize : bool, 是否归一化行和
    
    Returns
    -------
    weights_high : (N_high, 24)
    """
    N_high = len(edge_pairs)
    n_joints = weights_low.shape[1]
    weights_high = np.zeros((N_high, n_joints), dtype=weights_low.dtype)
    
    for i, (v0, v1) in enumerate(edge_pairs):
        if v0 == v1:
            weights_high[i] = weights_low[v0]
        else:
            weights_high[i] = 0.5 * (weights_low[v0] + weights_low[v1])
    
    if normalize:
        row_sums = weights_high.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # 避免除零
        weights_high /= row_sums
    
    return weights_high

def compute_high_subdivide_J_extra(J_extra_low, edge_pairs, normalize=True):
    """
    将 J_regressor_extra 扩展到高分辨率顶点。
    
    Parameters
    ----------
    J_extra_low : (n_joints, N_low)
    edge_pairs : (N_high, 2) 每个新顶点来源原顶点索引
    normalize : bool, 是否归一化每行
    
    Returns
    -------
    J_extra_high : (n_joints, N_high)
    """
    n_joints, N_low = J_extra_low.shape
    N_high = len(edge_pairs)
    
    J_extra_high = np.zeros((n_joints, N_high), dtype=J_extra_low.dtype)
    
    for i, (v0, v1) in enumerate(edge_pairs):
        if v0 == v1:
            J_extra_high[:, i] = J_extra_low[:, v0]
        else:
            J_extra_high[:, i] = 0.5 * (J_extra_low[:, v0] + J_extra_low[:, v1])
    
    if normalize:
        row_sums = J_extra_high.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        J_extra_high /= row_sums
    
    return J_extra_high

# 读取原始 SMPL 模板
mesh_low = trimesh.load('smpl.obj', process=False)
V_low = mesh_low.vertices
F_low = mesh_low.faces
print(V_low.shape, F_low.shape)
# (6890, 3), (13776, 3)


# 细分一次：每个三角形分成4个
V_high, F_high, origin_pairs = subdivide_with_origin(V_low, F_low)
mesh_high = trimesh.Trimesh(vertices=V_high, faces=F_high, process=False)
mesh_high.export('smpl_subdiv.obj')
print(V_high.shape, F_high.shape)
print("origin_pairs shape:",np.shape(origin_pairs))


# 处理 SMPL_NEUTRAL
with open('Path_to_smpl/SMPL_NEUTRAL.pkl', 'rb') as f:
    data = pickle.load(f,encoding='latin1')
data["v_template"] = V_high
data["f"] = F_high
data["J_regressor"] = compute_high_res_J_regressor(data["J_regressor"], V_high, origin_pairs)
data["shapedirs"] = compute_high_res_shapedirs(data["shapedirs"], origin_pairs)
data["posedirs"] = compute_high_res_posedirs(data["posedirs"], origin_pairs)
data["weights"] = compute_high__weights(data["weights"], origin_pairs, normalize=True)

# 保存 SMPL_NEUTRAL.pkl
new_file_path = './SMPL_NEUTRAL_HIGHRES.pkl'

with open(new_file_path, 'wb') as f:
    pickle.dump(data, f, protocol=4)  # protocol=4 支持大文件

# 处理 J_regressor_extra
J_extra = np.load('Path_to_save/J_regressor_extra.npy')
J_extra_high = compute_high_subdivide_J_extra(J_extra, origin_pairs, normalize=True)
np.save("J_regressor_extra_highres.npy", J_extra_high)
np.save("origin_pairs.npy", origin_pairs)

# print(" -", type(origin_pairs))
# print(" -", np.shape(origin_pairs))
# print(" -",J_extra_high.sum(axis=1))