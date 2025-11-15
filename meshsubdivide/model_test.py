#ENV: DSR
import trimesh
from util.smpl_head import smpl_head
import torch
import numpy as np



params = torch.load("smpl_params.pth")
smpl_model = smpl_head(img_res=224)
pose = params["pose"].to('cuda:0')

M = torch.tensor([[ 0.9641, -0.2234, -0.5351],
        [-0.0048,  0.7834, -0.1814],
        [ 0.8451,  0.4417,  0.4508]]).to('cuda:0')
M_ori = torch.tensor([[ 0.7641, -0.1134, -0.6351],
        [-0.0048,  0.9834, -0.1814],
        [ 0.6451,  0.1417,  0.7508]]).to('cuda:0')

pose[0,18] = (M + M_ori) / 2
print(pose[0,18])
cam = params["cam"].to('cuda:0')
shape = params["shape"].to('cuda:0')
smpl_model.to('cuda:0')

smpl_output,vertex_template,faces = smpl_model(
    rotmat=pose,
    shape=shape,
    cam=cam,
    normalize_joints2d=True,
)

vertices = smpl_output['smpl_vertices']
print("======",np.shape(vertices))

mesh = trimesh.Trimesh(vertices=vertices[0].cpu(), faces=faces, process=False)

mesh.export('smpl_male.obj')