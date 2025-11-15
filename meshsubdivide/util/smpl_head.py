import torch
import numpy as np
import torch.nn as nn

from smplx import SMPL as _SMPL
from smplx.body_models import SMPLOutput
from smplx.lbs import vertices2joints

from . import config, constants


class SMPL(_SMPL):
    """ Extension of the official SMPL implementation to support more joints """

    def __init__(self, *args, **kwargs):
        super(SMPL, self).__init__(*args, **kwargs)
        joints = [constants.JOINT_MAP[i] for i in constants.JOINT_NAMES]
        J_regressor_extra = np.load(config.JOINT_REGRESSOR_TRAIN_EXTRA)
        self.register_buffer('J_regressor_extra', torch.tensor(J_regressor_extra, dtype=torch.float32))
        self.joint_map = torch.tensor(joints, dtype=torch.long)

    def forward(self, *args, **kwargs):
        kwargs['get_skin'] = True
        smpl_output = super(SMPL, self).forward(*args, **kwargs)
        extra_joints = vertices2joints(self.J_regressor_extra, smpl_output.vertices)
        joints = torch.cat([smpl_output.joints, extra_joints], dim=1)
        joints = joints[:, self.joint_map, :]
        output = SMPLOutput(vertices=smpl_output.vertices,
                             global_orient=smpl_output.global_orient,
                             body_pose=smpl_output.body_pose,
                             joints=joints,
                             betas=smpl_output.betas,
                             full_pose=smpl_output.full_pose)
        return output


class smpl_head(nn.Module):
    def __init__(self, focal_length=5000., img_res=224):
        super(smpl_head, self).__init__()
        self.smpl = SMPL(config.SMPL_MODEL_DIR, create_transl=False)
        self.faces = torch.from_numpy(self.smpl.faces.astype('int32')[None])
        self.focal_length = focal_length
        self.img_res = img_res

    def forward(self, rotmat, shape, cam=None, normalize_joints2d=False):
        '''
        :param rotmat: rotation in euler angles format (N,J,3,3)
        :param shape: smpl betas
        :param cam: weak perspective camera
        :param normalize_joints2d: bool, normalize joints between -1, 1 if true
        :return: dict with keys 'vertices', 'joints3d', 'joints2d' if cam is True
        '''
        smpl_output = self.smpl(
            betas=shape,
            body_pose=rotmat[:, 1:].contiguous(),
            global_orient=rotmat[:, 0].unsqueeze(1).contiguous(),
            pose2rot=False,
        )

        vertex_template=self.smpl.v_template.clone()
        faces=self.smpl.faces

        output = {
            'smpl_vertices': smpl_output.vertices,
            'smpl_joints3d': smpl_output.joints,
        }

        return output,vertex_template,faces
