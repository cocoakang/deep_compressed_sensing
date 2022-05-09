from shutil import which
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from SENSING_NET import SENSING_NET
from GENERATOR_NET import GENERATOR_NET

class TRAINING_NET(nn.Module):
    def __init__(self,args):
        super(TRAINING_NET,self).__init__()

        self.loss_lambda = args["loss_lambda"]

        self.latent_code_len = args["latent_code_len"]
        self.pattern_num = args["pattern_num"]
        self.project_method = args["project_method"]
        self.max_inner_setp = args["max_inner_setp"]
        self.z_step_size = args["z_step_size"]

        self.sensing_net = SENSING_NET(args)
        self.generator_net = GENERATOR_NET(args)

        self.l2_loss_fn = torch.nn.MSELoss(reduction='mean')
        self.l1_loss_fn = torch.nn.L1Loss(reduction='mean')
        # self.tv_loss = TVLoss()
        # # self.sinogram_2d_refine_net = CYCLOPS_2DSINOGRAM_REFINE_NET(args)
        # self.vgg_loss = VGGPerceptualLoss(resize=False)
    
    def _project_z(self,z, project_method='clip'):
        if project_method == 'norm':
            z_p = torch.nn.functional.normalize(z, dim=1)
        elif project_method == 'clip':
            z_p = torch.clamp(z, -1.0, 1.0)
        else:
            raise ValueError('Unknown project_method: {}'.format(project_method))
        return z_p

    def _get_measurement_error(self, target_img, sample_img):
        """Compute the measurement error of sample images given the targets."""

        m_targets = self.sensing_net(target_img)
        m_samples = self.sensing_net(sample_img)

        return torch.sum(torch.square(m_targets - m_samples), dim=1)

    def _get_rip_loss(self, img1, img2):
        r"""Compute the RIP loss from two images.

        The RIP loss: (\sqrt{F(x_1 - x_2)^2} - \sqrt{(x_1 - x_2)^2})^2

        Args:
        img1: an image (x_1), 4D tensor of shape [batch_size, W, H, C].
        img2: an other image (x_2), 4D tensor of shape [batch_size, W, H, C].
        """

        m1 = self.sensing_net(img1)
        m2 = self.sensing_net(img2)

        img1_flatten = torch.flatten(img1,start_dim=1)
        img2_flatten = torch.flatten(img2,start_dim=1)

        img_diff_norm = torch.norm(img1_flatten - img2_flatten, dim=-1)
        m_diff_norm = torch.norm(m1 - m2, dim=-1)

        return torch.square(img_diff_norm - m_diff_norm)

    def forward(self,data_pkg,return_recon=False):
        sinogram_3d_img = data_pkg["sinogram_3d_img"]#(batch_size,1,height,width)
        init_z = data_pkg["init_z"]#(batch_size,latent_code_len)
        
        batch_size = sinogram_3d_img.shape[0]
        
        #---optimize_z
        init_z = self._project_z(init_z,self.project_method)
        z = init_z.requires_grad_(True)
        # print("optimizing z...")
        for which_inner_loop in range(self.max_inner_setp):
            # print("inner loop step:{}".format(which_inner_loop))
            loop_samples = self.generator_net(z)
            gen_loss = self._get_measurement_error(sinogram_3d_img, loop_samples)
            gen_loss = torch.sum(gen_loss)
            # gen_loss.backward()
            z_grad = torch.autograd.grad(gen_loss, z,create_graph=True)[0]
            z = z - self.z_step_size * z_grad
            z = self._project_z(z, self.project_method)
        optimised_z = z
        sinogram_3d_img_nn = self.generator_net(optimised_z)
        initial_samples = self.generator_net(init_z)

        #---compute generator loss
        generator_loss = torch.mean(self._get_measurement_error(sinogram_3d_img, sinogram_3d_img_nn))

        #--- compute the RIP loss
        r1 = self._get_rip_loss(sinogram_3d_img_nn, initial_samples)
        r2 = self._get_rip_loss(sinogram_3d_img_nn, sinogram_3d_img)
        r3 = self._get_rip_loss(initial_samples, sinogram_3d_img)
        rip_loss = torch.mean((r1 + r2 + r3) / 3.0)
        total_loss = generator_loss + rip_loss

        loss_pkg = {
            "total_loss":total_loss,
            "r1":r1,
            "r2":r2,
            "r3":r3,
            "rip_loss":rip_loss,
            "generator_loss":generator_loss
        }

        if not return_recon:
            return loss_pkg
        else:
            result_pkg = {
                # "volume_data_nn":volume_nn,
                # "sinogram_2d_nn":sinogram_2d_nn,
                # "sinogram_2d_refined":sinogram_2d_refined,
                # "sinogram_multiplexed":sinogram_multiplexed,
                # "sinogram_3d_nn":sinogram_3d_nn,
                "sinogram_3d_img":sinogram_3d_img,
                "sinogram_3d_img_nn":sinogram_3d_img_nn,
                # "feature_pkg":feature_pkg
            }
            return loss_pkg,result_pkg