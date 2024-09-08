import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math
import numpy as np

class l1_loss(nn.Module):
    def __init__(self):
        super(l1_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        loss = torch.nn.functional.l1_loss(depth_est, depth_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class l2_loss(nn.Module):
    def __init__(self):
        super(l2_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        loss = torch.nn.functional.mse_loss(depth_est, depth_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class smoothl1_loss(nn.Module):
    def __init__(self):
        super(smoothl1_loss, self).__init__()

    def forward(self, depth_est, depth_gt, mask):
        loss = torch.nn.functional.smooth_l1_loss(depth_est, depth_gt, reduction='none')
        loss = mask * loss
        loss = torch.sum(loss) / torch.sum(mask)
        return loss

class imgrad_loss(nn.Module):
    def __init__(self):
        super(imgrad_loss, self).__init__()

    def imgrad(self, img):
        img = torch.mean(img, 1, True)
        fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv1.weight = nn.Parameter(weight)
        grad_x = conv1(img)

        fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)

        if img.is_cuda:
            weight = weight.cuda()

        conv2.weight = nn.Parameter(weight)
        grad_y = conv2(img)
        return grad_y, grad_x

    def forward(self, pred, gt, mask=None):
        N,C,_,_ = pred.size()
        grad_y, grad_x = self.imgrad(pred)
        grad_y_gt, grad_x_gt = self.imgrad(gt)
        grad_y_diff = torch.abs(grad_y - grad_y_gt)
        grad_x_diff = torch.abs(grad_x - grad_x_gt)
        if mask is not None:
            grad_y_diff[~mask] = 0.1*grad_y_diff[~mask]
            grad_x_diff[~mask] = 0.1*grad_x_diff[~mask]
        return (torch.mean(grad_y_diff) + torch.mean(grad_x_diff))


class smoothness_loss_func(nn.Module):
    def __init__(self):
        super(smoothness_loss_func, self).__init__()
    
    def gradient_yx(self, T):
        '''
        Computes gradients in the y and x directions

        Arg(s):
            T : tensor
                N x C x H x W tensor
        Returns:
            tensor : gradients in y direction
            tensor : gradients in x direction
        '''

        dx = T[:, :, :, :-1] - T[:, :, :, 1:]
        dy = T[:, :, :-1, :] - T[:, :, 1:, :]
        return dy, dx
    
    def forward(self, predict, image):
        predict_dy, predict_dx = self.gradient_yx(predict)
        image_dy, image_dx = self.gradient_yx(image)

        # Create edge awareness weights
        weights_x = torch.exp(-torch.mean(torch.abs(image_dx), dim=1, keepdim=True))
        weights_y = torch.exp(-torch.mean(torch.abs(image_dy), dim=1, keepdim=True))

        smoothness_x = torch.mean(weights_x * torch.abs(predict_dx))
        smoothness_y = torch.mean(weights_y * torch.abs(predict_dy))
        
        return smoothness_x + smoothness_y
