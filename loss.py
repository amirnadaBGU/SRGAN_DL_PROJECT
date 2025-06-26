import torch
from torch import nn
from torchvision.models.vgg import vgg16


class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss (average over all batch) - scalar
        adversarial_loss = torch.mean(1 - out_labels)
        # Perception Loss - scalar = MSE over batch features
        perception_loss = self.mse_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss - scalar - MSE over batch of images
        image_loss = self.mse_loss(out_images, target_images)
        # TV Loss - scalar
        tv_loss = self.tv_loss(out_images)

        # TODO: return also single losses for documentation:
        im_p = 1.000
        al_p = 0.001
        pl_p = 0.006
        tvl_p = 2e-8

        total_loss = im_p * image_loss + al_p * adversarial_loss + pl_p * perception_loss + tvl_p * tv_loss
        return total_loss, image_loss, adversarial_loss, perception_loss, tv_loss , im_p, al_p, pl_p, tvl_p


class TVLoss(nn.Module):
    # Total Variation loss:
    # encourages smoothness by penalizing rapid changes in pixel values between neighboring pixels in the image.
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
