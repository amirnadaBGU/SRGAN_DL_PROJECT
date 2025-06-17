import torch
import torch.nn as nn
from timm import create_model


class GeneratorLoss(nn.Module):
    def __init__(self, vit_model_name='vit_base_patch16_224', feature_layer=5, pretrained=True, device='cuda'):
        super(GeneratorLoss, self).__init__()
        self.device = device

        # טוען ViT מאומן מ־timm
        vit = create_model(vit_model_name, pretrained=pretrained)
        vit.eval()
        vit.to(self.device)

        # קובע את השכבה שמתוכה נוציא את ה־features
        # אפשר לשנות לפי ניסוי. 5 = אחרי כמה Transformer blocks.
        self.feature_layer = feature_layer
        self.vit = vit

        # הקפאת כל הפרמטרים של ה־ViT
        for param in self.vit.parameters():
            param.requires_grad = False

        # Normalization כפי שמתאים ל־ViT שאומן על ImageNet
        self.register_buffer('mean', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1))

        self.mse_loss = nn.MSELoss()
        self.tv_loss = TVLoss()

    def _extract_features(self, x):
        
        # ViT מקבל קלט בגודל 224x224, נבצע resize
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # נורמליזציה
        x = (x - self.mean) / self.std

        # הוצאת features מתוך ViT
        # לוקחים את הטוקן של ה־[CLS] מתוך השכבה המבוקשת
        features = None
        hooks = []

        def hook_fn(module, input, output):
            nonlocal features
            features = output

        # חיבור hook לשכבה המבוקשת
        handle = self.vit.blocks[self.feature_layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = self.vit(x)
        handle.remove()

        return features

    def forward(self, out_labels, out_images, target_images):
        # Adversarial Loss
        adversarial_loss = torch.mean(1 - out_labels)

        # Feature-based Perceptual Loss
        out_features = self._extract_features(out_images)
        target_features = self._extract_features(target_images)
        perceptual_loss = self.mse_loss(out_features, target_features)

        # Image pixel-wise MSE Loss
        image_loss = self.mse_loss(out_images, target_images)

        # Total Variation Loss for smoothness
        tv_loss = self.tv_loss(out_images)

        total_loss = image_loss + 0.001 * adversarial_loss + 0.006 * perceptual_loss + 2e-8 * tv_loss
        return total_loss


class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=1):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size(0)
        h_x = x.size(2)
        w_x = x.size(3)
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size

    @staticmethod
    def tensor_size(t):
        return t.size(1) * t.size(2) * t.size(3)
