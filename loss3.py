import torch
from torch import nn
from transformers import AutoImageProcessor, AutoModel



class GeneratorLoss(nn.Module):
    def __init__(self,  pixel_loss_weight=1.0, dino_loss_weight=2.0):
        super(GeneratorLoss, self).__init__()
        self.dino_model =  AutoModel.from_pretrained("facebook/dinov2-base").eval() # DINOv2 base (768-dim patch embeddings or CLS token)
        for p in self.dino_model.parameters():
            p.requires_grad = False
        self.pixel_loss = nn.L1Loss()
        self.pixel_loss_weight = pixel_loss_weight
        self.dino_loss_weight = dino_loss_weight


    def get_dino_features(self, img_tensor):
        # Assuming img_tensor is normalized to [0,1], resize to 224 and normalize for DINO
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        inputs = processor(images=[img.permute(1,2,0).cpu().numpy() for img in img_tensor], return_tensors="pt")
        with torch.no_grad():
            outputs = self.dino_model(**{k: v.to(img_tensor.device) for k, v in inputs.items()})
        return outputs.last_hidden_state[:, 0]  # CLS token

    def forward(self, out_images, target_images):
        # Pixel loss (L1)
        image_loss = self.pixel_loss(out_images, target_images)

        # DINO feature loss (cosine similarity)
        out_features = self.get_dino_features(out_images)
        target_features = self.get_dino_features(target_images)
        cosine_sim = torch.nn.functional.cosine_similarity(out_features, target_features, dim=1)
        dino_loss = 1 - cosine_sim.mean()

        total = (self.pixel_loss_weight * image_loss +
                 self.dino_loss_weight * dino_loss)
        return total


if __name__ == "__main__":
    g_loss = GeneratorLoss()
    print(g_loss)
