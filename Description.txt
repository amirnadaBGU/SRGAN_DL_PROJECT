Files descriptions:

data_utils.py - general helper functions
inference files:
    a. infer_diffusion.py
    b. inference_2_diffusion_models.py
    c. inference_both_models_together.py
loss functions:
    a.loss.py - loss function class
    b.loss_bit.py - loss function with VIT perceptual loss class
models:
    model.py - srgan
    model_diffusion.py - original sr3 implementation (supports being integrated in GAN)
train files:
    train.py - srgan trainer
    train_diffusion_original.py - sr3 trainer
    train_diffusion.py - diffusion gan trainer
    train_vit.py - srgan trainer with perceptual vit loss