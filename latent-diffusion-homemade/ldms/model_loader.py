from clip import CLIP
from encoder import VAE_Encoder
from decoder import VAE_Decoder
from diffusion_copy import Diffusion

import model_converter

def preload_models_from_standard_weights(ckpt_path, device):
    state_dict = model_converter.load_from_standard_weights(ckpt_path, device)

    clear_encoder = VAE_Encoder().to(device)
    blur_encoder = VAE_Encoder().to(device)
    decoder = VAE_Decoder().to(device)
    diffusion = Diffusion().to(device)

    # Fine-tuned Encoder 및 Decoder 가중치 경로
    clear_encoder_weight_path = "/home/fall/latent-diffusion-homemade/ldms/checkpoints/sharp_encoder_bilinear_epoch_100.pth"
    blur_encoder_weight_path ="/home/fall/latent-diffusion-homemade/ldms/checkpoints/blur_encoder_conv_epoch_220.pth"
    decoder_weight_path = "/home/fall/latent-diffusion-homemade/ldms/checkpoints/sharp_decoder_bilinear_epoch_100.pth"

    # decoder = VAE_Decoder().to(device)
    # decoder.load_state_dict(state_dict['decoder'], strict=True)

    # blur_encoder = VAE_Encoder().to(device)
    # clear_encoder = VAE_Encoder().to(device)
    # blur_encoder.load_state_dict(state_dict['encoder'], strict=True)
    # clear_encoder.load_state_dict(state_dict['encoder'], strict=True)


    # diffusion = Diffusion().to(device)
    # diffusion.load_state_dict(state_dict['diffusion'], strict=True)

    clip = CLIP().to(device)
    clip.load_state_dict(state_dict['clip'], strict=True)

    return {
        'clip': clip,
        'blur_encoder': blur_encoder,
        'clear_encoder' : clear_encoder,
        'decoder': decoder,
        'diffusion': diffusion,
    }