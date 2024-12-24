import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(
    prompt,
    uncond_prompt=None,
    input_image=None,
    strength=0.8, # 얼마나 input image에 집중할지, strength가 클수록 noise가 강해지므로 창의적인 이미지 생성
    do_cfg=True,
    cfg_scale=7.5,
    sampler_name="ddpm",
    n_inference_steps=50,
    models={},
    seed=None,
    device=None,
    idle_device=None,
    tokenizer=None,
):
    with torch.no_grad():
        if not 0 < strength <= 1:
            raise ValueError("strength must be between 0 and 1")
        
        # move to cpu
        if idle_device:
            to_idle = lambda x: x.to(idle_device)
        else:
            to_idle = lambda x: x

        # Initialize random number generator according to the seed specified
        generator = torch.Generator(device=device)
        if seed is None:
            generator.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)
        
        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if do_cfg:
            # Convert into a list of length Seq_Len=77
            cond_tokens = tokenizer.batch_encode_plus(
                [prompt], padding="max_length", max_length=77
            ).input_ids
            # (Batch_Size, Seq_Len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            cond_context = clip(cond_tokens)
            # Convert into a list of length Seq_Len=77
            # uncond_tokens = tokenizer.batch_encode_plus(
            #     [uncond_prompt], padding="max_length", max_length=77
            # ).input_ids
            # # (Batch_Size, Seq_Len)
            # uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
            # uncond_context = clip(uncond_tokens)
            # (Batch_Size, Seq_Len, Dim) + (Batch_Size, Seq_Len, Dim) -> (2 * Batch_Size, Seq_Len, Dim)
                    # Image-based context
            if input_image:

                encoder = models["encoder"]
                encoder.to(device)

                input_image_tensor = preprocess_image(input_image, device)
                zero_noise = torch.zeros(latents_shape, device=device)
                image_latents = encoder(input_image_tensor, zero_noise)

                # Flatten and pad the image_latents to match context dimensions
                image_context = image_latents.flatten(2)
                batch_size, seq_len, dim = cond_context.shape
                image_context = torch.nn.functional.pad(
                    image_context, (0, dim - image_context.shape[2], 0, seq_len - image_context.shape[1])
                )
            else:
                raise ValueError("Input image is required when using image context.")

            if do_cfg:
                # Replace uncond_prompt with image_context for unconditioned context
                context = torch.cat([cond_context, image_context])
            else:
                context = cond_context
            print(f"cond_context shape: {cond_context.shape}")
            print(f"image_context shape: {image_context.shape}")
            print(f"final context shape: {context.shape}")
            # context = torch.cat([cond_context, uncond_context])
        #     print(f"cond_context shape: {cond_context.shape}")
        #     print(f"context shape after concat: {context.shape}")
        # else:
        #     # Convert into a list of length Seq_Len=77
        #     tokens = tokenizer.batch_encode_plus(
        #         [prompt], padding="max_length", max_length=77
        #     ).input_ids
        #     # (Batch_Size, Seq_Len)
        #     tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        #     # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
        #     context = clip(tokens)
        ## 이미지 인코딩 및 컨텍스트와 결합
        # if input_image:
        #     encoder = models["encoder"]
        #     encoder.to(device)

        #     input_image_tensor = preprocess_image(input_image, device)
        #     zero_noise = torch.zeros(latents_shape, device=device)
        #     image_latents = encoder(input_image_tensor, zero_noise)

        #     # Merge with text-based context
        #     image_context = image_latents.flatten(2)
        #     print(f"input_image_tensor shape after preprocess: {input_image_tensor.shape}")
        #     print(f"image_latents shape: {image_latents.shape}")

        #     linear_layer = torch.nn.Linear(image_context.shape[-1], context.shape[-1], device=device)
        #     image_context = linear_layer(image_context)
        #     print(f"image_context shape after flatten: {image_context.shape}")

        #     if image_context.shape[1] < context.shape[1]:
        #         padding = context.shape[1] - image_context.shape[1]
        #         image_context = torch.nn.functional.pad(image_context, (0, 0, 0, padding))
        #     elif image_context.shape[1] > context.shape[1]:
        #         image_context = image_context[:, :context.shape[1], :]
        #     print(f"image_context shape after adjustment: {image_context.shape}")

        #     if do_cfg:
        #         image_context = image_context.repeat(2, 1, 1)  # Repeat for conditioned & unconditioned
            
        #     context = torch.cat([context, image_context], dim=-1)


        #     print(f"context shape after combining image_context: {context.shape}")

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_timesteps(n_inference_steps)
        else:
            raise ValueError("Unknown sampler value %s. ")


        # Initialize latents
        if input_image:
            input_image_tensor = preprocess_image(input_image, device)
            encoder_noise = torch.rand(latents_shape, generator=generator, device=device)
            image_latents = encoder(input_image_tensor, encoder_noise)
            latents = sampler.add_noise(image_latents, sampler.timesteps[0])
            print(f"latents shape after adding noise: {latents.shape}")
        else:
            latents = torch.randn(latents_shape, generator=generator, device=device)
            
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)
            
            # (Batch_Size, 4, Latents_Height, Latents_Width)
            model_input = latents
            print(f"latents shape after adding noise: {latents.shape}")
            
            if do_cfg:
                # (Batch_Size, 4, Latents_Height, Latents_Width) -> (2 * Batch_Size, 4, Latents_Height, Latents_Width)
                model_input = model_input.repeat(2, 1, 1, 1)

            # model_output is the predicted noise
            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_prompt, output_image = model_output.chunk(2)
                model_output = cfg_scale * (output_prompt - output_image) + output_image

            # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 4, Latents_Height, Latents_Width)
            latents = sampler.step(timestep, latents, model_output)

        decoder = models["decoder"]
        decoder.to(device)
        # (Batch_Size, 4, Latents_Height, Latents_Width) -> (Batch_Size, 3, Height, Width)
        images = decoder(latents)
        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Height, Width, Channel)
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]


    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    return x

# def get_time_embedding(timestep):
#     # Shape: (160,)
#     freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160) 
#     # Shape: (1, 160)
#     x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
#     # Shape: (1, 160 * 2)
#     return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def get_time_embedding(timestep):
    # Shape: (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160).to(timestep.device)
    
    # timestep이 배치 크기를 가지는 텐서일 경우, 각 timestep에 대해 임베딩 계산
    if timestep.dim() > 0:
        timestep = timestep.float().unsqueeze(-1)  # (Batch_Size, 1)
        x = timestep * freqs  # Broadcasting to shape (Batch_Size, 160)
    else:
        x = torch.tensor([timestep], dtype=torch.float32, device=timestep.device)[:, None] * freqs[None]

    # Shape: (Batch_Size, 320) 또는 (1, 320)로 반환
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def preprocess_image(image, device):
    # Ensure the image is in RGB format
    image = image.convert("RGB")

    # 이미지 크기 가져오기
    width, height = image.size

    # 중앙 Crop 좌표 계산
    left = (width - WIDTH) // 2
    top = (height - HEIGHT) // 2
    right = left + WIDTH
    bottom = top + HEIGHT

    # 중앙 Crop 수행
    image = image.crop((left, top, right, bottom))

    # 이미지 배열 변환
    image = np.array(image)

    # 텐서 변환
    image = torch.tensor(image, dtype=torch.float32, device=device)

    # 픽셀 값 범위 변환 (0~255 → -1~1)
    image = rescale(image, (0, 255), (-1, 1))

    # 배치 및 채널 순서 변경
    image = image.unsqueeze(0).permute(0, 3, 1, 2)  # (Batch_Size, Channels, Height, Width)
    
    return image

class FeatureLinear(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)