## ldm3d inpainting
[Ldm3d](https://arxiv.org/pdf/2305.10853.pdf) generates an image and a depth map from a given text prompt unlike the existing text-to-image diffusion models such as Stable Diffusion which only generates an image.
This is a modified version that adds inpainting capabilities, so you can fill in Depth and Color in images.
The Unet was re-trained with an extra 5 channels (1 for the mask, 4 for the masked depth/color joint image).

# Example
Original Image+Depth: \
```
a photo of an astronaut riding a horse on mars
```
![Original Image+Depth](https://raw.githubusercontent.com/pablodawson/ldm3d-inpainting/main/github_misc/og.png) \
Inpainted Image+Depth: \
```
a photo of an astronaut riding a pig
```
![Inpainted Image+Depth](https://raw.githubusercontent.com/pablodawson/ldm3d-inpainting/main/github_misc/gen.png)\

Mask: \
![Mask](https://raw.githubusercontent.com/pablodawson/ldm3d-inpainting/main/github_misc/mask.png)\

## Usage
Use the *StableDiffusionLDM3DInpaintPipeline* class in this [diffusers](https://github.com/pablodawson/diffusers) fork

```
unet = UNet2DConditionModel.from_pretrained("pablodawson/ldm3d-inpainting", cache_dir="cache" )
pipe = StableDiffusionLDM3DInpaintPipeline.from_pretrained("Intel/ldm3d-4c", cache_dir="cache" )
pipe = pipe.to("cuda")
prompt = "a photo of an astronaut riding a horse on mars"
input_image = Image.open("input_image.jpg")
depth_image = Image.open("depth_image.png")
mask_image = Image.open("mask_image.png")
output = pipe(prompt=prompt, image=input_image, mask_image=mask_image, depth_image=depth_image)
```

## Training

This was the training script used:

```
accelerate launch train.py --mixed_precision="fp16"  --use_ema   --resolution=512 --random_flip   --train_batch_size=1   --gradient_accumulation_steps=4   --gradient_checkpointing   --max_train_steps=15000   --learning_rate=1e-05   --max_grad_norm=1   --checkpointing_steps=1000  --lr_scheduler="constant" --lr_warmup_steps=0
```
