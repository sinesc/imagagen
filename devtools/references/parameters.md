# woman-at-sunset.png

prompt: cinematic film still woman walking at the beach, summer outfit, sunset, beautiful sky . shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy, highly dynamic shot, motion blur, strong depth of field, smooth blurry background, gaussian blur
negative:
seed: 1000164628
model: epicrealismXL11.0.safetensors
CFG: 7.5
steps: 20
sampler: DPM++ 2M
scheduler: Polyexponential
resolution: 1600x512
SAG: false
user description: woman walking on the beach at sunset, viewed from behind, her right hand is distorted/blurry

# woman-at-sunset-sag.png

prompt: cinematic film still woman walking at the beach, summer outfit, sunset, beautiful sky . shallow depth of field, vignette, highly detailed, high budget Hollywood movie, bokeh, cinemascope, moody, epic, gorgeous, film grain, grainy, highly dynamic shot, motion blur, strong depth of field, smooth blurry background, gaussian blur
negative:
seed: 1000164628
model: epicrealismXL11.0.safetensors
CFG: 7.5
steps: 20
sampler: DPM++ 2M
scheduler: Polyexponential
resolution: 1600x512
SAG: true
user description: woman walking on the beach at sunset, viewed from behind, both hands correct (given the limited resolution)

# fantasy-island-lcm.png

prompt: fairy tale a magical island floating in the sky, an endless ocean below. magical, fantastical, enchanting, storybook style, highly detailed
negative (ignored due to CFG 1): realistic, modern, ordinary, mundane
seed: 3353242733
model: gonzalomoPhotoDMD2XL6.0.safetensors
CFG: 1
steps: 12
sampler: LCM
scheduler: Polyexponential
resolution: 1024x1024

# woman-with-dog-mask-generator-testing.png

for mask generation testing only, not reproducible

# cute-kitten-dmd2.png

prompt: professional photo of a cute kitten playing with a woolen ball, <lora:DMD2fp32XL1.0:0.7>
negative:
seed: 1666281685
model: epicrealismXL11.0.safetensors
CFG: 1.5
steps: 12
sampler: Euler a
scheduler: Polyexponential
resolution: 1024x1024

# lighthouse-keeper.png

prompt: close-up portrait of a weathered lighthouse keeper, detailed face, fine skin texture, deep wrinkles, film photography, 85mm lens, dramatic side lighting
negative:
seed: 42
model: epicrealismXL11.0.safetensors
CFG: 7
steps: 25
sampler: DPM++ 2M
scheduler: Karras
resolution: 768x768
user description: sharp facial details and hat, slightly out of focus ears and clothing, blurry grey background with a sliver of sky and ocean on the left of the image, image looks slightly burned (likely CFG to high for the model)

# white-bunny-eulera.png

prompt: a cute white bunny in a field of green grass, sniffing a yellow flower
negative:
seed: 1
model: epicrealismXL11.0.safetensors
CFG: 6
steps: 20
sampler: Euler a
scheduler: Karras
resolution: 1024x1024

# white-bunny-dpm2m.png

prompt: a cute white bunny in a field of green grass, sniffing a yellow flower
negative:
seed: 4
model: epicrealismXL11.0.safetensors
CFG: 6
steps: 20
sampler: DPM++ 2M
scheduler: Karras
resolution: 1024x1024

# white-bunny-dpm2m-zoom-3.png

prompt: a cute white bunny in a field of green grass, sniffing a yellow flower<lora:EnvyZoomSliderXL1.0:-3.0>
negative:
seed: 4
model: epicrealismXL11.0.safetensors
CFG: 6
steps: 20
sampler: DPM++ 2M
scheduler: Karras
resolution: 1024x1024

# white-bunny-dpm2msde.png

prompt: a cute white bunny in a field of green grass, sniffing a yellow flower
negative:
seed: 3
model: epicrealismXL11.0.safetensors
CFG: 6
steps: 20
sampler: DPM++ 2M SDE
scheduler: Karras
resolution: 1024x1024

# lighthouse-keeper-base.png

prompt: close-up portrait of a weathered lighthouse keeper, detailed face, fine skin texture, deep wrinkles, film photography, 85mm lens, dramatic side lighting
negative:
seed: 42
model: sd_xl_base_1.0.safetensors
CFG: 7
steps: 25
sampler: DPM++ 2M
scheduler: Karras
resolution: 768x768
user description: sharp facial details and hat, slightly out of focus ears and clothing, leaning against a mostly out of frame wooden or metal structure, image looks painted

# purple-flower.png

prompt: beautiful macro photo a purple flower in a field of green, gorgeous blue sky
negative:
seed: 40
model: epicrealismXL11.0.safetensors
CFG: 7
steps: 25
sampler: DPM++ 2M
scheduler: Karras
resolution: 768x768
user description: mostly in focus purple flower on the top left, blurry blue sky in the top background, blurry green field of flowers in the bottom, horizon is rotated abuot 20°

# sleek-spaceship.png

prompt: sci-fi style photo of a sleek aerodynamic spaceship for a sci-fi space shooter video game, symmetric design, hard edges, matte grey, glowing engines, top-down view, highly detailed, 4k, uhd, smooth lighting, global illumination, uniformly lit from above, black space background
negative:
seed: 35
model: epicrealismXL11.0.safetensors
CFG: 7
steps: 25
sampler: DPM++ 2M
scheduler: Karras
resolution: 768x768
user description: shiny grey fighter jet/spaceship viewed from the front and slightly above, blue beam coming out the back (at the top of the image), orange glowing engines, glass cockpit, mostly black background with a slight gradient in the top left corner, inconsistent lighting with some parts casting shadows to the left, some to the right
