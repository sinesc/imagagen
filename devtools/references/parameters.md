# lighthouse-keeper.png

prompt: close-up portrait of a weathered lighthouse keeper, detailed face, fine skin texture, deep wrinkles, film photography, 85mm lens, dramatic side lighting
negative:
seed: 42
model: epicrealismXL11.0.safetensors
CFG: 7
steps: 25
sampler: DMP++ 2M
scheduler: Karras
resolution: 768x768
user description: sharp facial details and hat, slightly out of focus ears and clothing, blurry grey background with a sliver of sky and ocean on the left of the image, image looks slightly burned (likely CFG to high for the model)

# lighthouse-keeper-base.png

prompt: close-up portrait of a weathered lighthouse keeper, detailed face, fine skin texture, deep wrinkles, film photography, 85mm lens, dramatic side lighting
negative:
seed: 42
model: sd_xl_base_1.0.safetensors
CFG: 7
steps: 25
sampler: DMP++ 2M
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
sampler: DMP++ 2M
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
sampler: DMP++ 2M
scheduler: Karras
resolution: 768x768
user description: shiny grey fighter jet/spaceship viewed from the front and slightly above, blue beam coming out the back (at the top of the image), orange glowing engines, glass cockpit, mostly black background with a slight gradient in the top left corner, inconsistent lighting with some parts casting shadows to the left, some to the right
