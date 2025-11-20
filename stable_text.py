from diffusers import StableDiffusionPipeline
import torch
from time import time

model_id = "sd-legacy/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# prompt = "a photo of an astronaut riding a horse on mars"
prompt=input("Generate a: ")
print(f"Beginning generation of {prompt}")
start = time()
image = pipe(prompt).images[0]  
end = time()
image.save("astronaut_rides_horse.png")
print(f"Image saved in {end-start:.2f}s!")
