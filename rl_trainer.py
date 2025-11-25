# @Time    : 2025-11-24 15:51
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : rl_trainer.py
# Run with `accelerate launch rl_trainer.py``

"""
Goal: Teach a diffusion model to perceptually enhance images of camouflaged
animals so that it maximizes the classification accuracy of a vision model on
the image.

Input to RL: x = {Image, Prompt} (which kind of prompt is an open question)
Output from RL: r(x) = score from CLIP on class "An image of {label}" out of all
possible labels
"""


from trl import DDPOConfig
from ddpo import ImageDDPOTrainer, I2IDDPOStableDiffusionPipeline
from reward import CLIPReward
from COD_dataset import build_COD_torch_dataset
from torch.utils.data import DataLoader
import os
import torch
import wandb

##################################
# Constants
##################################
LOADER_BATCH_SIZE = 16 # limited by CPU - faster to fetch many at once
GPU_BATCH_SIZE = 8 # limited by VRAM
# TODO: Play around with what prompt works best! Can't be class dependent
PROMPT = ""
NUM_WORKERS = 1
NOISE_STRENGTH = 0.2 # controls adherance to original image (1.0 = pure noise, 0.0 = no change)
SAMPLE_NUM_STEPS = 50 # diffusion steps to take
SAMPLE_BATCHES_PER_EPOCH = 8 # num batches to avg reward on per epoch
EPOCHS = 500
DEVICE = 'cuda'

##################################
# Construct dataset
##################################
dataset = build_COD_torch_dataset('train')
train_loader = DataLoader(dataset, batch_size=LOADER_BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# (called as ... = [self.prompt_fn() for _ in range(batch_size)])
def create_data_generator(dataloader, prompt):
    while True:
        for batch in dataloader:
            images, labels = batch['pixel_values'], batch['label']
            # iterature through batch and yield items one by one
            for i in range(len(images)):
                image = images[i].to(DEVICE)
                label = labels[i].item()
                
                # 3 necessary return items
                metadata = {
                    "label": label,
                    "label_str": dataset.label2str(label),
                    "original_image": image.clone()
                }
                
                # prompt + image are analogous to just prompt within DDPOTrainer
                yield prompt, image, metadata
                

my_generator = create_data_generator(train_loader, PROMPT) 


def my_image_loader():
    """
    Return (prompt_str, image_tensor_chw, metadata_dict)
    """
    return next(my_generator)

##################################
# Define reward
##################################
reward_fn = CLIPReward(class_names=dataset.all_classes, device=DEVICE)

##################################
# Build image hook
##################################
# get fixed validation sample to use for hook
val_prompt, val_image, val_meta = next(my_generator)
val_image = val_image.unsqueeze(0) # (1, C, H, W)

def validation_hook(pipeline, noise_strength, wandb_step):
    """
    Runs validation image through pipeline, logging before and after diffusion
    """
    pipeline.vae.eval()
    pipeline.unet.eval()
    
    device = DEVICE
    input_image = val_image.to(device=device, dtype=pipeline.vae.dtype)
    # convert [0, 1] -> [-1, 1] for VAE
    vae_input = 2.0 * input_image - 1.0
    
    # encode & noise (how we start rl)
    with torch.no_grad():
        # .18125 is vae encoding normalization factor
        init_latents = pipeline.vae.encode(vae_input).latent_dist.sample()
        init_latents *= 0.18215
        noise = torch.randn_like(init_latents)
        
        # calculate start step from noise_strength
        timesteps = torch.tensor([int(1000 * noise_strength)], device=device).long()
        noisy_latents = pipeline.scheduler.add_noise(init_latents, noise, timesteps)
        
        # denoise
        prompt_ids = pipeline.tokenizer(
            [val_prompt], 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=77
        ).input_ids.to(device)
        
        prompt_embeds = pipeline.text_encoder(prompt_ids)[0] # take first? why 0?
        neg_prompt_embeds = torch.zeros_like(prompt_embeds)
        
        output = pipeline(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            num_inference_steps = 50, # high for visualization quality
            guidance_scale = 7.0,
            output_type="pil",
            latents=noisy_latents,
            starting_step_ratio=noise_strength
        )   
        
        after_img = output.images[0]
    
    val_reward = reward_fn(input_image.unsqueeze(0), [val_prompt], [val_meta])
    
    # prepare before image
    # (H, W, C) - this should be in [0, 1] range by default so might be overkill
    before_img = input_image.clone().detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    wandb.log({
        "validation/before_vs_after": [
            wandb.Image(before_img, caption=f"Before (Label: {val_meta['label_str']})"),
            wandb.Image(after_img, caption=f"After (RL, prompt={val_prompt}, r={val_reward:.2f})")
        ]
    }, step=wandb_step)


##################################
# Configuration
##################################
config = DDPOConfig(
    # --- Logging & General ---
    num_epochs=EPOCHS,
    log_with= "wandb",               # Highly recommended to visualize the Reward curve
    mixed_precision="fp16",         # Standard for SD 1.5
    allow_tf32=True,
    
    # --- Sampling (Experience Collection) ---
    # Total samples per epoch = batch_size * num_batches * num_processes.
    # RL requires a decent "buffer" of experiences to learn effectively.
    sample_num_steps=SAMPLE_NUM_STEPS,
    sample_batch_size=GPU_BATCH_SIZE,           
    sample_num_batches_per_epoch=SAMPLE_BATCHES_PER_EPOCH, # steps * num batches = total samples / epoch
    
    # --- Training (Update Phase) ---
    train_batch_size=GPU_BATCH_SIZE,       # Must be <= sample_batch_size
    train_gradient_accumulation_steps=1,
    
    # --- Optimizer & LoRA Specifics ---
    # LoRA usually requires a slightly lower LR than full finetuning. 
    # 3e-4 (default) can be unstable; 1e-4 is safer for LoRA.
    train_learning_rate=1e-4,       
    train_use_8bit_adam=True,       # Saves VRAM, virtually no downside.
    
    # --- Critical for Image-to-Image ---
    # In I2I, some images are naturally "harder" to classify than others.
    # You don't want to punish the model for a hard image if it improved 
    # relative to the *last time* it saw that image.
    # This setting tracks a baseline *per prompt*, so we disable it b/c we care
    # about a baseline per image. There's a serious possibility this devastates
    # performance. Leave false, but MAY REVISIT LATER
    per_prompt_stat_tracking=False,  
    
    # --- Project Info ---
    project_kwargs={
        "project_dir": "./ddpo_logs",
        "logging_dir": "./ddpo_logs/runs",
    },
    tracker_project_name="67960-ddpo-classifier-optimization",
)

pipeline = I2IDDPOStableDiffusionPipeline(
    pretrained_model_name="stable-diffusion-v1-5/stable-diffusion-v1-5",
    use_lora=True, # only thing you need to enable LoRA (see pipeline class definition)
)

trainer = ImageDDPOTrainer(
    config=config,
    reward_function=reward_fn,
    prompt_function=my_image_loader, # custom prompt function to send images
    sd_pipeline=pipeline,
    noise_strength=NOISE_STRENGTH,
    debug_hook=validation_hook,
)

##################################
# Fix accelerate save_state bug
##################################
original_save_state = trainer.accelerator.save_state

def patched_save_state(output_dir=None, **kwargs):
    if output_dir is None:
        # Force the path if it's missing
        output_dir = os.path.join(config.project_kwargs["project_dir"], "checkpoints")
    
    # Ensure the directory exists (the root cause of your error)
    os.makedirs(output_dir, exist_ok=True)
    
    return original_save_state(output_dir=output_dir, **kwargs)

trainer.accelerator.save_state = patched_save_state


##################################
# Begin training!
##################################
if __name__ == "__main__":
    print("Commencing training!")
    trainer.train()