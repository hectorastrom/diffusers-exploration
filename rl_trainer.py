# @Time    : 2025-11-24 15:51
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : rl_trainer.py
# Run with `accelerate launch rl_trainer.py`

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
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import argparse
from pprint import pprint

def parse_args():
    parser = argparse.ArgumentParser(description="Run DDPO training for image enhancement.")
    
    # Dataset and Trainer Constants
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="ORACLE", 
        help="The prompt template to use ('ORACLE' uses the ground-truth label)."
    )
    parser.add_argument(
        "--reward_variant", 
        type=str, 
        default="logit_change", 
        help="The variant of the CLIP reward function to use."
    )
    parser.add_argument(
        "--overfit_dset_size", 
        type=int, 
        default=128, 
        help="Number of samples for the training dataset. Set to <= 0 to use the full dataset."
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=1e-4, 
        help="Learning rate for the DDPO trainer."
    )
    parser.add_argument(
        "--noise_strength", 
        type=float, 
        default=0.4, 
        help="Controls adherence to the original image (1.0 = pure noise, 0.0 = no change)."
    )
    parser.add_argument(
        "--sample_num_steps", 
        type=int, 
        default=50, 
        help="Diffusion steps to take for sampling."
    )
    parser.add_argument(
        "--use_per_prompt_stat_tracking", 
        type=lambda x: (str(x).lower() == 'true'), 
        default=True, 
        help="Enable per-prompt statistics tracking."
    )
    parser.add_argument(
        "--loader_batch_size", 
        type=int, 
        default=128, 
        help="Batch size for the torch DataLoader (CPU limited)."
    )
    parser.add_argument(
        "--gpu_batch_size", 
        type=int, 
        default=8, 
        help="Batch size per GPU."
    )
    parser.add_argument(
        "--target_global_batch_size", 
        type=int, 
        default=256, 
        help="Target global batch size for training."
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=500, 
        help="Number of training epochs."
    )
    parser.add_argument(
        "--num_workers", 
        type=int, 
        default=8, 
        help="Number of workers for the DataLoader."
    )
    
    return parser.parse_args()


# Parse arguments first
args = parse_args()

##################################
# Constants
##################################
acc_project_config = ProjectConfiguration(
    project_dir="./ddpo_logs",
    logging_dir="./ddpo_logs/runs"
)
# initialize accelerator instance exactly as DDPOTrainer will so they don't conflict
accelerator = Accelerator(
    log_with="wandb", 
    mixed_precision="fp16",
    project_config=acc_project_config
)

# NOTE: The actual number of diffusion steps take is noise_strength *
# sample_num_steps
total_gpu_throughput = args.gpu_batch_size * accelerator.num_processes
GRAD_ACCUM_STEPS = max(1, int(args.target_global_batch_size / total_gpu_throughput))
# num batches to avg reward on per epoch -> 256 / target_global_batch_size
SAMPLE_BATCHES_PER_EPOCH = max(GRAD_ACCUM_STEPS, 4) 

DEVICE = accelerator.device

if __name__ == "__main__":
    if accelerator.is_main_process:
        print(f"Running on {accelerator.num_processes} GPUs.")
        print(f"Per-device batch: {args.gpu_batch_size}")
        print(f"Total instantaneous batch: {total_gpu_throughput}")
        print(f"Gradient Accumulation steps: {GRAD_ACCUM_STEPS}")
        print(f"Effective Global Batch: {total_gpu_throughput * GRAD_ACCUM_STEPS}")
        
    ##################################
    # Construct dataset
    ##################################
    dataset = build_COD_torch_dataset('train')
    
    if args.overfit_dset_size > 0:
        # Use a subset for testing/overfitting
        train_dataset = Subset(dataset, torch.arange(args.overfit_dset_size))
        if accelerator.is_main_process:
            print(f"Using overfit dataset of size: {args.overfit_dset_size}")
    else:
        # Use the full dataset
        train_dataset = dataset
        if accelerator.is_main_process:
            print("Using full training dataset.")
            
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.loader_batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    train_loader = accelerator.prepare(train_loader)

    # (called as ... = [self.prompt_fn() for _ in range(batch_size)])
    def create_data_generator(dataloader, prompt):
        while True:
            for batch in dataloader:
                images, labels = batch['pixel_values'], batch['label']
                paths = batch['image_path']
                # iterature through batch and yield items one by one
                for i in range(len(images)):
                    image = images[i].to(DEVICE)
                    label = labels[i].item()
                    
                    # unique identifier to work with per_prompt_stat_tracking
                    unique_path = paths[i]
                    
                    # 3 necessary return items
                    metadata = {
                        "label": label,
                        "label_str": dataset.label2str(label),
                        "original_image": image.clone(),
                        "unique_path": unique_path
                    }
                    
                    prompt_str : str
                    if prompt == "ORACLE":
                        prompt_str = f"A clear photo of {dataset.label2str(label)}"
                    else:
                        prompt_str = prompt
                        
                    # always add id
                    if args.use_per_prompt_stat_tracking: 
                        prompt_str = f"{prompt_str} id:{unique_path}"
                    
                    # prompt + image are analogous to just prompt within DDPOTrainer
                    yield prompt_str, image, metadata
                    

    my_generator = create_data_generator(train_loader, args.prompt) 


    def my_image_loader():
        """
        Return (prompt_str, image_tensor_chw, metadata_dict)
        """
        return next(my_generator)

    ##################################
    # Define reward
    ##################################
    reward_fn = CLIPReward(class_names=dataset.all_classes, device=DEVICE, reward_variant=args.reward_variant)

    ##################################
    # Build image hook
    ##################################
    # 1. Get fixed validation sample
    val_prompt, val_image, val_meta = next(my_generator)
    val_image = val_image.unsqueeze(0) # (1, C, H, W)

    # 2. Calculate "Before" Reward
    with torch.no_grad():
        val_image_device = val_image.to(DEVICE)
        # Reward fn expects lists for prompts/meta
        init_rewards, _ = reward_fn(val_image_device, [val_prompt], [val_meta])
        val_before_reward = init_rewards[0].item()

    # 3. Prepare "Before" image for WandB (Numpy format)
    val_before_img_vis = val_image.clone().squeeze(0).permute(1, 2, 0).cpu().numpy()

    def validation_hook(pipeline, noise_strength, wandb_step):
        """
        Runs validation image through pipeline, logging before and after diffusion.
        Uses cached reward for the input image.
        """
        if not accelerator.is_main_process:
            return 
        
        pipeline.vae.eval()
        pipeline.unet.eval()
        
        device = DEVICE
        input_image = val_image.to(device=device, dtype=pipeline.vae.dtype)
        
        # NOTE: Ensure input_image is [0, 1] before doing this. 
        # If your dataset outputs [-1, 1], remove the "2.0 * ... - 1.0"
        vae_input = 2.0 * input_image - 1.0
        
        # encode & noise
        with torch.no_grad():
            init_latents = pipeline.vae.encode(vae_input).latent_dist.sample()
            init_latents *= 0.18215
            noise = torch.randn_like(init_latents)
            
            timesteps = torch.tensor([int(1000 * noise_strength)], device=device).long()
            noisy_latents = pipeline.scheduler.add_noise(init_latents, noise, timesteps)
            
            prompt_ids = pipeline.tokenizer(
                [val_prompt], 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=77
            ).input_ids.to(device)
            
            prompt_embeds = pipeline.text_encoder(prompt_ids)[0]
            uncond_ids = pipeline.tokenizer(
                [""],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).input_ids.to(device)

            with torch.no_grad():
                uncond_embeds = pipeline.text_encoder(uncond_ids)[0]

            neg_prompt_embeds = uncond_embeds.repeat(prompt_embeds.shape[0], 1, 1)
            
            output = pipeline(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_prompt_embeds,
                num_inference_steps=50,
                guidance_scale=7.0,
                output_type="pil",
                latents=noisy_latents,
                starting_step_ratio=noise_strength
            )   
            
            after_img_pil = output.images[0]
        
        # Convert PIL -> Tensor (1, C, H, W) for reward calculation
        after_img_tensor = torch.from_numpy(np.array(after_img_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        after_img_tensor = after_img_tensor.to(device)
    
        # Calculate the reward ONLY for the after image
        rewards, _ = reward_fn(after_img_tensor, [val_prompt], [val_meta])
        after_reward = rewards[0].item()
        
        after_str = val_prompt
        if args.use_per_prompt_stat_tracking:
            # I just don't wanna see the big ID think
            after_str = f"{val_prompt.split(' id:')[0]} id:..."
        
        wandb.log({
            "validation/before_vs_after": [
                # Use cached numpy image and cached reward
                wandb.Image(val_before_img_vis, caption=f"Before (label='{val_meta['label_str']}', r={val_before_reward:.2f})"),
                # Use new PIL image and new reward
                wandb.Image(after_img_pil, caption=f"After (RL, prompt='{after_str}', r={after_reward:.2f})")
            ]
        }, step=wandb_step)


    ##################################
    # Configuration
    ##################################
    config = DDPOConfig(
        # --- Logging & General ---
        num_epochs=args.epochs,
        log_with= "wandb",               # Highly recommended to visualize the Reward curve
        mixed_precision="fp16",         # Standard for SD 1.5
        allow_tf32=True,
        
        # Let's not leave eta up to chance (should default to 1.0 tho)
        sample_eta = 1.0,
        
        # --- Sampling (Experience Collection) ---
        # Total samples per epoch = batch_size * num_batches * num_processes.
        # RL requires a decent "buffer" of experiences to learn effectively.
        # num_train_timesteps = sample_num_steps * train_timestep_fraction 
        # Since we only want to do last NOISE_STRENGTH fraction of inference 
        # steps, we select train_timestep_fraction to be exactly NOISE_STRENGTH
        train_timestep_fraction=args.noise_strength, 
        sample_num_steps=args.sample_num_steps,
        sample_batch_size=args.gpu_batch_size,           
        sample_num_batches_per_epoch=SAMPLE_BATCHES_PER_EPOCH, # steps * num batches = total samples / epoch
        
        # --- Training (Update Phase) ---
        train_batch_size=args.gpu_batch_size,       # Must be <= sample_batch_size
        train_gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        
        # --- Optimizer & LoRA Specifics ---
        # LoRA usually requires a slightly lower LR than full finetuning. 
        train_learning_rate=args.learning_rate, # paper used 1e-5
        train_use_8bit_adam=True,    
        
        # --- Critical for Image-to-Image ---
        # Now that prompts are unique (id:X), this calculates a moving average 
        # of the reward specifically for that image. 
        # A hard image (reward -10) will eventually have a baseline of -10.
        # If the model gets -9, that's a +1 advantage!
        per_prompt_stat_tracking=False, # disable when we use logit_change reward
        
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
        noise_strength=args.noise_strength,
        debug_hook=validation_hook,
        # DDPOTrainer builds its own Accelerator instance
    )
     
    # add custom args to wandb config
    if accelerator.is_main_process:
        wandb.config.update(vars(args))

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
    accelerator.free_memory()
    if accelerator.is_main_process:
        print("Commencing training!")
        pprint(args.__dict__)
    trainer.train()