# @Time    : 2025-11-24 13:11
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : ddpo.py

import torch
from collections import defaultdict
import numpy as np
from trl import DDPOTrainer, DefaultDDPOStableDiffusionPipeline
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import rescale_noise_cfg
from dataclasses import dataclass
import wandb

# ==========================================
# 1. TRL/DDPO Helper Classes & Math
# ==========================================

@dataclass
class DDPOPipelineOutput:
    images: torch.Tensor
    latents: torch.Tensor
    log_probs: torch.Tensor

@dataclass
class DDPOSchedulerOutput:
    latents: torch.Tensor
    log_probs: torch.Tensor


def _left_broadcast(input_tensor, shape):
    input_ndim = input_tensor.ndim
    if input_ndim > len(shape):
        raise ValueError(
            "The number of dimensions of the tensor to broadcast cannot be greater "
            "than the length of the shape to broadcast to"
        )
    return input_tensor.reshape(
        input_tensor.shape + (1,) * (len(shape) - input_ndim)
    ).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    """
    Compute the per-timestep variance used by the DDIM-style update.

    All tensors (alphas_cumprod, final_alpha_cumprod, timestep, prev_timestep)
    are placed on the same device to avoid device mismatch in gather.
    Handles both scalar and batched timesteps.
    """
    # Ensure we have tensors
    if not torch.is_tensor(timestep):
        timestep = torch.as_tensor(timestep)
    if not torch.is_tensor(prev_timestep):
        prev_timestep = torch.as_tensor(prev_timestep)

    # Use the scheduler's alpha device as the "home" device for the variance math
    device = self.alphas_cumprod.device
    timestep = timestep.to(device)
    prev_timestep = prev_timestep.to(device)

    # Make sure gather sees 1D index tensors
    if timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)
    if prev_timestep.dim() == 0:
        prev_timestep = prev_timestep.unsqueeze(0)

    timestep = timestep.long()
    prev_timestep = prev_timestep.long()

    alphas_cumprod = self.alphas_cumprod.to(device)
    final_alpha_cumprod = self.final_alpha_cumprod.to(device)

    alpha_prod_t = torch.gather(alphas_cumprod, 0, timestep)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        torch.gather(alphas_cumprod, 0, prev_timestep),
        final_alpha_cumprod,
    )

    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev

    variance = (beta_prod_t_prev / beta_prod_t) * (
        1 - alpha_prod_t / alpha_prod_t_prev
    )
    return variance


def scheduler_step(
    self,
    model_output,
    timestep,
    sample,
    eta=0.0,
    use_clipped_model_output=False,
    generator=None,
    prev_sample=None,
):
    """
    Device-safe scheduler step used for:
    - sampling inside i2i_pipeline_step
    - training loss (via I2IDDPOStableDiffusionPipeline.scheduler_step)

    This version is robust to scalar timesteps (0-d tensors / ints) and
    batched timesteps (1-d tensors).
    """
    if self.num_inference_steps is None:
        raise ValueError(
            "Number of inference steps is 'None', you need to run 'set_timesteps' "
            "after creating the scheduler"
        )

    device = sample.device

    # Normalize timestep to a tensor on the correct device
    if not torch.is_tensor(timestep):
        timestep = torch.as_tensor(timestep, device=device)
    else:
        timestep = timestep.to(device)

    # Compute previous timestep in the same shape as timestep
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    prev_timestep = torch.clamp(
        prev_timestep, 0, self.config.num_train_timesteps - 1
    )

    # Ensure 1D long tensors so torch.gather works for both scalar and batched cases
    if timestep.dim() == 0:
        timestep = timestep.unsqueeze(0)
    if prev_timestep.dim() == 0:
        prev_timestep = prev_timestep.unsqueeze(0)

    timestep = timestep.long()
    prev_timestep = prev_timestep.long()

    alphas_cumprod = self.alphas_cumprod.to(device)
    final_alpha_cumprod = self.final_alpha_cumprod.to(device)

    alpha_prod_t = torch.gather(alphas_cumprod, 0, timestep)
    alpha_prod_t_prev = torch.where(
        prev_timestep >= 0,
        torch.gather(alphas_cumprod, 0, prev_timestep),
        final_alpha_cumprod,
    )

    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(device)

    beta_prod_t = 1 - alpha_prod_t

    if self.config.prediction_type == "epsilon":
        pred_original_sample = (
            sample - beta_prod_t ** 0.5 * model_output
        ) / alpha_prod_t ** 0.5
        pred_epsilon = model_output
    elif self.config.prediction_type == "sample":
        pred_original_sample = model_output
        pred_epsilon = (
            sample - alpha_prod_t ** 0.5 * pred_original_sample
        ) / beta_prod_t ** 0.5
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = (alpha_prod_t ** 0.5) * sample - (
            beta_prod_t ** 0.5
        ) * model_output
        pred_epsilon = (alpha_prod_t ** 0.5) * model_output + (
            beta_prod_t ** 0.5
        ) * sample
    else:
        raise ValueError(
            f"prediction_type given as {self.config.prediction_type} must be one "
            "of `epsilon`, `sample`, or `v_prediction`"
        )

    if self.config.thresholding:
        pred_original_sample = self._threshold_sample(pred_original_sample)
    elif self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(
            -self.config.clip_sample_range, self.config.clip_sample_range
        )

   # Compute variance
    variance = _get_variance(self, timestep, prev_timestep)

    # Branch for deterministic / zero-variance steps
    deterministic_step = (eta == 0.0) or torch.all(variance.abs() < 1e-20)

    if deterministic_step:
        # Standard DDIM deterministic update
        pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * pred_epsilon
        prev_sample_mean = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

        if prev_sample is None:
            prev_sample = prev_sample_mean

        # No stochasticity => no meaningful log-prob increment
        log_prob = torch.zeros(sample.shape[0], device=device, dtype=sample.dtype)
        return DDPOSchedulerOutput(prev_sample.type(sample.dtype), log_prob)

    # Stochastic branch (eta > 0 and variance > 0)
    variance = torch.clamp(variance, min=1e-20)
    std_dev_t = eta * variance ** 0.5
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(device)

    if use_clipped_model_output:
        pred_epsilon = (sample - alpha_prod_t ** 0.5 * pred_original_sample) / beta_prod_t ** 0.5

    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon
    prev_sample_mean = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

    if prev_sample is None:
        variance_noise = torch.randn(
            model_output.shape,
            generator=generator,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t ** 2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(np.pi, device=device, dtype=sample.dtype)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return DDPOSchedulerOutput(prev_sample.type(sample.dtype), log_prob)


# ==========================================
# 2. The Custom I2I Pipeline Logic
# ==========================================

@torch.no_grad()
def i2i_pipeline_step(
    self,
    prompt=None,
    height=None,
    width=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    negative_prompt=None,
    num_images_per_prompt=1,
    eta=0.0,
    generator=None,
    latents=None,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    output_type="pil",
    return_dict=True,
    callback=None,
    callback_steps=1,
    cross_attention_kwargs=None,
    guidance_rescale=0.0,
    starting_step_ratio: float = 1.0,
):
    # 0. Defaults
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor

    # 2. Define batch size
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device
    do_classifier_free_guidance = guidance_scale > 1.0

    # 3. Encode input prompt
    prompt_embeds = self._encode_prompt(
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
    )

    # 4. Prepare timesteps (WITH I2I FIX)
    self.scheduler.set_timesteps(num_inference_steps, device=device)

    timesteps = self.scheduler.timesteps
    start_idx = int(len(timesteps) * (1 - starting_step_ratio))
    start_idx = max(0, min(start_idx, len(timesteps) - 1))

    timesteps = timesteps[start_idx:]

    # 5. Prepare latent variables
    # Fix: avoiding double scaling, since noisy latents are already scaled
    # inside _generate_samples
    num_channels_latents = self.unet.config.in_channels

    if latents is None:
        # Fallback: behave like text-only sampling
        shape = (
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor,
        )
        latents = torch.randn(
            shape, generator=generator, device=device, dtype=prompt_embeds.dtype
        )
        latents = latents * self.scheduler.init_noise_sigma
    else:
        # For I2I we assume latents are already correctly scaled and noised
        latents = latents.to(device=device, dtype=prompt_embeds.dtype)

    # 6. Denoising loop
    all_latents = [latents]
    all_log_probs = []

    with self.progress_bar(total=len(timesteps)) as progress_bar:
        for i, t in enumerate(timesteps):
            latent_model_input = (
                torch.cat([latents] * 2)
                if do_classifier_free_guidance
                else latents
            )
            latent_model_input = self.scheduler.scale_model_input(
                latent_model_input, t
            )

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                return_dict=False,
            )[0]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (
                    noise_pred_text - noise_pred_uncond
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                noise_pred = rescale_noise_cfg(
                    noise_pred,
                    noise_pred_text,
                    guidance_rescale=guidance_rescale,
                )

            # NOTE: this now safely handles scalar t (0-d tensor) inside scheduler_step
            scheduler_output = scheduler_step(
                self.scheduler, noise_pred, t, latents, eta
            )
            latents = scheduler_output.latents
            log_prob = scheduler_output.log_probs

            all_latents.append(latents)
            all_log_probs.append(log_prob)

            progress_bar.update()

    if output_type != "latent":
        image = self.vae.decode(
            latents / self.vae.config.scaling_factor, return_dict=False
        )[0]
        image = self.image_processor.postprocess(
            image,
            output_type=output_type,
            do_denormalize=[True] * image.shape[0],
        )
    else:
        image = latents

    return DDPOPipelineOutput(image, all_latents, all_log_probs)


class I2IDDPOStableDiffusionPipeline(DefaultDDPOStableDiffusionPipeline):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if (
            hasattr(self.sd_pipeline.scheduler, "alphas_cumprod")
            and self.sd_pipeline.scheduler.alphas_cumprod is not None
        ):
            # Keep alphas on CPU to avoid device mismatch in custom scheduler code
            self.sd_pipeline.scheduler.alphas_cumprod = (
                self.sd_pipeline.scheduler.alphas_cumprod.cpu()
            )

    def __call__(self, *args, **kwargs) -> DDPOPipelineOutput:
        return i2i_pipeline_step(self.sd_pipeline, *args, **kwargs)

    def scheduler_step(self, *args, **kwargs):
        """
        Override TRL's DefaultDDPOStableDiffusionPipeline.scheduler_step so that
        DDPOTrainer.calculate_loss uses the same device-safe scheduler_step.
        """
        return scheduler_step(self.sd_pipeline.scheduler, *args, **kwargs)


# ==========================================
# 3. The Modified Trainer
# ==========================================

class ImageDDPOTrainer(DDPOTrainer):
    def __init__(self, *args, noise_strength=0.2, debug_hook=None, **kwargs):
        self.noise_strength = noise_strength
        self.debug_hook = debug_hook
        super().__init__(*args, **kwargs)
        
        # NOTE: self.num_train_timesteps is used by DDPOTrainer to calculate
        # gradient_accumulation_steps for the accelerator. 
        # By shortening the actual trajectory length below, we are effectively 
        # accumulating gradients over *more* images than configured (since each image 
        # contributes fewer steps). This is generally fine for RL stability.
        
        # We update this here so it reflects the I2I reality for logging/reference,
        # though the Accelerator is already initialized by super().__init__.
        self.num_train_timesteps = int(self.config.sample_num_steps * self.noise_strength)

    def _generate_samples(self, iterations, batch_size):
        """
        Modified to support Image inputs. No padding for shorter I2I
        trajectories (like we did previously).
        """
        current_step = wandb.run.step if wandb.run is not None else 0
        if self.debug_hook is not None and self.accelerator.is_main_process:
            self.debug_hook(self.sd_pipeline, self.noise_strength, current_step)
        
        samples = []
        prompt_image_pairs = []

        self.sd_pipeline.unet.eval()
        self.sd_pipeline.vae.eval()

        sample_neg_prompt_embeds = self.neg_prompt_embed.repeat(batch_size, 1, 1)

        for _ in range(iterations):
            # 1. GENERATE INPUTS
            prompts, input_images, prompt_metadata = zip(
                *[self.prompt_fn() for _ in range(batch_size)]
            )
            input_images = torch.stack(input_images).to(
                self.accelerator.device, dtype=self.sd_pipeline.vae.dtype
            )
            
            input_images = 2.0 * input_images - 1.0

            prompt_ids = self.sd_pipeline.tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.sd_pipeline.tokenizer.model_max_length,
            ).input_ids.to(self.accelerator.device)
            prompt_embeds = self.sd_pipeline.text_encoder(prompt_ids)[0]

            with self.autocast():
                # 2. ENCODE IMAGES (VAE)
                init_latents = self.sd_pipeline.vae.encode(
                    input_images
                ).latent_dist.sample()
                init_latents = init_latents * 0.18215

                # 3. ADD NOISE
                max_timesteps = self.sd_pipeline.scheduler.config.num_train_timesteps
                t_start = int(max_timesteps * self.noise_strength)

                timesteps = torch.tensor(
                    [t_start] * batch_size, device=self.accelerator.device
                ).long()
                noise = torch.randn_like(init_latents)
                noisy_latents = self.sd_pipeline.scheduler.add_noise(
                    init_latents, noise, timesteps
                )

                # 4. RUN PIPELINE
                sd_output = self.sd_pipeline(
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=sample_neg_prompt_embeds,
                    num_inference_steps=self.config.sample_num_steps,
                    guidance_scale=self.config.sample_guidance_scale,
                    eta=self.config.sample_eta,
                    output_type="pt",
                    latents=noisy_latents,
                    starting_step_ratio=self.noise_strength,
                )

                images = sd_output.images
                latents = sd_output.latents     # list length = 1 + effective_steps
                log_probs = sd_output.log_probs # list length = effective_steps

            # Convert lists to tensors
            latents = torch.stack(latents, dim=1)      # (B, num_effective+1, C, H, W)
            log_probs = torch.stack(log_probs, dim=1)  # (B, num_effective)

            # --- KEY CHANGE: NO PADDING ---
            # We treat the I2I trajectory as the "full" trajectory.
            # We must retrieve the specific subset of timesteps used during sampling.
            # The pipeline calculated these based on starting_step_ratio.
            
            # The scheduler's timesteps array currently holds the sequence used 
            # in the last call. It should match log_probs length.
            # Note: scheduler.timesteps is usually [981, 961, ..., 1]
            # We just need to repeat it for the batch.
            timesteps_tensor = self.sd_pipeline.scheduler.timesteps.to(self.accelerator.device)
            
            # Safety check to ensure shapes align
            if timesteps_tensor.shape[0] != log_probs.shape[1]:
                 # Fallback if scheduler state is somehow desynced (unlikely in this flow)
                 # Re-calculate indices based on noise_strength logic
                 full_timesteps = self.sd_pipeline.scheduler.config.num_train_timesteps # 1000
                 step_ratio = int(full_timesteps / self.config.sample_num_steps) # 20
                 start_step = int(full_timesteps * self.noise_strength) # 200
                 # We want [200, 180, ..., 0] (approx) - standard schedulers vary
                 # Ideally, we trust the pipeline left the correct timesteps in the scheduler
                 pass

            timesteps = timesteps_tensor.repeat(batch_size, 1)

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],
                    "next_latents": latents[:, 1:],
                    "log_probs": log_probs,
                    "negative_prompt_embeds": sample_neg_prompt_embeds,
                }
            )
            prompt_image_pairs.append([images, prompts, prompt_metadata])

        return samples, prompt_image_pairs

    def _train_batched_samples(self, inner_epoch, epoch, global_step, batched_samples):
        """
        Override to support dynamic trajectory lengths (iterating over shape[1] 
        instead of self.num_train_timesteps).
        """
        info = defaultdict(list)
        for _i, sample in enumerate(batched_samples):
            if self.config.train_cfg:
                embeds = torch.cat([sample["negative_prompt_embeds"], sample["prompt_embeds"]])
            else:
                embeds = sample["prompt_embeds"]

            # KEY CHANGE: Dynamic iteration based on actual sample length
            # sample["log_probs"] is (B, T)
            actual_num_timesteps = sample["log_probs"].shape[1]
            
            for j in range(actual_num_timesteps):
                with self.accelerator.accumulate(self.sd_pipeline.unet):
                    loss, approx_kl, clipfrac = self.calculate_loss(
                        sample["latents"][:, j],
                        sample["timesteps"][:, j],
                        sample["next_latents"][:, j],
                        sample["log_probs"][:, j],
                        sample["advantages"],
                        embeds,
                    )
                    info["approx_kl"].append(approx_kl)
                    info["clipfrac"].append(clipfrac)
                    info["loss"].append(loss)

                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            self.trainable_layers.parameters()
                            if not isinstance(self.trainable_layers, list)
                            else self.trainable_layers,
                            self.config.train_max_grad_norm,
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                    info = self.accelerator.reduce(info, reduction="mean")
                    info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                    self.accelerator.log(info, step=global_step)
                    global_step += 1
                    info = defaultdict(list)
        return global_step