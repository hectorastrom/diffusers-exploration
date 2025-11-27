# @Time    : 2025-11-24 13:42
# @Author  : Hector Astrom
# @Email   : hastrom@mit.edu
# @File    : reward.py

import torch
import torchvision.transforms.functional as TF
from transformers import pipeline
from typing import Any, List, Dict

eps = 1e-6

class CLIPReward:
    def __init__(self, class_names: List[str], device: str | int = 0, reward_variant: str = "logit_change"):
        """
        Initializes CLIP and pre-computes text embeddings for all class labels.
        """
        if reward_variant not in ["logit_max_margin", "logit_change"]:
            raise ValueError(f"Invalid reward_variant: {reward_variant}. Must be 'logit_max_margin' or 'logit_change'.")
            
        self.reward_variant = reward_variant
        self.device = device
        self.class_names = class_names

        print(f"Loading CLIP and caching {len(class_names)} class embeddings...")

        self.clip_pipe = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            dtype=torch.bfloat16, # accelerator runs in fp16 so will this even be bf16?
            use_fast=True,
            device=device
        )

        # Cache Text Embeddings
        prompts = [f"An image of {label}" for label in class_names]

        tokenizer = self.clip_pipe.tokenizer
        model = self.clip_pipe.model

        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            text_features = model.get_text_features(**inputs)
            norms = text_features.norm(p=2, dim=-1, keepdim=True)
            self.cached_text_embeds = text_features / (norms + eps)

        print("CLIP initialized and embeddings cached.")
    
    def _compute_target_logits(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Helper to run a batch of images through CLIP and return the target class logits.
        
        Args:
            images: (B, C, H, W) Tensor, values [0, 1]
            labels: (B,) Tensor of target class indices
            
        Returns:
            target_logits: (B,) Tensor of logits for the target class
        """
        batch_size = images.shape[0]

        # 1. Convert to PIL
        pil_images = []
        for i in range(batch_size):
            img_tensor = images[i].detach().cpu()
            img_tensor = torch.clamp(img_tensor, 0, 1)
            pil_images.append(TF.to_pil_image(img_tensor))

        # 2. Process Images
        image_inputs = self.clip_pipe.image_processor(
            images=pil_images, 
            return_tensors="pt"
        ).to(self.clip_pipe.model.device)

        with torch.no_grad():
            # 3. Get Image Embeddings & Normalize
            image_features = self.clip_pipe.model.get_image_features(**image_inputs)
            norms = image_features.norm(p=2, dim=-1, keepdim=True)
            image_features = image_features / (norms + eps)

            # 4. Cosine Similarity -> Logits -> Probs
            logit_scale = self.clip_pipe.model.logit_scale.exp()
            # gives us raw alignment score
            logits = logit_scale * (image_features @ self.cached_text_embeds.t())

            # 5. Gather specific label confidence
            labels = labels.to(logits.device)
            target_logits = logits[torch.arange(batch_size, device=logits.device), labels].to(torch.float32)
            
        return logits, target_logits
        
    def _logit_max_margin_reward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        The original reward function: target_logit - max_other_logit.
        """
        batch_size = images.shape[0]
        logits, target_logits = self._compute_target_logits(images, labels)
            
        # Compute delta from max other logit (strongest competitor)
        # mask out correct labels to find max of incorrect ones
        mask = torch.ones_like(logits, dtype=torch.bool) # tensor of True
        mask[torch.arange(batch_size), labels] = False
        
        # logits[mask] is simply a list we have to rearrange
        other_logits = logits[mask].view(batch_size, -1) 
        max_other_logits, _ = other_logits.max(dim=1) # discard idx
        
        rewards = target_logits - max_other_logits
        
        # TODO: investigate if we need to scale down rewards. massive
        # rewards are often not helpful for RL
        # seems like rewards range from -4 to 4, which feels like a good range
        
        return rewards
        
    def _logit_change_reward(self, images: torch.Tensor, metadata: List[Dict[str, Any]], labels: torch.Tensor) -> torch.Tensor:
        """
        The new reward function: target_logit_optimized - target_logit_original.
        """
        batch_size = images.shape[0]
        
        # 1. Get target logits for the diffusion optimized image (the action)
        _, target_logits_optimized = self._compute_target_logits(images, labels)
        
        # 2. Get target logits for the original (previous) image (the state)
        original_images = torch.stack([m["original_image"] for m in metadata], dim=0).to(self.device)
        # We don't need the full logits, just the target ones.
        _, target_logits_original = self._compute_target_logits(original_images, labels)
        
        # 3. Compute the change (difference) in logit
        rewards = target_logits_optimized - target_logits_original
        
        return rewards

    def __call__(self, images: torch.Tensor, prompts: List[str], metadata: List[Dict[str, Any]]) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            images: (B, C, H, W) Tensor, values [0, 1] (Standard SD Output)
            prompts: List of strings (THIS IS IGNORED! Not relevant to reward)
            metadata: List of dicts containing 'label' (Integer) and 'original_image' (Tensor)
            
        Returns:
            rewards: (B,) Tensor
            reward_metadata: Dict (Empty)
        """
        batch_size = images.shape[0]

        # Extract labels from metadata
        # metadata is a list of dicts: [{'label': 5, 'original_image':...}, ...]
        labels = torch.tensor([m["label"] for m in metadata], device=self.device)
        
        if self.reward_variant == "logit_max_margin":
            rewards = self._logit_max_margin_reward(images, labels)
        elif self.reward_variant == "logit_change":
            rewards = self._logit_change_reward(images, metadata, labels)
        else:
             # Should not happen due to check in __init__
            raise RuntimeError(f"Unknown reward choice: {self.reward_variant}")
        
        # Clipping logits
        rewards = torch.clamp(rewards, -10.0, 10.0)
        if torch.isnan(rewards).any():
            rewards = torch.nan_to_num(rewards, nan=0.0)

        # DDPOTrainer/Accelerate will call .cpu().numpy() on rewards.
        # NumPy does not support bfloat16, so we convert to float32 here.
        rewards = rewards.to(torch.float32)

        # Return a tuple (reward_tensor, metadata_dict)
        return rewards, {}