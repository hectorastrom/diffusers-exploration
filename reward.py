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
    def __init__(self, class_names: List[str], device: str | int = 0):
        """
        Initializes CLIP and pre-computes text embeddings for all class labels.
        """
        self.device = device
        self.class_names = class_names

        print(f"Loading CLIP and caching {len(class_names)} class embeddings...")

        self.clip_pipe = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32",
            dtype=torch.bfloat16, # Ensure your GPU supports bf16, otherwise use float16 or float32
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

    def __call__(self, images: torch.Tensor, prompts: List[str], metadata: List[Dict[str, Any]]) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            images: (B, C, H, W) Tensor, values [0, 1] (Standard SD Output)
            prompts: List of strings (Ignored here)
            metadata: List of dicts containing 'label' (Integer)
            
        Returns:
            rewards: (B,) Tensor
            reward_metadata: Dict (Empty)
        """
        batch_size = images.shape[0]

        # Extract labels from metadata
        # metadata is a list of dicts: [{'label': 5}, {'label': 2}...]
        labels = torch.tensor([m["label"] for m in metadata], device=self.device)

        # 1. Convert to PIL
        pil_images = []
        for i in range(batch_size):
            # Pretty sure images are in [0, 1] range from SD
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
            logits = logit_scale * (image_features @ self.cached_text_embeds.t())
            probs = logits.softmax(dim=-1)

            # 5. Gather specific label confidence
            labels = labels.to(probs.device)
            rewards = probs[torch.arange(batch_size, device=probs.device), labels].to(torch.float32)
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=0.0, neginf=0.0)

        # DDPOTrainer/Accelerate will call .cpu().numpy() on rewards.
        # NumPy does not support bfloat16, so we convert to float32 here.
        rewards = rewards.to(torch.float32)

        # Return a tuple (reward_tensor, metadata_dict)
        print(f"Avg reward for {len(rewards)} images: {rewards.mean()}")
        return rewards, {}
