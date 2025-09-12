import os
import re
import torch
import torch.distributed as dist
from pathlib import Path
from diffusers import FluxPipeline
from torch.utils.data import Dataset, DistributedSampler
from safetensors.torch import load_file
import json

class PromptDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.prompts = data
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def sanitize_filename(text, max_length=200):
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', text)
    return sanitized[:max_length].rstrip() or "untitled"
    # --node_rank $NODE_RANK \

def distributed_setup():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def main():
    rank, local_rank, world_size = distributed_setup()
    
    pipe = FluxPipeline.from_pretrained('',
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    ).to("cuda")
    mode1='LPD'

    state_dict = load_file("")
    pipe.transformer.load_state_dict(state_dict)
    sub='photo'
    dataset = PromptDataset(f"./downloads/{sub}.json")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    output_dir = Path(f"./assets/{sub}/{mode1}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg =3.5
    infer_step=50
    for idx in sampler:
        prompt = dataset[idx]
        try:
            generator = torch.Generator(device=f"cuda:{local_rank}")
            generator.manual_seed(42 + idx*1000)
            h =1024
            image = pipe(
                prompt,
                guidance_scale=cfg,
                height=h,
                width=1024,
                num_inference_steps=infer_step,
                max_sequence_length=512,
                generator=generator
            ).images[0]

            filename = sanitize_filename(prompt)
            save_path = output_dir / f"{idx}.jpg"
            image.save(save_path)
            print(f"[Rank {rank}] Generated: {save_path.name}")

        except Exception as e:
            print(f"[Rank {rank}] Error processing '{prompt[:20]}...': {str(e)}")
    sub='paintings'
    dataset = PromptDataset(f"./downloads/{sub}.json")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    output_dir = Path(f"./assets/{sub}/{mode1}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in sampler:
        prompt = dataset[idx]
        try:
            generator = torch.Generator(device=f"cuda:{local_rank}")
            generator.manual_seed(42 + idx*1000)
            h =1024
            image = pipe(
                prompt,
                guidance_scale=cfg,
                height=h,
                width=h,
                num_inference_steps=infer_step,
                max_sequence_length=512,
                generator=generator
            ).images[0]

            filename = sanitize_filename(prompt)
            save_path = output_dir / f"{idx}.jpg"
            image.save(save_path)
            print(f"[Rank {rank}] Generated: {save_path.name}")

        except Exception as e:
            print(f"[Rank {rank}] Error processing '{prompt[:20]}...': {str(e)}")
    sub='anime'
    dataset = PromptDataset(f"./downloads/{sub}.json")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    output_dir = Path(f"./assets/{sub}/{mode1}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in sampler:
        prompt = dataset[idx]
        try:
            generator = torch.Generator(device=f"cuda:{local_rank}")
            generator.manual_seed(42 + idx*1000)
            h =1024
            image = pipe(
                prompt,
                guidance_scale=cfg,
                height=h,
                width=h,
                num_inference_steps=infer_step,
                max_sequence_length=512,
                generator=generator
            ).images[0]

            filename = sanitize_filename(prompt)
            save_path = output_dir / f"{idx}.jpg"
            image.save(save_path)
            print(f"[Rank {rank}] Generated: {save_path.name}")

        except Exception as e:
            print(f"[Rank {rank}] Error processing '{prompt[:20]}...': {str(e)}")
    sub ='concept-art2'
    dataset = PromptDataset(f"./downloads/{sub}.json")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    output_dir = Path(f"./assets/{sub}/{mode1}")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx in sampler:
        prompt = dataset[idx]
        try:
            generator = torch.Generator(device=f"cuda:{local_rank}")
            generator.manual_seed(42 + idx*1000)
            h =1024
            image = pipe(
                prompt,
                guidance_scale=cfg,
                height=h,
                width=h,
                num_inference_steps=infer_step,
                max_sequence_length=512,
                generator=generator
            ).images[0]

            filename = sanitize_filename(prompt)
            save_path = output_dir / f"{idx}.jpg"
            image.save(save_path)
            print(f"[Rank {rank}] Generated: {save_path.name}")

        except Exception as e:
            print(f"[Rank {rank}] Error processing '{prompt[:20]}...': {str(e)}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
