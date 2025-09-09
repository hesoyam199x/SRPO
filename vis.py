import os
import re
import torch
import torch.distributed as dist
from pathlib import Path
from diffusers import FluxPipeline
from torch.utils.data import Dataset, DistributedSampler
from safetensors.torch import load_file
import json
from PIL import Image
import torchvision.transforms as T
class PromptDataset(Dataset):
    def __init__(self, file_path):
        if not file_path is None:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.prompts = data
        else:
            self.prompts = [     
                "The Death of Ophelia by John Everett Millais, Pre-Raphaelite painting, Ophelia floating in a river surrounded by flowers, detailed natural elements, melancholic and tragic atmosphere",
                'A field of vibrant red poppies with green stems under a blue sky.',
                "Girl with a Pearl Earring, by Johannes Vermeer, Dutch Baroque style painting",
                "A close-up of a white flower with orange stamens on a branch.",
                "Starry Night by Vincent van Gogh, Post-Impressionism",
                "A digital pen lineart sketch of a Japanese schoolgirl.",
                "A traditional Chinese building with red pillars and an ornate roof, with a pagoda visible in the background.",
                "A small blue-gray butterfly with black stripes rests on a white and yellow flower against a blurred green background.",
                "A grey tabby cat with yellow eyes rests on a weathered wooden log under bright sunlight..png",
                "A single yellow flower with green stems stands out against a dark, blurred green background.",
                "A black bird perches on a corrugated metal fence with skyscrapers in the background under a blue sky.",
                "An underwater view of a person with dark hair wearing a white shirt and black tie, submerged in blue water.",
                "A bridge spans a wide river with a cityscape on the far bank, viewed from a grassy embankment.",
                "A close-up of a white flower with yellow stamens against a dark green, blurred background.",
                "A person with dark skin and short black hair sits on a wooden chair, wearing a yellow floral dress against a grey wall with strong shadows",
                "A still frame from a black and white movie, featuring a man in classic attire, dramatic high contrast lighting, deep shadows, retro film grain, and a nostalgic cinematic mood.",
                "A first-person screenshot from Half-Life, FPS.",
                "Two young ladies seated with several other people at a dinner table.",
                "Side of a street, where there is a fire hydrant.",
                '16-year-old teenager wearing a white bear-ear hat with a smirk on their face.'
            ]*20

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def sanitize_filename(text, max_length=200):
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', text)
    return sanitized[:max_length].rstrip() or "untitled"
    # --node_rank $NODE_RANK \

def distributed_setup():
    try:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
    except Exception as e:
        rank =0
        world_size=8
    local_rank = int(os.environ['LOCAL_RANK'])

    
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size

def main():
    rank, local_rank, world_size = distributed_setup()
    
    pipe = FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-dev',
        torch_dtype=torch.bfloat16,
        use_safetensors=True
    ).to("cuda")
    save_path='srpo'
    state_dict = load_file("/our_checkpoint/diffusion_pytorch_model.safetensors")
    pipe.transformer.load_state_dict(state_dict)
    sub='hps'
    dataset = PromptDataset(None)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    output_dir = Path(f"./assets/{sub}/{save_path}")
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

if __name__ == "__main__":
    main()
