from tqdm import tqdm
import torch
from torchvision import utils
from torchvision.transforms import functional as TF

from sample import simple_sample
from data.drawbench import get_rand_drawbench_prompts
from config import DEVICE, STEPS

def get_simple_pos(num):
    simple_square = torch.tensor([0, 0, 1/512, 1/512])
    return simple_square.repeat(num, 1)

@torch.no_grad()
@torch.random.fork_rng()
def multi_demo(model, epoch, eta=1.):
    tqdm.write('\nSampling...')
    torch.manual_seed(0)
    noise = torch.randn([9, 4, 64, 64], device=model.main_device)
    fake_classes = get_rand_drawbench_prompts(9)
    fake_pos = get_simple_pos(9).to(model.main_device)
    fake_latents = simple_sample(model, noise, STEPS, eta, fake_classes, fake_pos)
    fakes = model.decode_latents(fake_latents.to(model.bg_device))
    grid = utils.make_grid(fakes, 3).cpu()
    filename = f'demo_{epoch:05}'
    TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(filename+".png")
    with open(filename + "_prompts.txt", "+a") as text_file:
        for prompt in fake_classes:
            text_file.write(prompt+"\n")
    tqdm.write('')

@torch.no_grad()
@torch.random.fork_rng()
def single_demo(model, epoch, eta=1.):
    tqdm.write('\nSampling...')
    torch.manual_seed(0)
    noise = torch.randn([1, 4, 64, 64], device=model.main_device)
    fake_classes = ["a blue dog in a park"]
    fake_pos = get_simple_pos(1).to(model.main_device)
    fake = simple_sample(model, noise, STEPS, eta, fake_classes, fake_pos)
    filename = f'demo_{epoch:05}'
    TF.to_pil_image(fake[0].add(1).div(2).clamp(0, 1)).save(filename+".png")
    with open(filename + "_prompts.txt", "+a") as text_file:
        for prompt in fake_classes:
            text_file.write(prompt+"\n")
    tqdm.write('')