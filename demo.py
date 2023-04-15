from tqdm import tqdm
import torch
from torchvision import utils
from torchvision.transforms import functional as TF

from sample import simple_sample
from drawbench import get_rand_drawbench_prompts
from config import DEVICE, STEPS

def get_simple_pos(num):
    simple_square = torch.tensor([0, 0, 512, 512])
    return simple_square.repeat(num)

@torch.no_grad()
@torch.random.fork_rng()
def demo(model, epoch, eta=1.):
    tqdm.write('\nSampling...')
    torch.manual_seed(0)

    noise = torch.randn([9, 4, 64, 64], device=DEVICE)
    fake_classes = get_rand_drawbench_prompts(9)
    fake_pos = get_simple_pos(9)
    fake_latents = simple_sample(model, noise, STEPS, eta, fake_classes, fake_pos)
    fakes = model.decode_latents(fake_latents)
    grid = utils.make_grid(fakes, 3).cpu()
    filename = f'demo_{epoch:05}.png'
    TF.to_pil_image(grid.add(1).div(2).clamp(0, 1)).save(filename)
    tqdm.write('')