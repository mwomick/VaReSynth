import torch
from torch import optim
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm

from data.cocodata import VRSCocoCaptions
from utils import get_alphas_sigmas, get_ddpm_schedule
from model import VaReSynth
from config import BATCH_SIZE, COCO_ANN_PTH, COCO_TRAIN_PTH


def eval_loss(model, rng, reals, classes, pos):
    # See: https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py#L792
    # Draw uniformly distributed continuous timesteps
    t = rng.draw(reals.shape[0])[:, 0].to(model.main_device)

    # Calculate the noise schedule parameters for those timesteps
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)
    weights = log_snrs.exp() / log_snrs.exp().add(1)

    # Combine the ground truth images and the noise
    alphas = alphas[:, None, None, None].to(model.main_device)
    sigmas = sigmas[:, None, None, None].to(model.main_device)
    latents = model.encode_image(reals.to(model.bg_device)).sample() * model.vae.config.scaling_factor
    latents.to(model.main_device)

    noise = torch.randn_like(latents).to(model.main_device)
    noised_latents = latents * alphas + noise * sigmas
    targets = noise * alphas - latents * sigmas

    # Compute the model output and the loss.
    with torch.cuda.amp.autocast():
        v = model(noised_latents, log_snrs, classes, pos).sample
        return (v - targets).pow(2).mean([1, 2, 3]).mul(weights).mean()


def train(model, opt, scaler, rng, epoch):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_set = VRSCocoCaptions(root = COCO_TRAIN_PTH,
                        annFile = COCO_ANN_PTH,
                        transform=tf)

    train_dl = data.DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    
    for i, (reals, classes, pos) in enumerate(tqdm(train_dl)):
        opt.zero_grad()
        # reals = reals
        # classes = classes.to(device)
        pos = pos.to(model.main_device)

        # Evaluate the loss
        loss = eval_loss(model, rng, reals, classes, pos)

        # Do the optimizer step and EMA update
        scaler.scale(loss).backward()
        scaler.step(opt)
        # ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)
        scaler.update()

        if i % 50 == 0:
            tqdm.write(f'Epoch: {epoch}, iteration: {i}, loss: {loss.item():g}')


def save(model, opt, scaler, epoch):
    filename = 'varesynth_coco_' + str(epoch) + '.pth'
    obj = {
        'model': model.unet.state_dict(),
        # 'model_ema': model_ema.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
    }
    torch.save(obj, filename)


from demo import demo

def run():
    # Create the model and optimizer
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using device:', device)
    torch.manual_seed(0)

    model = VaReSynth()
    # model_ema = deepcopy(model)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))

    opt = optim.Adam(model.unet.parameters(), lr=2e-4)
    scaler = torch.cuda.amp.GradScaler()

    # Use a low discrepancy quasi-random sequence to sample uniformly distributed
    # timesteps. This considerably reduces the between-batch variance of the loss.
    rng = torch.quasirandom.SobolEngine(1, scramble=True)

    epoch = 0
    while True:
        train(model, opt, scaler, rng, epoch)
        save(model, opt, scaler, epoch)
        epoch += 1
        if epoch % 5 == 0:
            demo(model, epoch)

if __name__ == "__main__":
    run()