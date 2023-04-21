import torch
from torch import optim
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm

from data.concat import train_set, train_loader

from utils import get_alphas_sigmas, get_ddpm_schedule
from model import VaReSynth
from config import BATCH_SIZE, COCO_ANN_PTH, COCO_TRAIN_PTH, ACCUMULATE_STEPS


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
    latents = latents.to(model.main_device)

    noise = torch.randn_like(latents).to(model.main_device)
    noised_latents = latents * alphas + noise * sigmas
    targets = noise * alphas - latents * sigmas

    # Compute the model output and the loss.
    with torch.cuda.amp.autocast():
        v = model(noised_latents, log_snrs, classes, pos).sample
        return (v - targets).pow(2).mean([1, 2, 3]).mul(weights).mean()


def init_log():
    with open("out/loss.csv", '+a') as log:
        log.write("Epoch, Iteration, Loss")
        log.close()


def log(epoch, iteration, loss):
    with open("out/loss.csv", '+a') as log:
        log.write(f'\n{epoch}, {iteration}, {loss.item():g}')
        log.close()


def train(model, opt, scaler, rng, epoch):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_set = train_set
    train_dl = train_loader

    for i, (reals, classes, pos) in enumerate(tqdm(train_dl)):
        # reals = reals
        # classes = classes.to(device)
        pos = pos.to(model.main_device)

        # Evaluate the loss
        loss = eval_loss(model, rng, reals, classes, pos) / ACCUMULATE_STEPS

        # Do the optimizer step and EMA update
        scaler.scale(loss).backward()
    
        if (i+1) % ACCUMULATE_STEPS == 0:
            opt.zero_grad()
            scaler.step(opt)
            # ema_update(model, model_ema, 0.95 if epoch < 20 else ema_decay)
            scaler.update()

        if i % (50*ACCUMULATE_STEPS) == 0:
            log(epoch, int(i/ACCUMULATE_STEPS), loss)
            # tqdm.write(f'Epoch: {epoch}, iteration: {i}, loss: {loss.item():g}')


def save(model, opt, scaler, epoch):
    filename = 'out/varesynth_coco_' + str(epoch) + '.pth'
    obj = {
        'model': model.unet.state_dict(),
        # 'model_ema': model_ema.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
    }
    torch.save(obj, filename)


from demo import single_demo

def run(checkpoint=None):
    # Create the model and optimizer
    torch.manual_seed(0)

    epoch = 0

    model = VaReSynth()

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.unet.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        scaler = torch.cuda.amp.GradScaler().load_state_dict(checkpoint['scaler'])
        opt = optim.Adam(model.unet.parameters(), lr=1e-6).load_state_dict(checkpoint['opt'])
        print("Loaded from checkpoint.")
    # model_ema = deepcopy(model)
    else:
        init_log()
        opt = optim.Adam(model.unet.parameters(), lr=1e-6)
        scaler = torch.cuda.amp.GradScaler()
        print("Initialized new model for training.")

    # print('Model parameters:', sum(p.numel() for p in model.parameters()))
    # Use a low discrepancy quasi-random sequence to sample uniformly distributed
    # timesteps. This considerably reduces the between-batch variance of the loss.
    rng = torch.quasirandom.SobolEngine(1, scramble=True)

    # single_demo(model, epoch)
    while True:
        train(model, opt, scaler, rng, epoch)
        save(model, opt, scaler, epoch)
        epoch += 1
        if epoch % 2 == 0:
            single_demo(model, epoch-1)

if __name__ == "__main__":
    run()