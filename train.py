import torch
from torch import optim
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm

from coco import VRSCocoCaptions
from utils import get_alphas_sigmas, get_ddpm_schedule
from model import VaReSynth
from config import BATCH_SIZE, DEVICE

device = DEVICE

def eval_loss(model, rng, reals, classes, pos):
    # Draw uniformly distributed continuous timesteps
    t = rng.draw(reals.shape[0])[:, 0].to(device)

    # Calculate the noise schedule parameters for those timesteps
    log_snrs = get_ddpm_schedule(t)
    alphas, sigmas = get_alphas_sigmas(log_snrs)
    weights = log_snrs.exp() / log_snrs.exp().add(1)

    # Combine the ground truth images and the noise
    alphas = alphas[:, None, None, None]
    sigmas = sigmas[:, None, None, None]
    latents = model.encode_image(reals)
    noise = torch.randn_like(latents)
    noised_latents = latents * alphas + noise * sigmas
    targets = noise * alphas - latents * sigmas

    # Compute the model output and the loss.
    with torch.cuda.amp.autocast():
        v = model(noised_latents, log_snrs, classes, pos)
        return (v - targets).pow(2).mean([1, 2, 3]).mul(weights).mean()


def train(model, opt, scaler, rng, epoch):
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    train_set = VRSCocoCaptions(root = '/pine/scr/m/w/rwomick/train2017',
                        annFile = '/pine/scr/m/w/rwomick/annotations/captions_train2017.json',
                        transform=tf)

    train_dl = data.DataLoader(train_set, BATCH_SIZE, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    
    for i, (reals, classes, pos) in enumerate(tqdm(train_dl)):
        opt.zero_grad()
        reals = reals.to(device)
        classes = classes.to(device)
        pos = pos.to(device)

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
        'model': model.state_dict(),
        # 'model_ema': model_ema.state_dict(),
        'opt': opt.state_dict(),
        'scaler': scaler.state_dict(),
        'epoch': epoch,
    }
    torch.save(obj, filename)


from demo import demo

if __name__ == "__main__":
    # Create the model and optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    torch.manual_seed(0)

    model = VaReSynth().to(device)
    # model_ema = deepcopy(model)
    print('Model parameters:', sum(p.numel() for p in model.parameters()))

    opt = optim.Adam(model.parameters(), lr=2e-4)
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