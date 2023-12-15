import time

import numpy as np
from torch import nn
from torch.optim import Adam

from dataloader.data_utils import add_noise, get_dataloaders
from losses import (
    GeneratorAdversarialLoss,
    DiscriminatorAdversarialLoss,
    FeatureMatchLoss,
    MultiMelSpectrogramLoss,
)
from dataloader.AudioDataset import AudioDataset
from torchmetrics.audio import (
    SignalNoiseRatio,
    # SignalDistortionRatio,
    # ShortTimeObjectiveIntelligibility
)
import torch
import os

from models.autoencoder_without_PQC.AudioDec import Generator as GeneratorAudioDec
from models.vocoder.HiFiGAN import Discriminator as DiscriminatorHiFiGAN
import yaml
from clearml import Task
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-e", "--environment", default="LAPTOP")
parser.add_argument("-sr", "--sample_rate", default=48000)
parser.add_argument("-s", "--seed", default=82)
parser.add_argument("-asr", "--audio_sample_rate", default=48000)

args = parser.parse_args()
SAMPLE_RATE = int(args.sample_rate)
ENVIRONMENT = args.environment
# ENVIRONMENT = "HPC"

if ENVIRONMENT == "LAPTOP":
    CLEAN_PATH = "corpus/train/clean"
    CLEAN_ROOT = "clean"
    NOISE_PATH = "corpus/train/noise"
    NOISE_ROOT = "noise"

    task_name = "Laptop-TEST"
    task = Task.init("dl-speech-enhancement", task_name)
    logger = task.get_logger()
    torch.set_num_threads(4)
elif ENVIRONMENT == "HPC":
    task_name = "24kHz-Multi-Mel_"
    task = Task.init("dl-speech-enhancement", task_name)
    logger = task.get_logger()
    CLEAN_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed"
    CLEAN_ROOT = "vctk_wav48_silence_trimmed"
    NOISE_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/noise_fullband"
    NOISE_ROOT = "noise_fullband"
else:
    raise Exception("Illegal argument: " + ENVIRONMENT)


def load_config(path_to_config):
    with open(path_to_config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config


# Seeds for reproducibility #########
generator_seed = int(args.seed)
random_generator = torch.manual_seed(generator_seed)

# device assignment
if ENVIRONMENT == "LAPTOP":
    tx_device = "cpu"
    rx_device = "cpu"
else:
    tx_device = "cuda:0"
    rx_device = "cuda:0"

device = torch.device(tx_device)

# Loading model #####################
# model = "vctk_denoise"
model: dict[str, torch.nn.Module] = {}
path_to_config = os.path.join("config", "denoise", "symAD_vctk_48000_hop300.yaml")

config = load_config(path_to_config)
model["generator"] = GeneratorAudioDec(model_sample_rate=SAMPLE_RATE, **config["generator_params"])

## Discriminator
model["discriminator"] = DiscriminatorHiFiGAN(**config["discriminator_params"])
model["discriminator"] = model["discriminator"].to(tx_device)

# Optimizer
# generator_optimizer_type: Adam
# generator_optimizer_params:
#     lr: 1.0e-4
#     betas: [0.5, 0.9]
#     weight_decay: 0.0
optimizer = {}
optimizer["generator"] = Adam(model["generator"].parameters(), lr=5e-5)
optimizer["discriminator"] = Adam(model["discriminator"].parameters(), lr=2.0e-4)


# model_sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model)
# TODOO - Create own config file since we don't use theirs as much anymore
def load_from_pretrained(checkpoint):
    state_dict = torch.load(checkpoint, map_location="cpu")
    # model['generator'].load_state_dict(state_dict["model"]["generator"])
    model["generator"].load_state_dict(state_dict)


## Generator
# encoder_checkpoint = os.path.join("exp", "denoise", "HPC-Fresh-Melcheckpoint-83745.pkl")
# load_from_pretrained(encoder_checkpoint)
encoder = model["generator"].encoder
encoder.to(device=tx_device)
decoder = model["generator"].decoder
decoder.to(device=tx_device)

measures = {
    "MAE": nn.L1Loss().to(device),
    # 'MSE': nn.MSELoss().to(device),
    # 'SNR': SignalNoiseRatio().to(device),
    # 'SDR': SignalDistortionRatio().to(device),
    # 'SI-SDR': ScaleInvariantSignalDistortionRatio().to(device),
    # 'PESQ': PerceptualEvaluationSpeechQuality(fs=16000, mode='wb'),
    # 'STOI': ShortTimeObjectiveIntelligibility(48000),
    "Mel-loss": MultiMelSpectrogramLoss(**config["mel_loss_params"]).to(device),
}
criterion = {
    "gen_adv": GeneratorAdversarialLoss(**config["generator_adv_loss_params"]).to(
        tx_device
    ),
    "dis_adv": DiscriminatorAdversarialLoss(
        **config["discriminator_adv_loss_params"]
    ).to(tx_device),
    "feat_match": FeatureMatchLoss().to(device),
}
lambda_adv = 1


def calculate_generator_loss(pred, target):
    gen_loss = int(config["lambda_mel_loss"]) * measures["Mel-loss"](
        pred, target
    )
    if model["discriminator"] != None:
        p_ = model["discriminator"](pred)
        with torch.no_grad():
            p = model["discriminator"](target)
        gen_loss += criterion["gen_adv"](pred) * int(config["lambda_adv"])
        gen_loss += criterion["feat_match"](p_, p) * int(config["lambda_feat_match"])
    return gen_loss


def calculate_discriminator_loss(pred, target):
    p = model["discriminator"](target)
    p_ = model["discriminator"](pred)

    real_loss, fake_loss = criterion["dis_adv"](p_, p)
    dis_loss = real_loss + fake_loss
    dis_loss *= int(config["lambda_adv"])

    return dis_loss


# Loading data ######################
clean_dataset = AudioDataset(CLEAN_PATH, CLEAN_ROOT, 48000)
noise_dataset = AudioDataset(NOISE_PATH, NOISE_ROOT, 48000)

batch_length = 48000
if ENVIRONMENT == "LAPTOP":
    batch_size = 4
else:
    batch_size = 6

split = [0.7, 0.15, 0.15]
train_clean_dataloader, val_clean_dataloader, _ = get_dataloaders(clean_dataset, split, batch_size, batch_length,
                                                                  args.seed)
train_noise_dataloader, val_noise_dataloader, _ = get_dataloaders(noise_dataset, split, batch_size, batch_length,
                                                                  args.seed)


def print_gradients(model):
    # Print gradients for each parameter
    max_grad = -float('inf')  # Initialize to negative infinity
    min_grad = float('inf')  # Initialize to positive infinity
    total_grad = 0.0
    num_params = 0

    # Iterate through the network parameters and compute max, min, and sum of gradients
    for name, param in model.named_parameters():
        if param.grad is not None:
            max_grad = max(max_grad, param.grad.max().item())
            min_grad = min(min_grad, param.grad.min().item())
            total_grad += param.grad.abs().sum().item()
            num_params += param.grad.numel()

    # Calculate average gradient
    avg_grad = total_grad / num_params if num_params > 0 else 0.0

    # Print max, min, and average gradients
    logger.report_scalar("Gradients", "Maximum", max_grad, steps)
    logger.report_scalar("Gradients", "Minimum", min_grad, steps)
    logger.report_scalar("Gradients", "Average (Abs)", avg_grad, steps)


def model_step(target, x, mode='train'):
    x = x.to(device)
    target = target.to(device)
    if mode == "train":
        model["generator"].train()
    else:
        model["generator"].eval()

    if model["discriminator"] is not None:
        if mode == "train":
            model["discriminator"].train()
        else:
            model["discriminator"].eval()

    # Predict
    y_pred = model['generator'](x)

    # Generator loss & backprop
    gen_loss = calculate_generator_loss(y_pred, target)
    if mode == 'train':
        optimizer["generator"].zero_grad()
        gen_loss.backward()

        # Printing max, min and avg absolute gradients

        # Clip gradient
        if config["generator_grad_norm"] > 0:
            torch.nn.utils.clip_grad_norm_(
                model["generator"].parameters(),
                config["generator_grad_norm"],
            )
        print_gradients(model['generator'])
        optimizer["generator"].step()

    # Discriminator loss
    dis_loss = torch.tensor(1)
    if model["discriminator"] is not None:
        with torch.no_grad():
            y_pred = model['generator'](x)

        dis_loss = calculate_discriminator_loss(y_pred.detach(), target)

        if mode == "train":
            # Clip gradient
            optimizer["discriminator"].zero_grad()
            dis_loss.backward()
            if config["discriminator_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model["discriminator"].parameters(),
                    config["discriminator_grad_norm"],
                )
            optimizer["discriminator"].step()
    return gen_loss, dis_loss


start_time = time.perf_counter()
steps = 0
epochs = [i for i in range(500)]
# Maybe have an epoch here
train_acc_batch_size = 0
train_steps = 0
model["discriminator"] = None
record_audio_snippets = False

print("Start training")


def report_time(phase):
    current_time = time.perf_counter()
    time_dif = current_time - start_time
    seconds, minutes, hours = (
        int(time_dif % 60),
        int((time_dif // 60) % 60),
        int(time_dif // 3600),
    )
    print(f"{phase}: Step {train_steps} \t Time: {hours}:{minutes}:{seconds}")


for epoch in epochs:
    # Training loop #####################
    i_batch = 0
    train_losses = {"generator": [], "discriminator": []}
    print("New epoch")
    for i_batch, (clean_sample_batch, noise_sample_batch) in enumerate(
            zip(train_clean_dataloader, train_noise_dataloader)
    ):
        if ENVIRONMENT == "LAPTOP" and i_batch == 3:
            break
        if len(clean_sample_batch) > len(noise_sample_batch):
            clean_sample_batch = clean_sample_batch[: len(noise_sample_batch)]
        else:
            noise_sample_batch = noise_sample_batch[: len(clean_sample_batch)]

        # Mix noise
        mixed_samples = add_noise(
            clean_sample_batch,
            noise_sample_batch,
            torch.randint(10, 20, (1,)).to(device),
        )

        gen_loss, dis_loss = model_step(clean_sample_batch, mixed_samples)
        gen_loss, dis_loss = gen_loss.item(), dis_loss.item()
        train_losses["generator"].append(gen_loss)
        train_losses["discriminator"].append(dis_loss)
        steps += 1
        train_steps += 1

        if steps % 100 == 0 or ENVIRONMENT == "LAPTOP":
            report_time("Training")
            logger.report_scalar(
                "Generator Batch Loss", "Train", gen_loss, iteration=train_steps
            )
            logger.report_scalar(
                "Discriminator Batch Loss", "Train", dis_loss, iteration=train_steps
            )

    avg_gen_train_loss = np.mean(train_losses["generator"])
    avg_dis_train_loss = np.mean(train_losses["discriminator"])
    # Do a checkpoint
    if ENVIRONMENT != "LAPTOP" and epoch % 1 == 0:
        check_point_path = os.path.join(
            "exp", "denoise", "fresh", f"{task_name}checkpoint-{train_steps}.pkl"
        )
        torch.save(model["generator"].state_dict(), check_point_path)

    # Validation loop ###################
    gen_val_loss = 0
    dis_val_loss = 0
    i_batch = 0
    for i_batch, (clean_sample_batch, noise_sample_batch) in enumerate(
            zip(val_clean_dataloader, val_noise_dataloader)
    ):
        if len(clean_sample_batch) > len(noise_sample_batch):
            clean_sample_batch = clean_sample_batch[: len(noise_sample_batch)]
        else:
            noise_sample_batch = noise_sample_batch[: len(clean_sample_batch)]
        if ENVIRONMENT == "LAPTOP" and i_batch == 3:
            break
        clean_sample_batch.to(device)
        noise_sample_batch.to(device)
        # Mix noise
        mixed_samples = add_noise(
            clean_sample_batch,
            noise_sample_batch,
            torch.randint(10, 20, (1,)).to(device),
        )

        with torch.no_grad():
            _gen_val_loss, _dis_val_loss = model_step(
                clean_sample_batch, mixed_samples, mode='eval'
            )
            gen_val_loss += _gen_val_loss.item()
            dis_val_loss += _dis_val_loss.item()

        if steps % 100 == 0 or ENVIRONMENT == "LAPTOP":
            report_time("Validation")
        steps += 1
    avg_gen_val_loss = gen_val_loss / (i_batch + 1)
    avg_dis_val_loss = dis_val_loss / (i_batch + 1)
    logger.report_scalar("Generator Loss", "Train", avg_gen_train_loss, iteration=epoch)
    logger.report_scalar(
        "Generator Loss", "Validation", avg_gen_val_loss, iteration=epoch
    )
    logger.report_scalar(
        "Discriminator Loss", "Train", avg_dis_train_loss, iteration=epoch
    )
    logger.report_scalar(
        "Discriminator Loss", "Validation", avg_dis_val_loss, iteration=epoch
    )
