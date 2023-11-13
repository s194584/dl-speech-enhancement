# import AudioDec
from utils.audiodec import AudioDec, assign_model
from dataloading.AudioDataset import AudioDataset
from torch.utils.data import DataLoader, random_split
import torch
import math
# import soundfile as sf
import os
# import numpy as np
# from plot_audio_functions import plot_specgram
# from trainer.denoise import Trainer as DenoiseTrainer

from models.autoencoder.AudioDec import Generator as generator_audiodec
import yaml
from bin.utils import load_config
from losses import DiscriminatorAdversarialLoss
from losses import FeatureMatchLoss
from losses import GeneratorAdversarialLoss
from losses import MultiMelSpectrogramLoss

ENVIRONMENT = 'LAPTOP'
# ENVIRONMENT = 'HPC'

if ENVIRONMENT == 'LAPTOP':
    CLEAN_PATH = "corpus/train/clean"
    NOISE_PATH = "corpus/train/noise"
elif ENVIRONMENT == 'HPC':
    CLEAN_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/clean_fullband/vctk_wav48_silence_trimmed"
    NOISE_PATH = "/work3/s164396/data/DNS-Challenge-4/datasets_fullband/noise_fullband"


def add_noise(speech, noise, snr):
    if noise.shape[0] < speech.shape[0]:
        pass
    assert speech.shape == noise.shape, "Shapes are not equal!"

    speech_power = speech.norm(p=2)
    noise_power = noise.norm(p=2)

    snr = math.exp(snr / 10)
    scale = snr * noise_power / speech_power
    noisy_speech = (scale * speech + noise) / 2

    return noisy_speech


def trim_audio(batch):
    batch = [
        b for b in batch if (len(b[0]) > 1)
    ]
    assert len(batch) > 0, f"No qualified audio pairs.!"
    return batch


def define_splits(dataset, generator, train_percentage=0.7, val_percentage=0.15, test_percentage=0.15):
    split_datasets = random_split(dataset, [train_percentage, val_percentage, test_percentage], generator=generator)
    return {'train': split_datasets[0], 'val': split_datasets[1], 'test': split_datasets[2]}


def load_config(path_to_config):
    with open(path_to_config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    return config


def define_optimizer_scheduler(model, config):
    generator_optimizer_class = getattr(
        torch.optim,
        config['generator_optimizer_type']
    )
    
    # discriminator_optimizer_class = getattr(
    #     torch.optim,
    #     config['discriminator_optimizer_type']
    # )
    
    optimizer = {
        'generator': generator_optimizer_class(
            model.parameters(),
            **config['generator_optimizer_params'],
        )
        # 'discriminator': discriminator_optimizer_class(
        #     model['discriminator'].parameters(),
        #     **config['discriminator_optimizer_params'],
        # ),
    }

    generator_scheduler_class = getattr(
        torch.optim.lr_scheduler,
        config.get('generator_scheduler_type', "StepLR"),
    )
    
    # discriminator_scheduler_class = getattr(
    #     torch.optim.lr_scheduler,
    #     config.get('discriminator_scheduler_type', "StepLR"),
    # )
    
    scheduler = {
        'generator': generator_scheduler_class(
            optimizer=optimizer['generator'],
            **config['generator_scheduler_params'],
        )
        # 'discriminator': discriminator_scheduler_class(
        #     optimizer=optimizer['discriminator'],
        #     **config['discriminator_scheduler_params'],
        # ),
    }
    
    return optimizer, scheduler


def define_criterion(config, device):
    criterion = {
    'gen_adv': GeneratorAdversarialLoss(
        **config['generator_adv_loss_params']).to(device),
    'dis_adv': DiscriminatorAdversarialLoss(
        **config['discriminator_adv_loss_params']).to(device),
    }
    if config.get('use_feat_match_loss', False):
        criterion['feat_match'] = FeatureMatchLoss(
            **config.get('feat_match_loss_params', {}),
        ).to(device)
        
    if config.get('use_mel_loss', False):
        criterion['mel'] = MultiMelSpectrogramLoss(
            **config['mel_loss_params'],
        ).to(device)
    
    return criterion


    
# Seeds for reproducibility #########
generator_seed = 81
generator = torch.Generator().manual_seed(generator_seed)

# device assignment
if -1 < 0:
    tx_device = 'cpu'
    rx_device = 'cpu'
else:
    tx_device = 'cuda:cuda'
    rx_device = 'cuda:cuda'
    
device = torch.device(tx_device)

# Loading model #####################
# TODO - Load model
model = 'vctk_denoise'
path_to_config = os.path.join('config', 'denoise', 'symAD_vctk_48000_hop300.yaml')
config = load_config(path_to_config)
generator_model = generator_audiodec(**config["generator_params"])
optimizer, shceduler = define_optimizer_scheduler(generator_model, config)
criterion = define_criterion(config, device)
# optimizer = torch.optim.Adam(**config["generator_optim_params"])

# Freeze components
# fix quantizer
for parameter in generator_model.quantizer.parameters():
    parameter.requires_grad = False
# fix decoder
for parameter in generator_model.decoder.parameters():
    parameter.requires_grad = False

torch.set_num_threads(1)
model_sample_rate, encoder_checkpoint, decoder_checkpoint = assign_model(model)

print("AudioDec initinalizing!")
audiodec = AudioDec(tx_device=tx_device, rx_device=rx_device)
audiodec.load_transmitter(encoder_checkpoint)
audiodec.load_receiver(encoder_checkpoint, decoder_checkpoint)

# Loading data ######################
sample_rate = 48000
assert sample_rate == model_sample_rate, "The model's sample rate is not 48000Hz."
batch_length = 1_000
snr = [20, 10, 3]
clean_dataset = AudioDataset(CLEAN_PATH, batch_length, sample_rate)
noise_dataset = AudioDataset(NOISE_PATH, batch_length, sample_rate)

clean_splits = define_splits(clean_dataset, generator)
noise_splits = define_splits(noise_dataset, generator)

train_clean_dataloader = DataLoader(clean_splits['train'], batch_size=1, shuffle=True, generator=generator)
train_noise_dataloader = DataLoader(noise_splits['train'], batch_size=1, shuffle=True, generator=generator)
# Loading Trainer ###################

steps = 0
# Training loop #####################
for i_batch, clean_sample_batch in enumerate(iter(train_clean_dataloader)):
    if i_batch == 2:
        break
    for j_batch, noise_sample_batch in enumerate(iter(train_noise_dataloader)):
        if j_batch == 2:
            break
        
        # Make pseudo batch
        for clean_sample in clean_sample_batch:
            for noise_sample in noise_sample_batch:
                x_noisy = add_noise(clean_sample, noise_sample, snr=10)
                x_target = clean_sample
                
                # Until we get a 
                x_noisy = x_noisy[None, :].float()
                x_target = x_target[None, :].float()
                # print(x_noisy.dtype)
        # Perform training on pseudo batch
                
                # Send input through model
                mode = 'train'
                x_noisy = x_noisy.to(device)
                x_target = x_target.to(device)
                
                # fix codebook
                print("Fix codebook")
                generator_model.quantizer.codebook.eval()
                
                # initialize generator loss
                gen_loss = 0.0

                # main genertor operation
                print("Main generator operation")
                
                ### TODO: channel first
                y_nc, zq, z, vqloss, perplexity = generator_model(x_noisy)

                # perplexity info
                # self._perplexity(perplexity, mode=mode)

                # vq loss
                # gen_loss += self._vq_loss(vqloss, mode=mode)
                print("VQ_loss")
                vqloss = torch.sum(vqloss)
                vqloss *= config["lambda_vq_loss"]
                gen_loss += vqloss
                
                # metric loss
                # gen_loss += self._metric_loss(y_nc, x_c, mode=mode)
                
                print("Mel_loss")
                print(y_nc.shape)
                print(x_target.shape)
                mel_loss = criterion["mel"](y_nc, x_target)
                mel_loss *= config["lambda_mel_loss"]
                # self._record_loss('mel_loss', mel_loss, mode=mode)
                gen_loss += mel_loss

                # update generator
                # self._record_loss('generator_loss', gen_loss, mode=mode)
                # self._update_generator(gen_loss)
                print("Optimizer")
                optimizer["generator"].zero_grad()
                gen_loss.backward()
                
                if config["generator_grad_norm"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        generator_model.parameters(),
                        config["generator_grad_norm"],
                    )
                optimizer["generator"].step()
                shceduler["generator"].step()


                # update counts
                print(steps)
                steps += 1
                # self.tqdm.update(1)
                # self._check_train_finish()
