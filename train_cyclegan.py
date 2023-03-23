import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from numpy import random
import imageio

from torch.nn.functional import mse_loss
from dataloader import get_data_loader
from models import Discriminator, CycleGenerator
from options import CycleGanOptions
from PIL import Image

class Trainer:
    def __init__(self, opts):
        self.opts = opts

        #config dirs
        self.expdir = './cycle_gan'
        self.plotdir = os.path.join(self.expdir, 'plots')
        self.ckptdir = os.path.join(self.expdir, 'checkpoints')

        os.makedirs(self.plotdir, exist_ok = True)
        os.makedirs(self.ckptdir, exist_ok = True)

        #config data
        self.apple_trainloader, self.apple_testloader = get_data_loader('Apple', self.opts.batch_size, self.opts.num_workers)
        self.windows_trainloader, self.windows_testloader = get_data_loader('Windows', self.opts.batch_size, self.opts.num_workers)

        #config models

        ##apple->windows generator
        self.G_a2w = CycleGenerator(self.opts).to(self.opts.device)
        ##windows->apple generator
        self.G_w2a = CycleGenerator(self.opts).to(self.opts.device)

        generator_params = list(self.G_a2w.parameters()) + list(self.G_w2a.parameters())

        ##apple discriminator
        self.D_a = Discriminator(self.opts).to(self.opts.device)

        ##windows discriminator
        self.D_w = Discriminator(self.opts).to(self.opts.device)

        discriminator_params = list(self.D_a.parameters()) + list(self.D_w.parameters())

        #config optimizers
        self.G_optim = torch.optim.Adam(generator_params, lr=self.opts.lr, betas=(0.5, 0.999))
        self.D_optim = torch.optim.Adam(discriminator_params, lr = self.opts.lr, betas = (0.5, 0.999))

        #config training
        self.niters = self.opts.niters


    def run(self):

        for i in range(self.niters):
            if i % self.opts.eval_freq == 0:
                self.eval_step(i)
            if i % self.opts.save_freq == 0:
                self.save_step(i)
            self.train_step(i)


    def train_step(self, epoch):
        self.G_w2a.train()
        self.G_a2w.train()

        self.D_a.train()
        self.D_w.train()

        apple_loader = iter(self.apple_trainloader)
        windows_loader = iter(self.windows_trainloader)

        num_iters = min(len(self.apple_trainloader) // self.opts.batch_size, len(self.windows_trainloader) // self.opts.batch_size)

        pbar = tqdm(range(num_iters))
        for i in pbar:

            #load data
            apple_data = next(apple_loader).to(self.opts.device)
            windows_data = next(windows_loader).to(self.opts.device)

            #####TODO:train discriminator on real data#####
            self.D_optim.zero_grad()

            #D_real_loss = 0
            D_W_real = self.D_w(windows_data)
            D_A_real = self.D_a(apple_data)
            D_W_real_loss = mse_loss(D_W_real, torch.ones_like(D_W_real).to(self.opts.device))
            D_A_real_loss = mse_loss(D_A_real, torch.ones_like(D_A_real).to(self.opts.device))

            D_real_loss = D_A_real_loss + D_W_real_loss
            #D_real_loss.backward()
            #self.D_optim.step()

            ###############################################

            #####TODO:train discriminator on fake data#####
            #self.D_optim.zero_grad()
            #D_fake_loss = 0

            fake_windows = self.G_a2w(apple_data)
            fake_apple = self.G_w2a(windows_data)
            D_W_fake = self.D_w(fake_windows.detach())
            D_A_fake = self.D_a(fake_apple.detach())

            D_A_fake_loss = mse_loss(D_A_fake, torch.zeros_like(D_A_fake).to(self.opts.device))
            D_W_fake_loss = mse_loss(D_W_fake, torch.zeros_like(D_W_fake).to(self.opts.device))

            D_fake_loss = D_A_fake_loss + D_W_fake_loss

            D_loss = (D_real_loss + D_fake_loss)/2
            #D_fake_loss.backward()
            D_loss.backward()
            self.D_optim.step()


            ###############################################

            #####TODO:train generator#####
            ##### windows--> fake apple--> reconstructed windows

            self.G_optim.zero_grad()
            #adversarial loss
            D_A_fake = self.D_a(fake_apple)
            # generator has to fool the discriminator so classify fake A as real
            loss_G_1 = mse_loss(D_A_fake, torch.ones_like(D_A_fake).to(self.opts.device))

            #cycle loss / reconstructed loss
            if self.opts.use_cycle_loss:
                #reconstructed windows from fake apple
                reconstructed_windows = self.G_a2w(fake_apple)
                cycle_loss_windows = nn.L1Loss()(windows_data, reconstructed_windows)
                #cycle_loss_windows = nn.MSELoss()(windows_data, reconstructed_windows)
                #cycle_loss_windows = torch.abs( reconstructed_windows - windows_data).sum() / windows_data.size(0)
                loss_G_1 += (self.opts.LAMBDA_CYCLE * cycle_loss_windows)


            ##### apple--> fake windows--> reconstructed apple
            #G_loss=0
            #self.G_optim.zero_grad()

            #adversarial loss
            D_W_fake = self.D_w(fake_windows)
            # generator has to fool the discriminator so classify fake W as real
            loss_G_2 = mse_loss(D_W_fake, torch.ones_like(D_W_fake).to(self.opts.device))

            #cycle loss / reconstructed loss
            if self.opts.use_cycle_loss:
                # #reconstructed apple from fake windows
                reconstructed_apple = self.G_w2a(fake_windows)
                cycle_loss_apple = nn.L1Loss()(apple_data, reconstructed_apple)
                #cycle_loss_apple = nn.MSELoss()(apple_data, reconstructed_apple)
                #cycle_loss_apple = torch.abs(reconstructed_apple - apple_data).sum() / apple_data.size(0)
                loss_G_2 += (self.opts.LAMBDA_CYCLE * cycle_loss_apple)

            G_loss = loss_G_1 + loss_G_2
            G_loss.backward()
            self.G_optim.step()

            #identity loss
            #identity_loss_apple = self.G_w2a(apple_data)
            #identity_loss_windows = self.G_a2w(windows_data)
            #identity_loss = identity_loss_apple +identity_loss_windows
            ##############################

            pbar.set_description('Epoch: {}, G_loss: {:.4f}, D_loss: {:.4f}, D_real_loss: {:.4f}, D_fake_loss: {:.4f}'.format(epoch, G_loss.item(), D_loss.item(), D_real_loss.item() , D_fake_loss.item()))


    def eval_step(self, epoch):
        #####TODO: generate 16 images from apple to windows and windows to apple from test data and save them in self.plotdir#####
        self.G_w2a.eval()
        self.G_a2w.eval()
        self.D_a.eval()
        self.D_w.eval()
        apple_loader   = iter(self.apple_testloader)
        windows_loader = iter(self.windows_testloader)
        num_iters = min(len(self.apple_testloader), len(self.windows_testloader)) // 6
        pbar = tqdm(range(num_iters))
        with torch.no_grad():
            for i in enumerate(pbar):
                #load data
                apple_data   = next(apple_loader).to(self.opts.device)
                windows_data = next(windows_loader).to(self.opts.device)
                # generate the fake images
                test_windows_fake = self.G_a2w(apple_data)
                test_apple_fake = self.G_w2a(windows_data)

                img_names = ['apple{}-epoch{}.png'.format(i,epoch), 'windows{}-epoch{}.png'.format(i,epoch)]
                # saves test image-to image translation
                self.save_samples(apple_data,   test_windows_fake, img_names[0], (self.plotdir + "/W2A/") )
                self.save_samples(windows_data, test_apple_fake, img_names[1], (self.plotdir + "/A2W/") )

    #       self.G_a2w.eval()
 #       noise = torch.randn((16, self.opts.noise_size)).to(self.opts.device)
 #       image = self.G_a2w(noise)
 #       image = image.reshape(16,-1).clamp(-1, 1).detach().cpu().numpy()
 #       image = ((image.reshape(16, 32, 32, -1) + 1) * 0.5 * 255).astype('uint8')
 #       image = np.concatenate(image, axis=0)
 #       Image.fromarray(image).save(os.path.join(self.plotdir, 'fig_a2w_{}.png'.format(epoch)))

 #       self.G_w2a.eval()
 #       noise = torch.randn((16, self.opts.noise_size)).to(self.opts.device)
 #       image = self.G_w2a(noise)
 #       image = image.reshape(16,-1).clamp(-1, 1).detach().cpu().numpy()
 #       image = ((image.reshape(16, 32, 32, -1) + 1) * 0.5 * 255).astype('uint8')
 #       image = np.concatenate(image, axis=0)
 #       Image.fromarray(image).save(os.path.join(self.plotdir, 'fig_w2a_{}.png'.format(epoch)))



    def save_step(self, epoch):
        #####TODO: save models in self.ckptdir#####
        checkpoint = {
            'D_a': [self.D_a.state_dict()],
            'D_w': [self.D_w.state_dict()],
            'G_w2a': [self.G_w2a.state_dict()],
            'G_a2w': [self.G_a2w.state_dict()],
        }
        torch.save(checkpoint, (self.ckptdir + "/{}_checkpoint.pt".format(epoch)))

        #G_a2w_path = os.path.join(self.ckptdir, 'G_a2w.pkl')
        #G_w2a_path = os.path.join(self.ckptdir, 'G_w2a.pkl')
        #D_A_path = os.path.join(self.ckptdir, 'D_A.pkl')
        #D_W_path = os.path.join(self.ckptdir, 'D_W.pkl')
        #torch.save(self.G_a2w.state_dict(), G_a2w_path)
        #torch.save(self.G_w2a.state_dict(), G_w2a_path)
        #torch.save(self.D_a.state_dict(), D_A_path)
        #torch.save(self.D_w.state_dict(), D_W_path)

    def merge_images(self, sources, targets, batch_size):
        # shape: (batch_size, num_channels, h, w)
        _, _, h, w = sources.shape
        rows = int(np.sqrt(batch_size))
        sample_grid = np.zeros([3, rows * w, rows * w * 2])
        for idx, (s, t) in enumerate(zip(sources, targets)):
            if idx >= rows ** 2:
                break
            i = idx // rows
            j = idx % rows
            sample_grid[:, i * h:(i + 1) * h, (j * 2) * h:(j * 2 + 1) * h] = s
            sample_grid[:, i * h:(i + 1) * h, (j * 2 + 1) * h:(j * 2 + 2) * h] = t
        return sample_grid.transpose((1, 2, 0))  # shape: (w, h, c)

    def save_samples(self, img_true, img_fake, img_name, folder_name):
        img_true = img_true.cpu().detach().numpy()
        img_fake = img_fake.cpu().detach().numpy()
        batch_size = img_true.shape[0]

        # generate grid of images
        sample_grid_win_fake = self.merge_images(img_true, img_fake, batch_size)

        # get path to save
        save_path = os.path.join(folder_name)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        path = os.path.join(save_path, img_name)

        # save images
        imageio.imwrite(path, (sample_grid_win_fake*255).astype(np.uint8))


if __name__ == '__main__':
    opts = CycleGanOptions()
    trainer = Trainer(opts)
    trainer.run()