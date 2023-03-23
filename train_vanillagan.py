import torch

from tqdm import tqdm
import os
import numpy as np
import matplotlib.pyplot as plt
from dataloader import get_data_loader
from models import Generator, Discriminator
from options import VanillaGANOptions
from torchvision.utils import save_image

from PIL import Image
import imageio

class Trainer:
    def __init__(self, opts):
        self.opts = opts

        #config dirs
        self.expdir = './vanilla_gan'
        self.plotdir = os.path.join(self.expdir, 'plots')
        self.ckptdir = os.path.join(self.expdir, 'checkpoints')

        os.makedirs(self.plotdir, exist_ok = True)
        os.makedirs(self.ckptdir, exist_ok = True)

        #config data
        self.trainloader, self.testloader = get_data_loader(self.opts.emoji_type, self.opts.batch_size, self.opts.num_workers)

        #config models
        self.G = Generator(self.opts).to( self.opts.device)
        self.D = Discriminator(self.opts).to(self.opts.device)

        #config optimizers
        self.G_optim = torch.optim.Adam(self.G.parameters(), lr = self.opts.lr, betas = (0.5, 0.999))
        self.D_optim = torch.optim.Adam(self.D.parameters(), lr = self.opts.lr, betas = (0.5, 0.999))

        self.criterion = torch.nn.BCELoss()

        #config training
        self.nepochs = self.opts.nepochs

    def run(self):
        for epoch in range(self.nepochs):
            self.train_step(epoch)

            if epoch % self.opts.eval_freq == 0:
                self.eval_step(epoch)
            if epoch % self.opts.save_freq == 0:
                self.save_checkpoint(epoch)



    def train_step(self, epoch):
        self.G.train()
        self.D.train()

        pbar = tqdm(self.trainloader)

        for i, data in enumerate(pbar):
            self.D_optim.zero_grad()
            self.G_optim.zero_grad()

            real= data.to(self.opts.device)
            noise = torch.randn(self.opts.batch_size, self.opts.noise_size, 1, 1).to(self.opts.device)
            fake = self.G(noise)

            #train discriminator  max log(D(x)) + log(1 - D(G(z)))
            #####TODO: compute discriminator loss and optimize#####
            disc_real = self.D(real).reshape(-1)
            real_labels = torch.ones_like(disc_real).to(self.opts.device)
            D_loss_real = self.criterion(disc_real, real_labels )
            #D_loss_real.backward()

            disc_fake = self.D(fake.detach()).reshape(-1)
            fake_labels = torch.zeros_like(disc_fake).to(self.opts.device)
            D_loss_fake = self.criterion(disc_fake, fake_labels)
            #D_loss_fake.backward()

            d_loss = (D_loss_real + D_loss_fake)/2
            d_loss.backward()
            self.D_optim.step()
            #d_loss = 0.
            ##########################################

            #train generator min log(1 - D(G(z))) <-> max log(D(G(z))
            #####TODO: compute generator loss and optimize#####
            #self.G.zero_grad()

            disc_fake = self.D(fake).reshape(-1)
            fake_labels = torch.ones_like(disc_fake).to(self.opts.device)
            g_loss = self.criterion(disc_fake, fake_labels)

            g_loss.backward()
            self.G_optim.step()
            #g_loss = 0.
            ###################################################
            pbar.set_description("Epoch: {}, Generator Loss: {:.4f}, Discriminator Loss: {:.4f}".format(epoch, g_loss.item(), d_loss.item()))


    def eval_step(self, epoch):
        #####TODO: sample from your test dataloader and save results in self.plotdir#####
        self.G.eval()
        self.D.eval()
        #pbar = tqdm(self.testloader)
        with torch.no_grad():
            noise = torch.rand((16, self.opts.noise_size)).unsqueeze(2).unsqueeze(3).to(self.opts.device)
            generated_images = self.G(noise)
            generated_images = (generated_images * 0.5) + 0.5
            save_image(generated_images, self.plotdir + "/save_image(0,1){}.png".format(epoch))
            #img_fake = generated_images.cpu().detach().numpy()
            #img_fake = (img_fake+1)*0.5 * 255
            #img_fake = img_fake.reshape(32,32,3).astype(np.uint8)
            #img_fake = Image.fromarray(img_fake, 'RGB')
            #imageio.imwrite((self.plotdir+"/{}.png".format(epoch)), img_fake,"png")


    def save_checkpoint(self, epoch):
        #####TODO: save your model in self.ckptdir#####
        """Saves the parameters of the generator G and discriminator D.
        """
        checkpoint = {
            "epoch": epoch,
            "generator_state_dict": self.G.state_dict(),
            "discriminator_state_dict": self.D.state_dict(),
            "optimizer_G_state_dict": self.G_optim.state_dict(),
            "optimizer_D_state_dict": self.D_optim.state_dict(),
        }
        torch.save(checkpoint, (self.ckptdir + "/{}_checkpoint.pt".format(epoch)))

if __name__ == '__main__':
    trainer = Trainer(VanillaGANOptions())
    trainer.run()