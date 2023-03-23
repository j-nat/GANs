import torch
import torch.nn as nn
import torch.nn.functional as F



def conv_layer(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    """Creates a convolutional layer, with optional batch normalization.
    """
    layers = []
    conv_layer = nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, padding = padding, bias = False)
    layers.append(conv_layer)
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)

def deconv_layer(in_channels, out_channels, kernel_size, stride = 2, padding = 1, batch_norm = True):
    """Creates a transposed-convolutional layer, with optional batch normalization.
    """
    layers = []
    layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias = False))
    if batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))
    return nn.Sequential(*layers)


class ResNetBlock(nn.Module):
    def __init__(self, channels):
        super(ResNetBlock, self).__init__()
        self.conv = conv_layer(in_channels = channels, out_channels = channels, kernel_size = 3, stride = 1, padding = 1, batch_norm = True)
    def forward(self, x):
        return x + self.conv(x)



class Discriminator(nn.Module):
    def __init__(self, opts):
        super(Discriminator, self).__init__()
        self.opts = opts

        #####TODO: Define the discriminator network#####
        self.d_conv1 = conv_layer(in_channels=3, out_channels= self.opts.discriminator_channels[0], kernel_size=4)
        self.d_conv2 = conv_layer(in_channels=self.opts.discriminator_channels[0], out_channels=self.opts.discriminator_channels[1], kernel_size=4)
        self.d_conv3 = conv_layer(in_channels=self.opts.discriminator_channels[1], out_channels=self.opts.discriminator_channels[2], kernel_size=4)
        self.d_conv4 = conv_layer(in_channels=self.opts.discriminator_channels[2], out_channels=self.opts.discriminator_channels[3], kernel_size=4, stride=1, padding=0, batch_norm=False)

        ################################################


    def forward(self, x):
        #####TODO: Define the forward pass#####
        out1 = F.leaky_relu(self.d_conv1(x), negative_slope=0.2, inplace=True)
        #out1 = F.dropout(out1, p=0.3)
        out2 = F.leaky_relu(self.d_conv2(out1), negative_slope=0.2, inplace=True)
        #out2 = F.dropout(out2, p=0.3)
        out3 = F.leaky_relu(self.d_conv3(out2), negative_slope=0.2, inplace=True)
        #out3 = F.dropout(out3, p=0.3)
        out = self.d_conv4(out3)

        #out = self.d_conv4(out3).squeeze()
        out = torch.sigmoid(out)
        return out
        #######################################


class Generator(nn.Module):
    def __init__(self, opts):
        super(Generator, self).__init__()
        self.opts = opts

        #####TODO: Define the generator network######
        self.g_conv1 = deconv_layer(in_channels=self.opts.noise_size, out_channels=self.opts.generator_channels[0], kernel_size=4, stride=1, padding=0)
        self.g_conv2 = deconv_layer(in_channels=self.opts.generator_channels[0], out_channels=self.opts.generator_channels[1], kernel_size=4)
        self.g_conv3 = deconv_layer(in_channels=self.opts.generator_channels[1], out_channels=self.opts.generator_channels[2], kernel_size=4)
        self.g_conv4 = deconv_layer(in_channels=self.opts.generator_channels[2], out_channels=self.opts.generator_channels[3], kernel_size=4, batch_norm=False)

    #############################################

    def forward(self, x):
        #####TODO: Define the forward pass#####
        out1 = F.relu(self.g_conv1(x), inplace=True)
        out2 = F.relu(self.g_conv2(out1), inplace=True)
        out3 = F.relu(self.g_conv3(out2), inplace=True)
        #out1 = F.leaky_relu(self.g_conv1(x), negative_slope=0.2)
        #out2 = F.leaky_relu(self.g_conv2(out1), negative_slope=0.2)
        #out3 = F.leaky_relu(self.g_conv3(out2), negative_slope=0.2)
        out = self.g_conv4(out3)
        res = torch.tanh(out)

        return res
        #######################################

class CycleGenerator(nn.Module):
    def __init__(self, opts):
        super(CycleGenerator, self).__init__()
        self.opts = opts

        #####TODO: Define the cyclegan generator network######
        #encoder part of the generator to extract features from the input image
        self.cycle_conv1 = conv_layer(in_channels=3, out_channels=self.opts.generator_channels[0], kernel_size=4)
        self.cycle_conv2 = conv_layer(in_channels=self.opts.generator_channels[0], out_channels=self.opts.generator_channels[1], kernel_size=4)
        #transformation
        self.res_block = ResNetBlock(channels=self.opts.generator_channels[1])

        #decoder part to build the image from the features
        self.cycle_conv3 = deconv_layer(in_channels=self.opts.generator_channels[1], out_channels=self.opts.generator_channels[0], kernel_size=4)
        self.cycle_conv4 = deconv_layer(in_channels=self.opts.generator_channels[0], out_channels=3, kernel_size=4)
        ######################################################


    def forward(self, x):
        #####TODO: Define the forward pass#####
        out1 = F.leaky_relu(self.cycle_conv1(x), negative_slope=0.2)
        out2 = F.leaky_relu(self.cycle_conv2(out1), negative_slope=0.2)
        out3 = self.res_block(out2)
        out4 = F.leaky_relu(self.cycle_conv3(out3), negative_slope=0.2)
        out = torch.tanh(self.cycle_conv4(out4))

        return out
        #######################################


