import abc

class BaseOptions(metaclass=abc.ABCMeta):
    def __init__(self):
        self.device = 'cpu'
        self.expdir = ''
        self.debug = True


class VanillaGANOptions(BaseOptions):
    def __init__(self):
        super(VanillaGANOptions, self).__init__()

        #Dataset options
        self.data_dir = './emojis'
        self.emoji_type = 'Apple'
        self.batch_size = 16
        self.num_workers = 1

        #Discriminator options
        self.discriminator_channels = [32, 64, 128, 1]

        #Generator options
        self.generator_channels = [128, 64, 32, 3]
        self.noise_size = 100

        #Training options
        self.nepochs = 1000
        self.lr = 0.0002

        self.eval_freq = 20

        self.save_freq = 20

class CycleGanOptions(BaseOptions):
    def __init__(self):
        super(CycleGanOptions, self).__init__()

        #Generator options
        self.generator_channels = [32, 64]

        # Dataset options
        self.data_dir = './emojis'
        self.batch_size = 8
        self.num_workers = 1

        # Discriminator options
        self.discriminator_channels = [32, 64, 128, 1]

        # Training options
        self.niters = 1000
        self.lr = 0.0003

        self.eval_freq = 50
        self.save_freq = 50

        self.LAMBDA_CYCLE = 10
        self.use_cycle_loss = True
