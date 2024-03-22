class Config(object):
    def __init__(self):
        # model configs
        self.num_classes = 23

        # training configs
        self.num_epoch = 1000# 40

        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr =  5e-5 # 3e-4

        # data parameters
        self.batch_size = 32
        self.drop_last = True

        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()

        """New hyperparameters"""
        self.TSlength_aligned = 1000
        self.lr_f = 5e-5
        self.target_batch_size = 100 # 82 # 41
        self.increased_dim = 1
        self.final_out_channels = 128
        self.num_classes_target = 23

class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 1.1
        self.jitter_ratio = 0.8
        self.max_seg = 8


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 6
