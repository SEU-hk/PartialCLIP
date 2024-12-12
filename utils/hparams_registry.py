import numpy as np
import hashlib

MLP_DATASET = [
    "FGNET",
    "Lost",
    "MSRCv2",
    "Mirflickr",
    "Birdsong",
    "Malagasy",
    "SoccerPlayer",
    "Italian",
    "YahooNews",
    "English"
]

LENET_DATASET = [
    "SynMNIST",
    "SynKMNIST",
    "SynFashion"
]

RESNET_DATASET = [
    "PLCIFAR10",
    "CLCIFAR10",
    "CLCIFAR20",
    "SynCIFAR10",
    "PLCIFAR10_Aggregate",
    "PLCIFAR10_Vaguest",
    "PLCIFAR10_Vaguest_Syn"
]

def seed_hash(*args):
    """
    Derive an integer hash from all args, for use as a random seed.
    """
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2**31)


def _define_hparam(hparams, hparam_name, default_val, random_val_fn):
    hparams[hparam_name] = (hparams, hparam_name, default_val, random_val_fn)


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """

    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))

    # Unconditional hparam definitions.

    if dataset in MLP_DATASET:
        _hparam('model', 'MLP', lambda r: 'MLP')
    elif dataset in LENET_DATASET:
        _hparam('model', 'LeNet', lambda r: 'LeNet')
    elif dataset in RESNET_DATASET:
        _hparam('model', 'ResNet', lambda r: 'ResNet')

    _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))

    _hparam('weight_decay', 1e-5, lambda r: 10**r.uniform(-6, -3))
    if dataset in RESNET_DATASET:
        _hparam('batch_size', 256, lambda r: 2**int(r.uniform(6, 9)))
    elif dataset in MLP_DATASET:
        _hparam('batch_size', 128, lambda r: 2**int(r.uniform(5, 8)))

    # algorithm-specific hyperparameters
    if algorithm == 'LWS':
        _hparam('lw_weight', 2, lambda r: r.choice([1, 2]))
    elif algorithm == 'POP':
        _hparam('rollWindow', 5, lambda r: r.choice([3, 4, 5, 6, 7]))
        _hparam('warm_up', 20, lambda r: r.choice([10, 15, 20]))
        _hparam('theta', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
        _hparam('inc', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))
        _hparam('pop_interval', 1000, lambda r: r.choice([500, 1000, 1500, 2000]))
    elif algorithm == 'IDGP':
        _hparam('warm_up_epoch', 10, lambda r: r.choice([5, 10, 15, 20]))
    elif algorithm == 'ABS_GCE':
        _hparam('q', 0.7, lambda r: 0.7)
    elif algorithm == 'DIRK':
        _hparam('momentum',0.99, lambda r: r.choice([0.5,0.9,0.99]))
        _hparam('moco_queue',8192,lambda r:r.choice([4096,8192]))
        _hparam('feat_dim', 128, lambda r: r.choice([64, 128, 256]))
        _hparam('prot_start',178,lambda r:r.choice([178]))
        _hparam('weight',1.0,lambda r: r.choice([0.1,0.3,0.5,0.7,0.9,1.0]))
        _hparam('dist_temperature',0.4,lambda r: r.choice([0.2,0.4,0.6]))
        _hparam('feat_temperature', 0.07, lambda r: r.choice([0.03, 0.05, 0.07, 0.09]))
    elif algorithm == 'ABLE':
        _hparam('feat_dim',128,lambda r: r.choice([64,128,256]))
        _hparam('loss_weight',1.0,lambda r: r.choice([0.5,1.0,2.0]))
        _hparam('temperature',0.07,lambda r:r.choice([0.03,0.05,0.07,0.09]))
    elif algorithm == 'PiCO':
        _hparam('prot_start',178,lambda r:r.choice([178]))
        _hparam('momentum', 0.9, lambda r: r.choice([0.8,0.9]))
        _hparam('feat_dim',128,lambda r: r.choice([64,128,256]))
        _hparam('moco_queue',8192,lambda r:r.choice([4096,8192]))
        _hparam('moco_m',0.999,lambda r:r.choice([0.9,0.999]))
        _hparam('proto_m',0.99,lambda r:r.choice([0.99,0.9]))
        _hparam('loss_weight',0.5,lambda r:r.choice([0.5,1.0]))
        _hparam('conf_ema_range','0.95,0.8',lambda r:r.choice(['0.95,0.8']))
    elif algorithm == 'VALEN':
        _hparam('warm_up', 1000, lambda r: r.choice([1000,2000, 3000, 4000, 5000]))
        _hparam('knn',3, lambda r:r.choice([3,4]))
        _hparam('alpha',1.0,lambda r:r.choice([1.0]))
        _hparam('beta',1.0,lambda r:r.choice([1.0]))
        _hparam('lambda',1.0,lambda r:r.choice([1.0]))
        _hparam('gamma',1.0,lambda r:r.choice([1.0]))
        _hparam('theta',1.0,lambda r:r.choice([1.0]))
        _hparam('sigma',1.0,lambda r:r.choice([1.0]))
        _hparam('correct',1.0,lambda r:r.choice([1.0]))
    elif algorithm == 'NN':
        _hparam('beta', 0., lambda r: 0.)

    elif algorithm == 'FREDIS':
        _hparam('theta', 1e-6, lambda r: 1e-6)
        _hparam('inc', 1e-6, lambda r: r.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]))
        _hparam('delta', 1.0, lambda r: 1.0)
        _hparam('dec', 1e-6, lambda r: r.choice([1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]))
        _hparam('times', int(2), lambda r: r.choice([int(1), int(2), int(3), int(4), int(5)]))
        _hparam('change_size', int(500), lambda r: r.choice([int(50), int(100), int(500), int(300), int(800), int(1000)]))
        _hparam('update_interval', int(20), lambda r: r.choice([int(10), int(20)]))
        _hparam('lam', 10, lambda r: r.choice([1, 10]))
        _hparam('alpha', 1.0, lambda r: r.choice([1e-3, 1e-2, 1e-1, 1]))
    elif algorithm == 'PiCO_plus':
        _hparam('prot_start', 250, lambda r: r.choice([1,250]))
        _hparam('momentum', 0.9, lambda r: r.choice([0.8, 0.9]))
        _hparam('feat_dim', 128, lambda r: r.choice([64, 128, 256]))
        _hparam('moco_queue', 8192, lambda r: r.choice([4096, 8192]))
        _hparam('moco_m', 0.999, lambda r: r.choice([0.9, 0.999]))
        _hparam('proto_m', 0.99, lambda r: r.choice([0.99, 0.9]))
        _hparam('loss_weight', 0.5, lambda r: r.choice([0.5, 1.0]))
        _hparam('conf_ema_range', '0.95,0.8', lambda r: r.choice(['0.95,0.8']))
        if dataset == 'PLCIFAR10_Aggregate':
            _hparam('pure_ratio', 1 - 0.00132, lambda r: 1 - 0.00132)
        elif dataset == 'PLCIFAR10_Vaguest':
            _hparam('pure_ratio', 1 - 0.17556, lambda r: 1 - 0.17556)
        else:
            _hparam('pure_ratio', 1, lambda r: r.choice([0.6,0.8,0.95,0.99,1]))
        _hparam('knn_start',5000,lambda r:r.choice([2,5000,100]))
        _hparam('chosen_neighbors',16,lambda r:r.choice([8,16]))
        _hparam('temperature_guess',0.07,lambda r:r.choice([0.1,0.07]))
        _hparam('ur_weight',0.1,lambda r:r.choice([0.1,0.5]))
        _hparam('cls_weight',2,lambda r:r.choice([2,3,5]))
    elif algorithm == 'ALIM':
        _hparam('prot_start',178,lambda r:r.choice([178]))
        _hparam('momentum', 0.9, lambda r: r.choice([0.8,0.9]))
        _hparam('feat_dim',128,lambda r: r.choice([64,128,256]))
        _hparam('moco_queue',8192,lambda r:r.choice([4096,8192]))
        _hparam('moco_m',0.999,lambda r:r.choice([0.9,0.999]))
        _hparam('proto_m',0.99,lambda r:r.choice([0.99,0.9]))
        _hparam('loss_weight',0.5,lambda r:r.choice([0.5,1.0]))
        _hparam('conf_ema_range','0.95,0.8',lambda r:r.choice(['0.95,0.8']))

        _hparam('start_epoch', 40, lambda r: r.choice([20, 40, 80, 100, 140]))
        _hparam('loss_weight_mixup', 1.0, lambda r: 1.0)
        _hparam('mixup_alpha', 4, lambda r: 4)
        if dataset == 'PLCIFAR10_Aggregate':
            _hparam('noise_rate', 0.00132, lambda r: 0.00132)
        elif dataset == 'PLCIFAR10_Vaguest':
            _hparam('noise_rate', 0.17556, lambda r: 0.17556)
        else:
            _hparam('noise_rate', 0.2, lambda r: r.choice([0.1, 0.2, 0.3, 0.4, 0.5, 0.7]))

    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
