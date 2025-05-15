import numpy as np
import hashlib


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

    if dataset in RESNET_DATASET:
        _hparam('model', 'ResNet', lambda r: 'ResNet')

    _hparam('lr', 1e-3, lambda r: 10**r.uniform(-4.5, -2.5))

    _hparam('weight_decay', 1e-5, lambda r: 10**r.uniform(-6, -3))   
    if dataset in RESNET_DATASET:
        _hparam('batch_size', 256, lambda r: 2**int(r.uniform(6, 9)))

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

    elif algorithm == 'ABLE':
        _hparam('dim',128,lambda r: r.choice([64,128,256]))
        _hparam('loss_weight',1.0,lambda r: r.choice([0.5,1.0,2.0]))
        _hparam('temperature',0.07,lambda r:r.choice([0.03,0.05,0.07,0.09]))
            
    elif algorithm == 'Solar':
        _hparam('warmup_epoch', 2, lambda r: r.choice([1, 2]))
        _hparam('queue_length', 64, lambda r:r.choice([64, 128]))
        _hparam('lamd', 3, lambda r:r.choice([1.0]))
        _hparam('eta', 0.9, lambda r:r.choice([1.0]))
        _hparam('rho_start', 0.2, lambda r:r.choice([1.0]))
        _hparam('rho_end', 0.8, lambda r:r.choice([1.0]))
        _hparam('tau', 0.99, lambda r:r.choice([1.0]))
        
    return hparams


def default_hparams(algorithm, dataset):
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}


def random_hparams(algorithm, dataset, seed):
    return {a: c for a, (b, c) in _hparams(algorithm, dataset, seed).items()}
