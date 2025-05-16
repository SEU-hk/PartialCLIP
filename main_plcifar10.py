import os
import random
import argparse
import numpy as np
import torch

from utils.config import _C as cfg
from utils.logger import setup_logger
from utils.hparams_registry import *

from trainer_plcifar10 import Trainer

def main(args):
    cfg_data_file = os.path.join("./configs/data", args.data + ".yaml")
    cfg_model_file = os.path.join("./configs/model", args.model + ".yaml")

    cfg.defrost()
    cfg.merge_from_file(cfg_data_file)
    print(cfg_model_file)
    cfg.merge_from_file(cfg_model_file)
    cfg.merge_from_list(args.opts)
    cfg['partial_rate'] = args.partial_rate
    cfg['loss_type'] = args.loss_type
    cfg['num_epochs'] = args.num_epochs
    cfg['gpu'] = args.gpu
    hparams = default_hparams(args.loss_type, args.data)
    for key, value in hparams.items():
        cfg[key] = value
        
    cfg['lr'] = args.learning_rate
    
    if cfg.pre_filter == True:
        path_to_confidence = f'confidence/{cfg.dataset}.pth'
        cfg['zsclip'] = torch.load(path_to_confidence)

    # cfg.freeze()

    if cfg.output_dir is None:
        cfg_name = "_".join([args.data, "pr" + str(args.partial_rate), cfg['loss_type'], args.model])
        opts_name = "".join(["_" + item for item in args.opts])
        cfg.output_dir = os.path.join("./output", cfg_name + opts_name)
    else:
        cfg.output_dir = os.path.join("./output", cfg.output_dir)
    print("Output directory: {}".format(cfg.output_dir))
    setup_logger(cfg.output_dir)
    
    print("** Config **")
    print(cfg)
    print("************")
    
    if cfg.seed is not None:
        seed = cfg.seed
        print("Setting fixed seed: {}".format(seed))
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if cfg.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    trainer = Trainer(cfg)
    
    if cfg.model_dir is not None:
        trainer.load_model(cfg.model_dir)
    
    if cfg.zero_shot:
        if cfg.infer_train == True:
            trainer.test_confidence("train")
        else:
            trainer.test()
        return

    
    if cfg.test_train == True:
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[:cfg.output_dir.index("_test_train_True")]
            print("Model directory: {}".format(cfg.model_dir))

        trainer.load_model(cfg.model_dir)
        
        trainer.test("train")
        return

    if cfg.test_only == True:
        if cfg.model_dir is None:
            cfg.model_dir = cfg.output_dir[:cfg.output_dir.index("_test_only_True")]
            print("Model directory: {}".format(cfg.model_dir))
        
        trainer.load_model(cfg.model_dir)
        
        trainer.test1()
        return

    trainer.train()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", type=str, default="", help="data config file")
    parser.add_argument("--model", "-m", type=str, default="", help="model config file")
    parser.add_argument("--partial_rate", "-p", type=float, default=0.1, help="partial rate")
    parser.add_argument("--loss_type", "-l", type=str, default="Proden", help="loss type")
    parser.add_argument("--num_epochs", "-e", type=int, default="10", help="epochs")
    parser.add_argument("--gpu", "-g", type=int, default=1, help="gpu_id")
    parser.add_argument("--learning_rate","-lr", type=float, default=0.0005, help="learning_rate")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="modify config options using the command-line")


    args = parser.parse_args()
    main(args)
