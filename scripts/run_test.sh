# CIFAR10

python main_PE.py -d cifar10 -m zsclip_vit_b16 infer_train True

python main_PE.py -d cifar10 -m clip_vit_b16 -p 0.5 -l CC adaptformer True 

python main_PE.py -d cifar10 -m clip_vit_b16 -p 0.5 -l PRODEN adaptformer True 

python main_PE.py -d cifar10 -m clip_vit_b16 -p 0.5 -l CRDPLL adaptformer True 


# CIFAR100

python main_PE.py -d cifar100 -m zsclip_vit_b16 infer_train True

python main_PE.py -d cifar100 -m clip_vit_b16 -p 0.1 -l CC adaptformer True 

python main_PE.py -d cifar100 -m clip_vit_b16 -p 0.1 -l PRODEN adaptformer True 

python main_PE.py -d cifar100 -m clip_vit_b16 -p 0.1 -l CRDPLL adaptformer True