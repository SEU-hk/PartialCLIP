# CIFAR10
python main.py -d cifar10 -m clip_vit_b16 -p 0.3 -l POP adaptformer True
python main.py -d cifar10 -m head_init -p 0.3 -l POP adaptformer True

python main.py -d cifar10 -m clip_vit_b16 -p 0.3 -l POP adapter True

python main.py -d cifar10 -m clip_vit_b16 -p 0.3 -l POP vpt_deep True
python main.py -d cifar10 -m head_init -p 0.3 -l POP vpt_deep True

python main.py -d cifar10 -m clip_vit_b16 -p 0.3 -l POP vpt_shallow True

python main.py -d cifar10 -m clip_vit_b16 -p 0.3 -l POP bias_tuning True

python main.py -d cifar10 -m clip_vit_b16 -p 0.3 -l POP lora True

python main.py -d cifar10 -m clip_vit_b16 -p 0.3 -l POP linear_probing True

python main.py -d cifar10 -m clip_vit_b16 -p 0.3 -l POP -lr 0.0005 full_tuning True lr 0.0005



# CIFAR100
python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l POP adaptformer True
python main.py -d cifar100 -m head_init -p 0.1 -l POP adaptformer True

python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l POP adapter True

python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l POP vpt_deep True
python main.py -d cifar100 -m head_init -p 0.1 -l POP vpt_deep True

python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l POP vpt_shallow True

python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l POP bias_tuning True

python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l POP lora True

python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l POP linear_probing True

python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l POP -lr 0.0005 full_tuning True lr 0.0005




# CUB200

python main.py -d cub200 -m clip_vit_b16 -p 0.01 -l POP -e 20 adaptformer True
python main.py -d cub200 -m head_init -p 0.01 -l POP -e 20 adaptformer True

python main.py -d cub200 -m clip_vit_b16 -p 0.01 -l POP -e 20 adapter True

python main.py -d cub200 -m clip_vit_b16 -p 0.01 -l POP -e 20 vpt_deep True
python main.py -d cub200 -m head_init -p 0.01 -l POP -e 20 vpt_deep True

python main.py -d cub200 -m clip_vit_b16 -p 0.01 -l POP -e 20 vpt_shallow True

python main.py -d cub200 -m clip_vit_b16 -p 0.01 -l POP -e 20 bias_tuning True

python main.py -d cub200 -m clip_vit_b16 -p 0.01 -l POP -e 20 lora True

python main.py -d cub200 -m clip_vit_b16 -p 0.01 -l POP -e 20 linear_probing True

python main.py -d cub200 -m clip_vit_b16 -p 0.01 -l POP -e 20 -lr 0.001 full_tuning True lr 0.001




# CARS196

python main.py -d car196 -m clip_vit_b16 -p 0.01 -l POP -e 60 adaptformer True
python main.py -d car196 -m head_init -p 0.01 -l POP -e 60 adaptformer True

python main.py -d car196 -m clip_vit_b16 -p 0.01 -l POP -e 60 adapter True

python main.py -d car196 -m clip_vit_b16 -p 0.01 -l POP -e 60 vpt_deep True
python main.py -d car196 -m head_init -p 0.01 -l POP -e 60 vpt_deep True

python main.py -d car196 -m clip_vit_b16 -p 0.01 -l POP -e 60 vpt_shallow True

python main.py -d car196 -m clip_vit_b16 -p 0.01 -l POP -e 60 bias_tuning True

python main.py -d car196 -m clip_vit_b16 -p 0.01 -l POP -e 60 lora True

python main.py -d car196 -m clip_vit_b16 -p 0.01 -l POP -e 60 linear_probing True

python main.py -d car196 -m clip_vit_b16 -p 0.01 -l POP -e 60 -lr 0.001 full_tuning True lr 0.001