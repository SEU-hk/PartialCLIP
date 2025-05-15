# CIFAR10

python main.py -d cifar10 -m zsclip_vit_b16 infer_train True

python main.py -d cifar10 -m clip_vit_b16 -p 0.7 -l LWS adaptformer True pre_filter True


# CIFAR100

python main.py -d cifar100 -m zsclip_vit_b16 infer_train True

python main.py -d cifar100 -m clip_vit_b16 -p 0.1 -l LWS adaptformer True pre_filter True

python main.py -d cifar100 -m clip_vit_b16 -p 0.2 -l LWS adaptformer True pre_filter True


# Imagenet-lt

python main.py -d imagenet_lt -m zsclip_vit_b16 infer_train True

python main.py -d imagenet_lt -m clip_vit_b16 -p 0.1 -l RECORDS adaptformer True pre_filter True


# places-lt

python main.py -d places_lt -m zsclip_vit_b16 infer_train True

python main.py -d places_lt -m clip_vit_b16 -p 0.1 -l RECORDS adaptformer True pre_filter True