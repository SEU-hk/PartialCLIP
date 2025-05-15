python main.py -d cifar100_ir100 -m clip_vit_b16 -p 0.1 -l RECORDS -e 10 adaptformer True
python main.py -d cifar100_ir100 -m head_init -p 0.1 -l RECORDS -e 10 adaptformer True

python main.py -d cifar100_ir100 -m clip_vit_b16 -p 0.1 -l RECORDS -e 10 adapter True

python main.py -d cifar100_ir100 -m clip_vit_b16 -p 0.1 -l RECORDS -e 10 vpt_deep True
python main.py -d cifar100_ir100 -m head_init -p 0.1 -l RECORDS -e 10 vpt_deep True

python main.py -d cifar100_ir100 -m clip_vit_b16 -p 0.1 -l RECORDS -e 10 vpt_shallow True

python main.py -d cifar100_ir100 -m clip_vit_b16 -p 0.1 -l RECORDS -e 10 bias_tuning True

python main.py -d cifar100_ir100 -m clip_vit_b16 -p 0.1 -l RECORDS -e 10 lora True

python main.py -d cifar100_ir100 -m clip_vit_b16 -p 0.1 -l RECORDS -e 10 linear_probing True

python main.py -d cifar100_ir100 -m clip_vit_b16 -p 0.1 -l RECORDS -e 10 -lr 0.0005 full_tuning True lr 0.0005



python main.py -d places_lt -m clip_vit_b16 -p 0.05 -l RECORDS -e 10 adaptformer True
python main.py -d places_lt -m head_init -p 0.05 -l RECORDS -e 10 adaptformer True

python main.py -d places_lt -m clip_vit_b16 -p 0.05 -l RECORDS -e 10 adapter True

python main.py -d places_lt -m clip_vit_b16 -p 0.05 -l RECORDS -e 10 vpt_deep True
python main.py -d places_lt -m head_init -p 0.05 -l RECORDS -e 10 vpt_deep True

python main.py -d places_lt -m clip_vit_b16 -p 0.05 -l RECORDS -e 10 vpt_shallow True

python main.py -d places_lt -m clip_vit_b16 -p 0.05 -l RECORDS -e 10 bias_tuning True

python main.py -d places_lt -m clip_vit_b16 -p 0.05 -l RECORDS -e 10 lora True

python main.py -d places_lt -m clip_vit_b16 -p 0.05 -l RECORDS -e 10 linear_probing True

python main.py -d places_lt -m clip_vit_b16 -p 0.05 -l RECORDS -e 10 -lr 0.0005 full_tuning True lr 0.0005