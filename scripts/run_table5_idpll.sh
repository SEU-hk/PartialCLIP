python main.py -d dogs120 -m clip_vit_b16 -p 2 -l POP -e 100 adaptformer True
python main.py -d dogs120 -m head_init -p 2 -l POP -e 100 adaptformer True

python main.py -d dogs120 -m clip_vit_b16 -p 2 -l POP -e 100 adapter True

python main.py -d dogs120 -m clip_vit_b16 -p 2 -l POP -e 100 vpt_deep True
python main.py -d dogs120 -m head_init -p 2 -l POP -e 100 vpt_deep True

python main.py -d dogs120 -m clip_vit_b16 -p 2 -l POP -e 100 vpt_shallow True

python main.py -d dogs120 -m clip_vit_b16 -p 2 -l POP -e 100 bias_tuning True

python main.py -d dogs120 -m clip_vit_b16 -p 2 -l POP -e 100 lora True

python main.py -d dogs120 -m clip_vit_b16 -p 2 -l POP -e 100 linear_probing True

python main.py -d dogs120 -m clip_vit_b16 -p 2 -l POP -e 100 -lr 0.001 full_tuning True lr 0.001



python main.py -d fgvc100 -m clip_vit_b16 -p 2 -l POP -e 200 adaptformer True
python main.py -d fgvc100 -m head_init -p 2 -l POP -e 200 adaptformer True

python main.py -d fgvc100 -m clip_vit_b16 -p 2 -l POP -e 200 adapter True

python main.py -d fgvc100 -m clip_vit_b16 -p 2 -l POP -e 200 vpt_deep True
python main.py -d fgvc100 -m head_init -p 2 -l POP -e 200 vpt_deep True

python main.py -d fgvc100 -m clip_vit_b16 -p 2 -l POP -e 200 vpt_shallow True

python main.py -d fgvc100 -m clip_vit_b16 -p 2 -l POP -e 200 bias_tuning True

python main.py -d fgvc100 -m clip_vit_b16 -p 2 -l POP -e 200 lora True

python main.py -d fgvc100 -m clip_vit_b16 -p 2 -l POP -e 200 linear_probing True

python main.py -d fgvc100 -m clip_vit_b16 -p 2 -l POP -e 200 -lr 0.002 full_tuning True lr 0.002