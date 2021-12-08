import subprocess
import os
# 'resnet26' : 5, 'resnet34' : 5, 'vgg13': 5, 
networks = { 'mobilenetv2_100' : range(0, 3), 'resnet26' : range(0, 3), 'resnet34' : range(0, 3), 'resnet50': range(0)}
epochs = 80

save_dir = '/home/macron/Praca Magisterska/Kod/augm_es_auto_batch'
transform_chances = None # [1.0, 0.8, 0.5, 0.25, 0.05, 0.0]
auto_transform = True
# env PYTHONPATH=. python3 ./train/scripts/in9l_network.py -n resnet34 -e 40 -p /home/maciejziolkowski/storage/robust_networks/test.pkl
for net in networks:
	os.makedirs(f'{save_dir}/{net}', exist_ok=True)
	for i in networks[net]:
		if transform_chances is not None:
			for j in transform_chances:
				print(f'Training {net} copy {i} - chance {j}!')
				os.makedirs(f'{save_dir}/{net}/{j}', exist_ok=True)
				subprocess.run(['env', 'PYTHONPATH=.', 'python3', './train/scripts/in9l_network.py', '-n', net, '-e', str(epochs), '-t', 'true', '--background-transform-chance', str(j), '-p', f'{save_dir}/{net}/{j}/{i}.pkl'])
		elif auto_transform == True:
			subprocess.run(['env', 'PYTHONPATH=.', 'python3', './train/scripts/in9l_network.py', '-n', net, '-e', str(epochs), '-t', 'true', '--use-auto-background-transform', 'true', '-p', f'{save_dir}/{net}/{i}.pkl'])
		else:
			print(f'Training {net} copy {i}!')
			subprocess.run(['env', 'PYTHONPATH=.', 'python3', './train/scripts/in9l_network.py', '-n', net, '-e', str(epochs), '-p', f'{save_dir}/{net}/{i}.pkl'])
