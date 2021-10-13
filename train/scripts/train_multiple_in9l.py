import subprocess
import os

networks = ['resnet26', 'resnet34', 'resnet50', 'mobilenetv2_100', 'vgg13']
copies_per_network = 10
epochs = 50

for net in networks:
	os.makedir(f'networks/{net}', exist_ok=True)
	for i in range(copies_per_network):
		subprocess.run(['env', 'PYTHONPATH=.', 'python3', './train/scripts/in9l_network.py', '-n', net, '-e', str(epochs), '-p', f'networks/{net}/{i}.pkl'])
