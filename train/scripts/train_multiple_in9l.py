import subprocess
import os
import argparse

parser = argparse.ArgumentParser(description='Train multiple networks with a bit of possible out-of-the-script-configuration!')
parser.add_argument('-d', '--dataset-path', type=str, default='data/original/')
parser.add_argument('-s', '--save-directory', type=str, default='/home/macron/Praca Magisterska/Kod/augm_es_batch_test_no')
parser.add_argument('-l', '--log-dir', type=str, default='logs/')
parser.add_argument('-g', '--gpus', type=int, default=1)
parser.add_argument('-e', '--epochs', type=int, default=30)
parser.add_argument('-w', '--workers', type=int, default=4)
parser.add_argument('-b', '--use-background-blur', type=bool, default=False)
parser.add_argument('-t', '--use-background-transform', type=bool, default=False)
parser.add_argument('--use-auto-accuracy-background-transform', type=bool, default=False)
parser.add_argument('--use-auto-entropy-background-transform', type=bool, default=False)
parser.add_argument('--use-auto-top-entropy-background-transform', type=bool, default=False)
parser.add_argument('--reverse-transform-chances', type=bool, default=False)
parser.add_argument('--augmentation-dataset-size', type=float, default=0.2)
parser.add_argument('--backgrounds-path', type=str, default='data/only_bg_t')
parser.add_argument('--foregrounds-path', type=str, default='data/only_fg')
parser.add_argument('-c', '--chances', nargs='+', type=float)

args = parser.parse_args()

networks = {'resnet18': [0]}

epochs = args.epochs
save_dir = args.save_directory
transform_chances = args.chances
generic_args = ['env', 'PYTHONPATH=.', 'python3', './train/scripts/in9l_network.py', '-e', str(epochs), '-d', args.dataset_path, '--backgrounds-path', args.backgrounds_path, '--foregrounds-path', args.foregrounds_path, '-w', str(args.workers), '-l', args.log_dir, '-g', str(args.gpus),
	'--augmentation-dataset-size', str(args.augmentation_dataset_size)]

if args.use_auto_accuracy_background_transform:
	generic_args += ['--use-auto-accuracy-background-transform', 'true']

if args.use_auto_entropy_background_transform:
	generic_args += ['--use-auto-entropy-background-transform', 'true']

if args.use_auto_top_entropy_background_transform:
	generic_args += ['--use-auto-top-entropy-background-transform', 'true']

if args.use_background_transform:
	generic_args += ['-t', 'true']

if args.reverse_transform_chances:
    generic_args += ['--reverse-transform-chances', 'true']

for net in networks:
	os.makedirs(f'{save_dir}/{net}', exist_ok=True)
	for i in networks[net]:
		if transform_chances is not None:
			for j in transform_chances:
				print(f'Training {net} copy {i} - chance {j}!')
				os.makedirs(f'{save_dir}/{net}/{j}', exist_ok=True)
				in9l_network_args = generic_args + ['-n', net, '-t', 'true', '--background-transform-chance', str(j), '-p', f'{save_dir}/{net}/{j}/{i}.pkl']
				subprocess.run(in9l_network_args)
		elif args.use_background_blur:
			print(f'Training {net} copy {i}!')
			in9l_network_args = generic_args + ['-n', net, '--use-background-blur', 'true', '-p', f'{save_dir}/{net}/{i}.pkl']
			subprocess.run(in9l_network_args)
		else:
			print(f'Training {net} copy {i}!')
			in9l_network_args = generic_args + ['-n', net, '-p', f'{save_dir}/{net}/{i}.pkl']
			subprocess.run(in9l_network_args)
