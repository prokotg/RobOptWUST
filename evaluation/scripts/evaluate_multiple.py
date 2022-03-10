import subprocess
import argparse
import os


parser = argparse.ArgumentParser(description='Evaluates multiple networks on any datasets, creating logs to be later read!')
parser.add_argument('-n', '--networks-path', type=str, required=True, help="Path for networks to be read from")
parser.add_argument('-l', '--logs-path', type=str, required=True, help="Path for logs to be written in")
parser.add_argument('--parent-dataset-path', type=str, required=True, help="Path to a folder with all of datasets")
parser.add_argument('--is-parametrized', type=bool, default=False)

args = parser.parse_args()

networks_dir = args.networks_path
logs_dir = args.logs_path
is_parametrized = args.is_parametrized

print(networks_dir)
for net_type in os.listdir(networks_dir):
	if os.path.isdir(f'{networks_dir}/{net_type}'):
		os.makedirs(f'{logs_dir}/{net_type}/', exist_ok=True)
		for model_file in os.listdir(f'{networks_dir}/{net_type}'):
			if is_parametrized:
				os.makedirs(f'{logs_dir}/{net_type}/{model_file}', exist_ok=True)
				for model_example in os.listdir(f'{networks_dir}/{net_type}/{model_file}'):
					print(f'Evaluating {net_type} param {model_file} file {model_example}!')
					subprocess.run(['env', 'PYTHONPATH=.', 'python3', './evaluation/scripts/evaluate.py', '-p', f'{networks_dir}/{net_type}/{model_file}/{model_example}', '-l', f'{logs_dir}/{net_type}/{model_file}/{model_example}.log', '--parent-dataset-path', args.parent_dataset_path])
			else:
				print(f'Evaluating {net_type} copy {model_file}!')
				subprocess.run(['env', 'PYTHONPATH=.', 'python3', './evaluation/scripts/evaluate.py', '-p', f'{networks_dir}/{net_type}/{model_file}', '-l', f'{logs_dir}/{net_type}/{model_file}.log', '--parent-dataset-path', args.parent_dataset_path])
