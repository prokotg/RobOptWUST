import argparse
import pickle
import os

from tqdm import tqdm

import data.imagenet as ImageNet9

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate any network on only_fg vs original!')
    parser.add_argument('-p', '--model-path', type=str, default='trained_model.pkl', required=True)
    parser.add_argument('-f', '--parent-dataset-path', type=str, default='data/', required=True)
    parser.add_argument('-l', '--log-filename', type=str, default='evaluate_log/evaluation.log')
    parser.add_argument('-w', '--workers', type=int, default=4)
    args = parser.parse_args()

    model_file = open(args.model_path, 'rb')
    model = pickle.load(model_file)
    model_file.close()
    model.cuda()


def log_to_file(filename, lines):
    f = open(filename, 'a')
    f.writelines(lines)
    f.close()


def out_to_lines(out):
    lines = []
    for (n, i1, i2, succs, paths) in out:
        line = f'{n};{i1};{i2}'
        for succ in succs:
            line += ';' + str(succ)
        for path in paths:
            line += ';' + path
        lines.append(f"{line};\n")
    return lines


def clear_files(filename, datasets):
    f = open(filename, 'w')
    for v in ('class', 'nr', 'id'):
        f.write(f'{v};')
    paths = []
    for dataset in datasets:
        f.write(f"{dataset.replace('/', '')}_loader;")
        paths.append(f"{dataset.replace('/', '')}_path;")
    for path in paths:
        f.write(path)
    f.write('\n')
    f.close()


def eval_model_averages(loaders, model):
    model = model.eval()
    iterator = tqdm(enumerate(zip(*loaders)), total=len(loaders[0]))
    corrects = [0] * len(loaders)
    for chunk_index, data in iterator:
        for i, ((c_paths, inp), target) in enumerate(data):
            output = model(inp.cuda())
            if type(output) is tuple:
                output = output[0]
            _, pred = output.topk(1, 1, True, True)
            pred = pred.cpu().detach()[:, 0]
            for index, (tr, p) in enumerate(zip(target, pred)):
                if (p == tr).item():
                    corrects[i] += 1
    
    return [c / len(loaders[0].dataset) * 100 for c in corrects]


def eval_model(loaders, model, log_filename, datasets):
    model = model.eval()
    iterator = tqdm(enumerate(zip(*loaders)), total=len(loaders[0]))
    corrects = [0] * len(loaders)
    clear_files(log_filename, datasets)
    outs = []
    size_a = None
    for chunk_index, data in iterator:
        outputs = []
        succs = []
        paths = []
        for i, ((c_paths, inp), target) in enumerate(data):
            output, _ = model(inp.cuda())
            outputs.append(output)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.cpu().detach()[:, 0]
            for index, (tr, p) in enumerate(zip(target, pred)):
                if (p == tr).item():
                    corrects[i] += 1
                if len(succs) <= index:
                    succs.append([])
                    paths.append([])
                succs[index].append((p == tr).item())
                paths[index].append(c_paths[index])
        if size_a is None:
            size_a = len(inp)
        for s, p in zip(succs, paths):
            c = chunk_index * size_a + index
            outs.append((int(c / 450), c % 450, c, s, p))
    
    for i, c in enumerate(corrects):
        print(f"Correct {i} - {loaders[i].dataset.root}: {c / len(loaders[0].dataset) * 100}")
    
    log_to_file(log_filename, out_to_lines(outs))


def create_loaders(datasets_path, workers=4):
    datasets = []
    for directory in os.listdir(datasets_path):
        if os.path.isdir(f'{datasets_path}/{directory}'):
            for subdir in os.listdir(f'{datasets_path}/{directory}'):
                if os.path.isdir(f'{datasets_path}/{directory}/{subdir}') and subdir == 'val':
                    datasets.append(f'{datasets_path}/{directory}')
                    break
    val_loaders = []
    for dataset_path in datasets:
        dataset = ImageNet9.ImageNet9(dataset_path)
        _, val_loader = dataset.make_loaders(batch_size=4, workers=workers, add_path=True)
        val_loaders.append(val_loader)
    return val_loaders, datasets


if __name__ == "__main__":
    val_loaders, datasets = create_loaders(args.parent_dataset_path, workers=args.workers)
    eval_model(val_loaders, model, args.log_filename, datasets)
