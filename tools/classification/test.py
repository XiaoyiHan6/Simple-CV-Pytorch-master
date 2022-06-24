import logging
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

from data import *
import torchvision
from torchvision import transforms
import torch.nn.parallel
import time
from utils.get_logger import get_logger
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Detection Evaluation')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset',
                        type=str,
                        default='ImageNet',
                        choices=['ImageNet', 'cifar'],
                        help='ImageNet,  CIFAR')
    parser.add_argument('--images_root',
                        type=str,
                        default=config.images_root,
                        help='Dataset root directory path')
    parser.add_argument('--basenet',
                        type=str,
                        default='resnet',
                        help='Pretrained base model')
    parser.add_argument('--depth',
                        type=int,
                        default=50,
                        help='Backbone depth, including ResNet of 18, 34, 50, 101, 152')
    parser.add_argument('--evaluate',
                        type=str,
                        default=config.classification_evaluate,
                        help='Checkpoint state_dict file to evaluate training from')
    parser.add_argument('--save_folder',
                        type=str,
                        default=config.checkpoint_path,
                        help='Directory for saving checkpoint models')
    parser.add_argument('--log_folder',
                        type=str,
                        default=config.log,
                        help='Log Folder')
    parser.add_argument('--log_name',
                        type=str,
                        default=config.classification_test_log,
                        help='Log Name')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to train model')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='image size, like ImageNet:224, cifar:32')
    parser.add_argument('--pretrained',
                        type=str,
                        default=False,
                        help='Models was pretrained')

    return parser.parse_args()


args = parse_args()

# 1. Torch choose cuda or cpu
if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but you aren't using it" +
              "\n You can set the parameter of cuda to True.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

# 2. Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)


def get_label_file(filename):
    if not os.path.exists(filename):
        print("The dataset label.txt is empty, We need to create a new one.")
        os.mkdir(filename)
    return filename


def dataset_labels_results(filename, output):
    filename = os.path.join(BASE_DIR, 'data', filename + '_labels.txt')
    get_label_file(filename=filename)
    with open(file=filename, mode='r') as f:
        dict = f.readlines()
        output = output.cpu().numpy()
        output = output[0]
        output = dict[output]
        f.close()
    return output


def test():
    # 3. Ready image
    if args.images_root is None:
        raise ValueError("The images is None, you should load image!")

    image = Image.open(args.images_root)
    transform = transforms.Compose([
        transforms.Resize((args.image_size,
                           args.image_size)),
        transforms.ToTensor()])

    image = transform(image)

    image = image.reshape(1, 3, args.image_size, args.image_size)

    # 4. Define to train mode
    if args.basenet == 'resnet':
        if args.depth == 18:
            model = torchvision.models.resnet18(pretrained=args.pretrained)
        elif args.depth == 34:
            model = torchvision.models.resnet34(pretrained=args.pretrained)
        elif args.depth == 50:
            model = torchvision.models.resnet50(pretrained=args.pretrained)  # False means the models is not trained
        elif args.depth == 101:
            model = torchvision.models.resnet101(pretrained=args.pretrained)
        elif args.depth == 152:
            model = torchvision.models.resnet152(pretrained=args.pretrained)
        else:
            raise ValueError('Unsupported model depth!')
    else:
        raise ValueError('Unsupported model type!')

    if args.cuda:
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    # 5. Loading model
    if args.evaluate:
        other, ext = os.path.splitext(args.evaluate)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            model_evaluate_load = os.path.join(args.save_folder, args.evaluate)
            model.load_state_dict(torch.load(model_evaluate_load))
        else:
            print('Sorry only .pth and .pkl files supported.')
    elif args.evaluate is None:
        print("Sorry, you should load weights! ")

    model.eval()

    # 6. print
    logger.info(f"args - {args}")

    # 7. Test
    with torch.no_grad():
        t0 = time.time()
        # 8. Forward
        if args.cuda:
            image = image.cuda()
        output = model(image)
        output = output.argmax(1)
        t1 = time.time()
        m = (t1 - t0) // 60
        s = (t1 - t0) % 60
        folder_name = args.dataset
        output = dataset_labels_results(filename=folder_name, output=output)
        logger.info(f"output: {output}")
        print("It took a total of {}m{}s to complete the testing.".format(int(m), int(s)))
    return output


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    logger.info("Program started")
    output = test()
    logger.info("Done!")
