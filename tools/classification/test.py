import logging
import os
import argparse
import warnings

warnings.filterwarnings('ignore')

import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(BASE_DIR)

import time
from data import *
from PIL import Image
import torch.nn.parallel
from torchvision import transforms
from utils.get_logger import get_logger
from models.basenets.lenet5 import lenet5
from models.basenets.alexnet import alexnet
from models.basenets.vgg import vgg11, vgg13, vgg16, vgg19
from models.basenets.googlenet import googlenet, GoogLeNet
from models.basenets.mobilenet_v2 import mobilenet_v2, MobileNet_v2
from models.basenets.resnet import resnet18, resnet34, resnet50, resnet101, resnet152


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Testing')
    parser.add_mutually_exclusive_group()
    parser.add_argument('--dataset',
                        type=str,
                        default='ImageNet',
                        choices=['ImageNet', 'CIFAR'],
                        help='ImageNet,  CIFAR')
    parser.add_argument('--images_root',
                        type=str,
                        default=config.images_cls_root,
                        help='Dataset root directory path')
    parser.add_argument('--basenet',
                        type=str,
                        default='googlenet',
                        choices=['resnet', 'vgg', 'lenet', 'alexnet', 'googlenet', 'mobilenet'],
                        help='Pretrained base model')
    parser.add_argument('--depth',
                        type=int,
                        default=0,
                        help='BaseNet depth, including: LeNet of 5, AlexNet of 0, VGG of 11, 13, 16, 19, ResNet of 18, 34, 50, 101, 152, GoogLeNet of 0, MobileNet of 2, 3')
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
    parser.add_argument('--num_classes',
                        type=int,
                        default=1000,
                        help='the number classes, like ImageNet:1000, cifar:10')
    parser.add_argument('--image_size',
                        type=int,
                        default=224,
                        help='image size, like ImageNet:224, cifar:32')
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
    # vgg16, alexnet and lenet5 need to resize image_size, because of fc.
    if args.basenet == 'vgg' or args.basenet == 'alexnet' or args.basenet == 'googlenet':
        args.image_size = 224
    elif args.basenet == 'lenet':
        args.image_size = 32

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
    if args.basenet == 'lenet':
        if args.depth == 5:
            model = lenet5(num_classes=args.num_classes)
        else:
            raise ValueError('Unsupported LeNet depth!')

    elif args.basenet == 'alexnet':
        if args.depth == 0:
            model = alexnet(num_classes=args.num_classes)
        else:
            raise ValueError('Unsupported AlexNet depth!')

    elif args.basenet == 'googlenet':
        if args.depth == 0:
            model = googlenet(num_classes=args.num_classes,
                              aux_logits=False)
        else:
            raise ValueError('Unsupported GoogLeNet depth!')

    elif args.basenet == 'vgg':
        if args.depth == 11:
            model = vgg11(num_classes=args.num_classes)
        elif args.depth == 13:
            model = vgg13(num_classes=args.num_classes)
        elif args.depth == 16:
            model = vgg16(num_classes=args.num_classes)
        elif args.depth == 19:
            model = vgg19(num_classes=args.num_classes)
        else:
            raise ValueError('Unsupported VGG depth!')

    elif args.basenet == 'resnet':
        if args.depth == 18:
            model = resnet18(num_classes=args.num_classes)
        elif args.depth == 34:
            model = resnet34(num_classes=args.num_classes)
        elif args.depth == 50:
            model = resnet50(num_classes=args.num_classes)  # False means the models is not trained
        elif args.depth == 101:
            model = resnet101(num_classes=args.num_classes)
        elif args.depth == 152:
            model = resnet152(num_classes=args.num_classes)
        else:
            raise ValueError('Unsupported ResNet depth!')
    elif args.basenet == 'mobilenet':
        if args.depth == 2:
            model = mobilenet_v2(num_classes=args.num_classes)
        else:
            raise ValueError('Unsupported MobileNet depth!')
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
            model_evaluate_load = torch.load(model_evaluate_load)
            if args.basenet == 'googlenet':
                model_evaluate_load = {k: v for k, v in model_evaluate_load.items() if "aux" not in k}
            model.load_state_dict(model_evaluate_load)
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
