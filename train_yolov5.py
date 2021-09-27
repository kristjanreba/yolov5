# YOLOv5 classifier training
# Usage: python classifier.py --model yolov5s --data mnist --epochs 10 --img 128

import argparse
import logging
import math
import os
from copy import deepcopy
from pathlib import Path

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchvision
import torchvision.transforms as T
from torch.cuda import amp
from torchvision.transforms.transforms import Normalize, ToTensor
from tqdm import tqdm

from sklearn.metrics import confusion_matrix

from models.common import Classify
from utils.general import set_logging, check_file, increment_path, check_git_status, check_requirements
from utils.torch_utils import model_info, select_device, is_parallel

import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class WoodTextureDataset(Dataset):

    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(annotation_file)
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_id = self.annotations.iloc[index, 0]
        img = cv2.imread(os.path.join(self.root_dir, img_id))

        img_shape = (640, 120)
        img = cv2.resize(img, img_shape)
        #img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #img = img_gray

        #img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
        #sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)  # Sobel Edge Detection on the X axis
        #sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=3)  # Sobel Edge Detection on the Y axis
        #sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=3)  # Combined X and Y Sobel Edge Detection
        #img = sobely
        #cv2.imshow('img', img)
        #cv2.waitKey(0)
        #exit()

        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform is not None:
            img = self.transform(img)

        return (img.float(), y_label)


# Settings
logger = logging.getLogger(__name__)
set_logging()


# Show images
def imshow(img):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.imshow(np.transpose((img / 2 + 0.5).numpy(), (1, 2, 0)))  # unnormalize
    plt.savefig('images.jpg')


def train():
    save_dir, data, bs, epochs, nw, imgsz = Path(opt.save_dir), opt.data, opt.batch_size, opt.epochs, \
                                            min(os.cpu_count(), opt.workers), opt.img_size

    # Directories
    wdir = save_dir / 'weights'
    wdir.mkdir(parents=True, exist_ok=True)  # make dir
    last, best = wdir / 'last.pt', wdir / 'best.pt'

    # Load data
    path_images_train = data
    # '../data/texture_data/train/images/'
    path_images_test = data.replace('train', 'test')
    # '../data/texture_data/test/images/'

    names = ('Radial', 'Semiradial', 'Tangential')
    nc = len(names)

    # Transforms
    trainform = T.Compose([T.ToTensor(),
                           T.RandomGrayscale(p=0.01),
                           T.RandomHorizontalFlip(p=0.5),
                           T.RandomVerticalFlip(p=0.5),
                           T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])  # PILImage from [0, 1] to [-1, 1]

    testform = T.Compose([T.ToTensor(),
                          T.Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25))])

    # Dataloaders
    if opt.train:
        trainset = WoodTextureDataset(path_images_train, "../train_csv.csv", transform=trainform)
        trainloader = DataLoader(dataset=trainset, shuffle=True, batch_size=bs, num_workers=nw)
        print(f'Training {opt.model} on {data} dataset with {nc} classes...')

    testset = WoodTextureDataset(path_images_test, "../test_csv.csv", transform=testform)
    testloader = DataLoader(dataset=testset, shuffle=False, batch_size=bs, num_workers=1)

    # Show images
    # images, labels = iter(trainloader).next()
    # imshow(torchvision.utils.make_grid(images[:16]))
    # print(' '.join('%5s' % names[labels[j]] for j in range(16)))

    # Model
    if opt.train:
        if opt.model.startswith('yolov5'):
            model = torch.hub.load('ultralytics/yolov5', opt.model, pretrained=True, autoshape=False)
            model.model = model.model[:8]
            m = model.model[-1]  # last layer
            ch = m.conv.in_channels if hasattr(m, 'conv') else sum([x.in_channels for x in m.m])  # ch into module
            c = Classify(ch, nc)  # Classify()
            c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
            model.model[-1] = c  # replace
            for p in model.parameters():
                p.requires_grad = True  # for training
        elif opt.model in torch.hub.list('rwightman/gen-efficientnet-pytorch'):  # i.e. efficientnet_b0
            model = torch.hub.load('rwightman/gen-efficientnet-pytorch', opt.model, pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features, nc)
        else:  # try torchvision
            model = torchvision.models.__dict__[opt.model](pretrained=True)
            model.fc = nn.Linear(model.fc.weight.shape[1], nc)

    else:  # load pretrained model
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False, autoshape=False)
        model.model = model.model[:8]
        m = model.model[-1]  # last layer
        ch = m.conv.in_channels if hasattr(m, 'conv') else sum([x.in_channels for x in m.m])  # ch into module
        c = Classify(ch, nc)  # Classify()
        c.i, c.f, c.type = m.i, m.f, 'models.common.Classify'  # index, from, type
        model.model[-1] = c  # replace
        
        model.load_state_dict(torch.load(opt.model, map_location=torch.device('cuda'))['model'])

    #model = model
    #print(model)  # debug
    model_info(model)

    if opt.train:
        # Optimizer
        lr0 = 0.0001 * bs  # intial lr
        lrf = 0.01  # final lr (fraction of lr0)
        if opt.adam:
            optimizer = optim.Adam(model.parameters(), lr=lr0 / 10)
        else:
            optimizer = optim.SGD(model.parameters(), lr=lr0, momentum=0.9, nesterov=True)

        # Scheduler
        lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - lrf) + lrf  # cosine
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

        # Train
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()  # loss function
        best_fitness = 0.0
        # scaler = amp.GradScaler(enabled=cuda)
        print(f'Image sizes {imgsz} train, {imgsz} test\n'
            f'Using {nw} dataloader workers\n'
            f'Logging results to {save_dir}\n'
            f'Starting training for {epochs} epochs...\n\n'
            f"{'epoch':10s}{'gpu_mem':10s}{'train_loss':12s}{'val_loss':12s}{'accuracy':12s}")
        for epoch in range(epochs):  # loop over the dataset multiple times
            mloss = 0.  # mean loss
            model.train()
            pbar = tqdm(enumerate(trainloader), total=len(trainloader))  # progress bar
            for i, (images, labels) in pbar:
                images, labels = resize(images.to(device)), labels.to(device)

                # Forward
                with amp.autocast(enabled=cuda):
                    loss = criterion(model(images), labels)

                # Backward
                loss.backward()  # scaler.scale(loss).backward()

                # Optimize
                optimizer.step()  # scaler.step(optimizer); scaler.update()
                optimizer.zero_grad()

                # Print
                mloss += loss.item()
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                pbar.desc = f"{'%s/%s' % (epoch + 1, epochs):10s}{mem:10s}{mloss / (i + 1):<12.3g}"

                # Test
                if i == len(pbar) - 1:
                    fitness = val(model, testloader, names)  # test

            # Scheduler
            scheduler.step()

            # Best fitness
            if fitness > best_fitness:
                best_fitness = fitness

            # Save model
            final_epoch = epoch + 1 == epochs
            if (not opt.nosave) or final_epoch:
                ckpt = {'epoch': epoch,
                        'best_fitness': best_fitness,
                        'model': model.state_dict(),
                        'optimizer': None}

                # Save last, best and delete
                torch.save(ckpt, last)
                if best_fitness == fitness:
                    torch.save(ckpt, best)
                del ckpt

        # Train complete
        if final_epoch:
            fitness = val(model, testloader, names)  # validation
            print(f'Training complete. Results saved to {save_dir}.')
    else:
        print('Evaluating model')
        fitness = val(model, testloader, names)


def val(model, dataloader, names, verbose=True):
    model.eval()
    pred, targets = [], []
    with torch.no_grad():
        i = 0
        for images, labels in dataloader:
            images, labels = resize(images.to(device)), labels.to(device)
            y = model.to(device)(images)
            pred.append(torch.max(y, 1)[1])
            targets.append(labels)
            if i % 10 == 0:
                print(i)
            i += 1

    pred, targets = torch.cat(pred), torch.cat(targets)
    correct = (targets == pred).float()

    accuracy = correct.mean().item()
    if verbose:  # all classes
        print("class  number   accuracy")
        print(f"all:   {correct.shape[0]}  {accuracy}")
        for i, c in enumerate(names):
            t = correct[targets == i]
            print(f"{c}  {t.shape[0]}  {t.mean().item()}")
        
        targets = targets.cpu()
        pred = pred.cpu()
        print(confusion_matrix(targets, pred))
        print(confusion_matrix(targets, pred, normalize='true'))

    # Show predictions
    #images, labels = iter(testloader).next()
    #predicted = torch.max(model(images), 1)[1]
    #imshow(torchvision.utils.make_grid(images))
    #print('GroundTruth: ', ' '.join('%5s' % names[labels[j]] for j in range(4)))
    #print('Predicted: ', ' '.join('%5s' % names[predicted[j]] for j in range(4)))
    return accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov5s', help='initial weights path')
    parser.add_argument('--data', type=str, default='mnist', help='cifar10, cifar100 or mnist')
    parser.add_argument('--hyp', type=str, default='data/hyp.scratch.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128, help='total batch size for all GPUs')
    parser.add_argument('--img-size', type=int, default=64, help='train, test image sizes (pixels)')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--adam', action='store_true', help='use torch.optim.Adam() optimizer')
    parser.add_argument('--evolve', action='store_true', help='evolve hyperparameters')
    parser.add_argument('--cache-images', action='store_true', help='cache images for faster training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=4, help='maximum number of dataloader workers')
    parser.add_argument('--project', default='runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--test', action='store_true', help='Test model')
    opt = parser.parse_args()

    # Checks
    check_git_status()
    check_requirements()

    # Parameters
    print(torch.cuda.device_count())
    print(torch.version.cuda)
    print(torch.version)
    print(torch.cuda.is_available())
    device = select_device(opt.device, batch_size=opt.batch_size)
    cuda = device.type != 'cpu'
    opt.hyp = check_file(opt.hyp)  # check files
    opt.save_dir = increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok | opt.evolve)  # increment run
    resize = torch.nn.Upsample(size=(opt.img_size, opt.img_size), mode='bilinear', align_corners=False)  # image resize

    # Train and test
    train()
