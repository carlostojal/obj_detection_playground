#! /usr/bin/python3
import torch
from argparse import ArgumentParser
import os
import sys
import yaml
sys.path.append(".")
from yolo.models.yolov1 import YOLOv1
from datasets.FSOCO_FiftyOne import FSOCO_FiftyOne
from yolo.utils import YOLOv1Loss, fsoco_to_yolo_bboxes
from torch.optim.lr_scheduler import StepLR, ExponentialLR

if __name__ == "__main__":

    print("** YOLOv1 Training Script **\n")

    # parse command line arguments
    print("Parsing command line arguments...", end=" ")
    parser = ArgumentParser()
    parser.add_argument("--conf", "-c", help="Path to YOLO configuration file", type=str, required=True)
    parser.add_argument("--num_epochs", "-e", help="Number of epochs to train", type=int, default=100)
    parser.add_argument("--img_width", "-iw", help="Width of the input image", type=int, default=640)
    parser.add_argument("--img_height", "-ih", help="Height of the input image", type=int, default=480)
    args = parser.parse_args()
    print("Done.")

    # parse the YOLO configuration file
    print("Parsing the YOLO configuration file...", end=" ")
    if not os.path.isfile(args.conf):
        print("Provided conf. file path is not a file")
        sys.exit(-1)
    if not os.path.exists(args.conf):
        print("Provided conf. file path does not exist")
        sys.exit(-2)
    # open the file for reading
    try:
        h = open(args.conf, "r")
    except Exception as e:
        print("Error opening configuration file: ", e)
        sys.exit(-3)
    # parse the YAML configuration
    conf = yaml.safe_load(h)
    try:
        h.close()
    except Exception as e:
        print("Error closing configuration file: ", e)
        sys.exit(-4)
    print("Done.")

    print(conf)

    # build the model
    print("Building the model...", end=" ")
    model = YOLOv1(conf)
    print("Done.")
    print(model)

    # create the datasets
    train_set = FSOCO_FiftyOne("train")
    val_set = FSOCO_FiftyOne("val")
    test_set = FSOCO_FiftyOne("test")

    # create the dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=int(conf['batch_size']), shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=int(conf['batch_size']), shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=int(conf['batch_size']), shuffle=False)

    # create the criterion
    criterion = YOLOv1Loss(conf)

    # create the optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=float(conf['learning_rate']), momentum=float(conf['momentum']), weight_decay=float(conf['weight_decay']))

    # create the scheduler
    scheduler = ExponentialLR(optimizer, gamma=float(conf['lr_decay']))

    # verify the available devices
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU...")
        device = torch.device("cuda")

    # move the model to the device
    model = model.to(device)

    # set the model to training mode
    model.train()

    # training loop
    for epoch in range(int(args.num_epochs)):

        # iterate the training set
        for i, (imgs, bboxes) in enumerate(train_loader):

            # move the data to the device
            imgs = imgs.to(device)
            bboxes = bboxes.to(device)

            # convert the bounding boxes to the YOLO format
            bboxes = fsoco_to_yolo_bboxes(bboxes, (int(args.img_height), int(args.img_width)), 
                                          grid_size=int(conf['grid_size']), n_predictors=int(conf['n_predictors']),
                                          n_classes=int(conf['n_classes']))

            # zero the gradients
            optimizer.zero_grad()

            # forward pass
            preds = model(imgs)

            # calculate the loss
            loss = criterion(preds, bboxes)

            # backward pass
            loss.backward()

            # update the weights
            optimizer.step()

            # print the loss
            print(f"Epoch: {epoch+1}, Batch: {i}, Loss: {loss.item()}", end="\r")

    sys.exit(0)
