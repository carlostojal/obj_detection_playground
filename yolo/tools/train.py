#! /usr/bin/python3
import torch
from argparse import ArgumentParser
import os
import sys
import yaml
sys.path.append(".")
from models.yolov1 import YOLOv1
import time

if __name__ == "__main__":

    print("** YOLOv1 Training Script **\n")

    # parse command line arguments
    print("Parsing command line arguments...", end=" ")
    parser = ArgumentParser()
    parser.add_argument("--conf", "-c", help="Path to YOLO configuration file", type=str, required=True)
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

    # verify the available devices
    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU...")
        device = torch.device("cuda")

    # move the model to the device
    model = model.to(device)
    model.eval()

    # make a dummy forward pass
    print("Making a dummy forward pass...", end=" ")
    warmup_passes = 50
    for i in range(warmup_passes):
        x = torch.randn(1, 3, 448, 448).to(device)
        start_t = time.time()
        bboxes = model(x)
        end_t = time.time()

        if i == warmup_passes - 1:
            print(f"Done in {end_t-start_t}s.")

    print(bboxes.shape)

    sys.exit(0)
