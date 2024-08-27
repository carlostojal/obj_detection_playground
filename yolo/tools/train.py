#! /usr/bin/python3
from argparse import ArgumentParser
import os
import sys
import yaml
sys.path.append(".")
from models.yolov1 import YOLOv1

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

    sys.exit(0)
