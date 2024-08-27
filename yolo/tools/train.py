#! /usr/bin/python3
from argparse import ArgumentParser
import os
import sys
import yaml

if __name__ == "__main__":
    print("YOLO TRAINING SCRIPT")
    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("--conf", "-c", help="Path to YOLO configuration file", type=str, required=True)
    args = parser.parse_args()

    # parse the YOLO configuration file
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
        print("Error opening file: ", e)
        sys.exit(-3)
    # parse the YAML configuration
    conf = yaml.safe_load(h)

    print(f"YOLO configuration: {conf}")

    raise NotImplementedError("YOLO training script not implemented")

    sys.exit(0)
