import fiftyone as fo
from argparse import ArgumentParser
import os
import sys
import json
import random
from typing import List

if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--output', type=str, required=False)
    parser.add_argument('--train_split', type=float, default=0.7)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # verify the provided path is valid
    if not os.path.exists(args.path):
        raise ValueError(f"Provided path {args.path} does not exist")
    
    # set the random seed
    random.seed(int(args.seed))

    # create a list of samples
    samples: List[fo.Sample] = []

    # read the metadata
    try:
        f = open(os.path.join(args.path, 'meta.json'))
    except Exception as e:
        print(f"Error reading metadata file: {e}")
        sys.exit(-1)
    metadata = json.load(f)
    f.close()

    # create dictionaries of classes
    classes_dict_id = {}
    classes_dict_name = {}
    for cls in metadata['classes']:
        classes_dict_id[cls['id']] = cls['title']
        classes_dict_name[cls['title']] = cls['id']

    # initialize a list of sample paths
    filepaths: List[str] = []

    # initialize a sample of detection paths
    detections_paths: List[str] = []

    # count of samples
    sample_count: int = 0

    print("Searching the dataset...", end=' ')
    # iterate the teams
    teams = os.listdir(args.path)
    for team in teams:

        team_path = os.path.join(args.path, team)
        team_img_path = os.path.join(team_path, 'img')
        team_ann_path = os.path.join(team_path, 'ann')

        # verify if is a subdirectory
        if os.path.isdir(team_img_path):

            # list all the images
            for image in os.listdir(team_img_path):

                # verify if is a file
                img_path = os.path.join(team_img_path, image)
                if os.path.isfile(img_path):

                    # append the filepath to the list
                    filepaths.append(img_path)

                    # append the detections path to the list
                    detections_paths.append(os.path.join(team_ann_path, image + '.json'))

                    # increment the sample count
                    sample_count += 1
                
                else:
                    print(f"Skipping non-file {img_path}")
        else:
            print(f"Skipping non-directory {team_img_path}")
    print("Done.")

    # threshold per split
    train_threshold = int(args.train_split * sample_count)
    val_threshold = int((args.train_split + args.val_split) * sample_count)
    print(f"Train samples: {train_threshold}")
    print(f"Validation samples: {val_threshold - train_threshold}")
    print(f"Test samples: {sample_count - val_threshold}")

    # zip the images and the detections for shuffling
    print("Shuffling the dataset...", end=' ')
    zipped = list(zip(filepaths, detections_paths))
    random.shuffle(zipped)
    print("Done.")

    # iterate the images and detections paths
    print("Processing the samples...")
    cur_sample: int = 0
    for img_path, detections_path in zipped:

        print(f"Sample {cur_sample+1}/{sample_count}", end='\r')

        # create the sample
        sample = fo.Sample(filepath=img_path)
        sample.compute_metadata() # compute the metadata

        # parse the detections
        try:
            f = open(detections_path)
        except Exception as e:
            print(f"Error reading detections file: {e}")
        detections_json = json.load(f)
        f.close()

        # get the sample tags
        sample_tags = []
        for tag in detections_json['tags']:
            if tag['name'] in {'train', 'val', 'test'}:
                continue
            sample_tags.append(tag['name'])

        # check the split
        if cur_sample < train_threshold:
            sample_tags.append('train')
        elif cur_sample < val_threshold:
            sample_tags.append('val')
        else:
            sample_tags.append('test')

        # iterate the detections, creating a list
        detections_l: List[fo.Detection] = []
        for detection in detections_json['objects']:

            # get the list of tags for the detection
            det_tags = []
            for tag in detection['tags']:
                det_tags.append(tag['name'])

            # create the detection instance
            detection_obj = fo.Detection(
                label=detection['classTitle'],
                bounding_box=[
                    int(detection['points']['exterior'][0][0]) / sample.metadata.width,
                    int(detection['points']['exterior'][0][1]) / sample.metadata.height,
                    (int(detection['points']['exterior'][1][0]) - int(detection['points']['exterior'][0][0])) / sample.metadata.width,
                    (int(detection['points']['exterior'][1][1]) - int(detection['points']['exterior'][0][1])) / sample.metadata.height
                ],
                tags=det_tags
            )
            # add the detection to the list
            detections_l.append(detection_obj)


        # create a detections object
        detections = fo.Detections(detections=detections_l)

        # add the detections to the sample
        sample['ground_truth'] = detections

        # add the tags to the sample
        sample.tags = sample_tags

        # add the sample to the list
        samples.append(sample)

        # increment the sample count
        cur_sample += 1
    print("Done.")


    # delete the dataset if it exists
    if fo.dataset_exists("fsoco"):
        print("Dataset fsoco already exists. Deleting it.")
        fo.delete_dataset("fsoco")

    # create a dataset
    print("Creating dataset fsoco")
    dataset = fo.Dataset("fsoco")
    dataset.add_samples(samples)
    dataset.persistent = True

    # save the dataset, if the output directory was provided
    if args.output:
        print("Exporting dataset in COCO format")
        # create the output directory if it does not exist
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        dataset.export(args.output, dataset_type=fo.types.COCODetectionDataset, label_field='ground_truth', overwrite=True)

    print("Done")
