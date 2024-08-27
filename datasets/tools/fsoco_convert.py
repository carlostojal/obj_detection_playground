import fiftyone as fo
import glob
from argparse import ArgumentParser
import os
import sys
import json
from typing import List

if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--output', type=str, required=False)
    args = parser.parse_args()

    # verify the provided path is valid
    if not os.path.exists(args.path):
        raise ValueError(f"Provided path {args.path} does not exist")
    
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

                    print(f"Processing {img_path}")

                    # create the sample
                    sample = fo.Sample(filepath=img_path)
                    sample.compute_metadata() # compute the metadata

                    # parse the detections
                    detections_fname = os.path.join(team_ann_path, image + '.json')
                    try:
                        f = open(os.path.join(team_ann_path, detections_fname))
                    except Exception as e:
                        print(f"Error reading detections file: {e}")
                    detections_json = json.load(f)
                    f.close()

                    # get the sample tags
                    tags = []
                    for tag in detections_json['tags']:
                        tags.append(tag['name'])
                    sample.tags = tags

                    # iterate the detections, creating a list
                    detections_l: List[fo.Detection] = []
                    for detection in detections_json['objects']:

                        # get the list of tags for the detection
                        tags = []
                        for tag in detection['tags']:
                            tags.append(tag['name'])

                        # create the detection instance
                        detection_obj = fo.Detection(
                            label=detection['classTitle'],
                            bounding_box=[
                                int(detection['points']['exterior'][0][0]) / sample.metadata.width,
                                int(detection['points']['exterior'][0][1]) / sample.metadata.height,
                                (int(detection['points']['exterior'][1][0]) - int(detection['points']['exterior'][0][0])) / sample.metadata.width,
                                (int(detection['points']['exterior'][1][1]) - int(detection['points']['exterior'][0][1])) / sample.metadata.height
                            ],
                            tags=tags
                        )
                        # add the detection to the list
                        detections_l.append(detection_obj)


                    # create a detections object
                    detections = fo.Detections(detections=detections_l)

                    # add the detections to the sample
                    sample['ground_truth'] = detections

                    # add the tags to the sample
                    sample.tags = tags

                    # add the sample to the list
                    samples.append(sample)
                
                else:
                    print(f"Skipping non-file {img_path}")
        else:
            print(f"Skipping non-directory {team_img_path}")


    # delete the dataset if it exists
    if fo.dataset_exists("fsoco"):
        print("Dataset fsoco already exists. Deleting it.")
        fo.delete_dataset("fsoco")

    # create a dataset
    print("Creating dataset fsoco")
    dataset = fo.Dataset("fsoco")
    dataset.add_samples(samples)

    # save the dataset, if the output directory was provided
    if args.output:
        print("Exporting dataset in COCO format")
        # create the output directory if it does not exist
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        dataset.export(args.output, dataset_type=fo.types.COCODetectionDataset, label_field='ground_truth', overwrite=True)

    print("Done")
