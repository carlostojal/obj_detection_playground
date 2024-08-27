import fiftyone as fo
from argparse import ArgumentParser

if __name__ == '__main__':

    # parse command line arguments
    parser = ArgumentParser()
    parser.add_argument('--path', type=str, help="Path to the dataset in COCO format.", required=False)
    args = parser.parse_args()

    # verify if the dataset is already loaded
    if fo.dataset_exists("fsoco"):
        # load the dataset
        dataset = fo.load_dataset("fsoco")
    else:
        # load the dataset from the provided path
        dataset = fo.Dataset.from_dir(args.path, fo.types.COCODetectionDataset)
        # save the dataset
        dataset.persistent = True
        dataset.name = "fsoco"

    fo.close_app()

    # create the session
    session = fo.launch_app(dataset)
    session.wait() # wait for the session to finish to close the app
