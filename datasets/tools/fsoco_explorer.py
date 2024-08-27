import fiftyone as fo

if __name__ == '__main__':

    # load the dataset
    dataset = fo.load_dataset("fsoco")

    # create the session
    session = fo.launch_app(dataset)
