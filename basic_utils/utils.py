from argparse import Namespace
import yaml

def load_config(filename):
    config = Namespace()
    with open(filename, "r") as stream:
        try:
            config = Namespace(**yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)

    return config