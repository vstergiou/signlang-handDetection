import os
import utils.constants as cs
import pandas as pd 
#from models import feature_extractor
from tqdm import tqdm
import numpy as np
import torch
from torch import optim, nn
from torchvision import models, transforms


def create_directory(directory):
    """
    creates a directory
    :param directory: string
                      directory path to be created
    """
    os.makedirs(directory)


def check_existence(directory):
    """Test whether a path exists.  Returns False for broken symbolic links"""
    return os.path.exists(directory)


def load_data():

    hand_df = pd.read_csv(cs.HANDANNOTATIONS_CSV)
    chicago_df = pd.read_csv(cs.CHICAGOFS_CSV)
    merged_df = pd.merge(chicago_df, hand_df, on='filename')


    return chicago_df, hand_df, merged_df
