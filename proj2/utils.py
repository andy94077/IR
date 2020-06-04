import numpy as np
import pandas as pd

def load_data(path):
    """Load csv file and return the contents and the number of users and items.

    Args:
        path: The path of the csv file.

    Returns:
        positiveX, num_users, num_items,
        where `positiveX` is a list of lists containing positive samples of each users.
    """

    df = pd.read_csv(path)
    positiveX = [[int(i) for i in ll.split()] for ll in df.iloc[:, 1]]
    max_item_idx = 0
    for ll in positiveX:
        max_item_idx = max(ll+[max_item_idx])

    return positiveX, len(positiveX), max_item_idx + 1

def generate_csv(pred, output_file):
    df = pd.DataFrame(list(zip(range(len(pred)), map(' '.join, pred))), columns=['UserId', 'ItemId'])
    pd.to_csv(output_file, index=False)
