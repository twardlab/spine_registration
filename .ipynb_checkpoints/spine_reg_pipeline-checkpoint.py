import argparse


def main():
    """
    Command Line Arguments:
    =======================

    Raises:
    =======
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('fname_I', type=str, help = 'The path to the interpolated atlas .npz file')
    parser.add_argument('fname_L', type=str, help = 'The path to the interpolated atlas labels .npz file')
    parser.add_argument('fname_J', type=str, help = 'The path to the spine reflection .npz file')
    parser.add_argument('fname_pointJ', type=str, help = 'The input .swc file used for more precise registration')
    