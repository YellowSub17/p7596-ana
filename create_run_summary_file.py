

import h5py
from constants import *
import argparse






if __name__ == '__main__':


    parser = argparse.ArgumentParser("Create a run summary file.")


    parser.add_argument("run", type=int, help='Run number to summarize.')
    parser.add_argument("--h5dir", default=None, help='Directory to save h5 file.')
    parser.add_argument("--h5name", default=None, help='Name of the h5 file.')
    parser.add_argument("--n-total-trains", type=int, default=-1, help='Number of trains to summarize.')


    args = parser.parse_args()

    if args.h5dir is None:
        args.h5dir = f'{H5OUT_DIR}'

    if args.h5name is None:
        args.h5name =f'r{args.run}_summary.h5'



    with h5py.File(f'{args.h5dir}/r{args.h5name}', 'w') as f:

        for key, value in vars(args).items():
            f[f'/args/{key}'] = value






