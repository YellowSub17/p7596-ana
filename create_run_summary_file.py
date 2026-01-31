

import h5py
from constants import *
import argparse
import extra_data
import numpy as np






if __name__ == '__main__':


    parser = argparse.ArgumentParser("Create a run summary file.")


    parser.add_argument("run", type=int, help='Run number to summarize.')
    parser.add_argument("--h5dir", default=None, help='Directory to save h5 file.')
    parser.add_argument("--h5name", default=None, help='Name of the h5 file.')
    parser.add_argument("--n-trains", type=int, default=-1, help='Number of trains to summarize.')


    args = parser.parse_args()

    if args.h5dir is None:
        args.h5dir = f'{H5OUT_DIR}'

    if args.h5name is None:
        args.h5name =f'r{args.run}_summary.h5'


    run = extra_data.open_run(proposal=PROPOSAL_NUM, run=args.run)

    sel = run.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data')
    n_total_trains = len(sel.train_ids)
    if args.n_trains ==-1:
        args.n_trains = n_total_trains

    tid, train_data = sel.train_from_index(0)

    stack = extra_data.stack_detector_data(train_data, 'image.data') #pulses, ??, modules, fast scan, slow scan 
    n_pulses = stack.shape[0]



    with h5py.File(f'{args.h5dir}/{args.h5name}', 'w') as f:

        for key, value in vars(args).items():
            f[f'/args/{key}'] = value

        f['/run/n_pulses'] = n_pulses
        f['/run/n_total_trains'] = n_total_trains






