

import h5py
import cnst
import argparse
import extra_data
import numpy as np






if __name__ == '__main__':


    parser = argparse.ArgumentParser("Create a run summary file.")

    parser.add_argument("run", type=int, help='Run number to summarize.')
    parser.add_argument("--h5out", default=None, help='Directory to save h5 file.')
    parser.add_argument("--n-trains", type=int, default=-1, help='Number of trains to summarize.')

    args = parser.parse_args()

    if args.h5out is None:
        args.h5out =f'{cnst.H5OUT_DIR}/r{args.run:04}_ana.h5'

    run = extra_data.open_run(proposal=PROPOSAL_NUM, run=args.run)

    sel = run.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data')

    n_total_trains = len(sel.train_ids)

    if args.n_trains ==-1:
        args.n_trains = n_total_trains

    train_ids = sel.train_ids[:args.n_trains]

    tid, train_data = sel.train_from_index(0)

    stack = extra_data.stack_detector_data(train_data, 'image.data') #pulses, ??, modules, fast scan, slow scan 

    n_pulses = stack.shape[0]


    with h5py.File(f'{args.h5out}', 'w') as f:

        f[f'/train_ids'] = train_ids
        f[f'/n_trains'] = args.n_trains
        f[f'/n_pulses'] = n_pulses
        f[f'/run'] = args.run






