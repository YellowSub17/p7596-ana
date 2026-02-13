

import h5py
import numpy as np
import cnst
import argparse
import extra_data
import extra_geom
from extra_data.components import AGIPD1M
import time


import os
from mpi4py import MPI






if __name__ == '__main__':

    t0  = time.perf_counter()

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank ==0:
        print('calc_train_inten.py')
        print(f'Running MPI with {mpi_size} rank(s).')

    parser = argparse.ArgumentParser("Calculate maxium intensity of trains")

    parser.add_argument("run", type=int, help='Run number.')
    parser.add_argument("--h5fname", default=None, help='Name of the output h5 file.')
    parser.add_argument("--h5mask", default=None, help='Name of the mask h5 file.')
    parser.add_argument("--h5trains", default=None, help='Name of the h5 file that has the train_ids')

    args = parser.parse_args()

    run = extra_data.open_run(proposal=cnst.PROPOSAL_NUM, run=args.run)
    sel = run.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data')

    if args.h5fname is None:
        args.h5fname =f'{cnst.H5OUT_DIR}/r{args.run:04}_iten.h5'

    if args.h5trains is None:
        args.h5trains = f'{cnst.H5OUT_DIR}/r{args.run:04}_mean.h5'
    with h5py.File(f'{args.h5trains}', 'r') as f:
        run_train_ids = f['/train_ids'][...]


    if args.h5mask is None:
        args.h5mask =f'{cnst.MASK_PATH}'
    with h5py.File(f'{args.h5mask}', 'r') as f:
        mask = f['/mask'][...]




    run = extra_data.open_run(proposal=cnst.PROPOSAL_NUM, run=args.run)
    sel = run.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data')


    worker_train_ids = np.array_split(run_train_ids, mpi_size)[mpi_rank]
    worker_run = run.select_trains(extra_data.by_id[worker_train_ids])
    worker_sel = worker_run.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data')

    worker_train_inten = np.zeros( (worker_train_ids.size) )


    worker_skip_count = 0
    worker_skip_ids = []

    if mpi_rank==0:
        print(f'Worker 0  handling {worker_train_ids.size} trains.')

    t_loop0 = time.perf_counter()
    for i_train, (train_id, train_data) in enumerate(worker_sel.trains()):

        stack = extra_data.stack_detector_data(train_data, 'image.data')[:, 0,...] #pulses, ??, modules, fast scan, slow scan 

        train_mean_im = stack.mean(axis=0)
        train_mean_im[mask>0] = 0

        worker_train_inten[i_train] = np.max(train_mean_im)


        if mpi_rank==0 and i_train%10==0:
            t_loop1 = time.perf_counter() - t_loop0
            print(f'Loop {i_train} took {round(t_loop1)} seconds.')
            t_loop0 = time.perf_counter()


    all_train_inten = mpi_comm.gather(worker_train_inten, root=0)


    if mpi_rank==0:

        run_train_inten = np.concatenate(all_train_inten)


        t1 = time.perf_counter() - t0
        print(f'Total calculation time: {round(t1/60, 2)} minutes')

        with h5py.File(args.h5fname, 'w') as h5out:

            h5out['/train_ids'] = run_train_ids
            h5out['/train_inten'] = run_train_inten
            h5out['/run'] = args.run
















