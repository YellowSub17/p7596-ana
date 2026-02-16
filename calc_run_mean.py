

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
        print('calc_run_mean.py')
        print(f'Running MPI with {mpi_size} rank(s).')

    parser = argparse.ArgumentParser("Calculate sum and mean of run.")

    parser.add_argument("run", type=int, help='Run number.')
    parser.add_argument("--h5fname", default=None, help='Name of the output h5 file.')
    parser.add_argument("--n-trains", type=int, default=-1, help='Number of trains to analyse.')


    parser.add_argument("--h5mask", default=None, help='Name of the mask h5 file.')


    args = parser.parse_args()

    if args.h5fname is None:
        args.h5fname =f'{cnst.H5OUT_DIR}/r{args.run:04}_mean.h5'

    if args.h5mask is None:
        args.h5mask =f'{cnst.MASK_PATH}'

    if os.path.exists(args.h5mask):
        with h5py.File(args.h5mask, 'r') as f:
            mask = f['/mask'][...]
    else:
        if mpi_rank ==0:
            print('Warning: No mask file found. Using empty mask.')
        mask = np.zeros(cnst.DET_SHAPE)




    run_proc = extra_data.open_run(proposal=cnst.PROPOSAL_NUM, run=args.run, data='proc')
    sel_proc = run_proc.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data', require_all=True)

    run_train_ids = sel_proc.train_ids[:]
    n_total_trains = len(run_train_ids)
    if args.n_trains ==-1:
        args.n_trains = n_total_trains
    run_train_ids = run_train_ids[:args.n_trains]

    if mpi_rank ==0:
        print(f'Calculating average for run {args.run} with {args.n_trains} trains.')



    worker_train_ids = np.array_split(run_train_ids, mpi_size)[mpi_rank]
    print(f'Worker {mpi_rank}: First/Last train id is {worker_train_ids[0]}, {worker_train_ids[-1]}, with {len(worker_train_ids)} trains.')
    worker_sel_proc = sel_proc.select_trains(extra_data.by_id[worker_train_ids])

    worker_sum_im = np.zeros(cnst.DET_SHAPE)
    worker_sumsq_im = np.zeros(cnst.DET_SHAPE)
    worker_train_inten = np.zeros( (worker_train_ids.size, 202) )





    t_loop0 = time.perf_counter()
    for i_train, (train_id, train_data) in enumerate(worker_sel_proc.trains()):

        stack = extra_data.stack_detector_data(train_data, 'image.data') #pulses, modules, fast scan, slow scan 

        stack[:, mask>0] = np.nan

        worker_train_inten[i_train, :stack.shape[0]] = np.nanmean(stack, axis=(1,2,3))

        train_sum_im = np.nansum(stack, axis=0)
        worker_sum_im += train_sum_im
        worker_sumsq_im += train_sum_im**2

        if mpi_rank==0 and i_train%10==0:
            t_loop1 = time.perf_counter() - t_loop0
            print(f'Loop {i_train} took {round(t_loop1)} seconds.')
            t_loop0 = time.perf_counter()

    n_pulses = stack.shape[0]


    if mpi_rank ==0:
        run_sum_im = np.zeros(cnst.DET_SHAPE)
        run_sumsq_im = np.zeros(cnst.DET_SHAPE)
        run_train_inten = np.zeros(args.n_trains)
    else:
        run_sum_im = None
        run_sumsq_im = None
        run_train_inten = None

    mpi_comm.Reduce(
            [worker_sum_im, MPI.FLOAT],
            [run_sum_im, MPI.FLOAT],
            op=MPI.SUM,
            root=0
            )

    mpi_comm.Reduce(
            [worker_sumsq_im, MPI.FLOAT],
            [run_sumsq_im, MPI.FLOAT],
            op=MPI.SUM,
            root=0
            )


    run_train_inten_gathered = mpi_comm.gather(worker_train_inten, root=0)








    if mpi_rank==0:


        # run_train_inten = [inten for worker in run_train_inten_gathered for inten in worker]

        run_train_inten = np.concatenate(run_train_inten_gathered, axis=0)

        run_mean_im = run_sum_im/np.sum(args.n_trains*n_pulses)
        run_vari_im = run_sumsq_im/np.sum(args.n_trains*n_pulses)

        t1 = time.perf_counter() - t0
        print(f'Total calculation time: {round(t1/60, 2)} minutes')

        with h5py.File(args.h5fname, 'w') as h5out:
            h5out['/mean_im'] =run_mean_im
            h5out['/mask'] = mask
            h5out['/sum_im'] = run_sum_im
            h5out['/sumsq_im'] = run_sumsq_im
            h5out['/vari_im'] = run_vari_im

            h5out['/train_ids'] = run_train_ids
            h5out['/train_inten'] = run_train_inten[:,:n_pulses]

            h5out['/n_pulses'] = n_pulses
            h5out['/n_trains'] = args.n_trains
            h5out['/run'] = args.run


















