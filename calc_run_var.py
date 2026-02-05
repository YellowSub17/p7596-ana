
import os

import h5py
import numpy as np
import cnst
import argparse
import extra_data
import extra_geom
from extra_data.components import AGIPD1M
import time

from mpi4py import MPI






if __name__ == '__main__':

    t0  = time.perf_counter()

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank ==0:
        print(f'Running MPI with {mpi_size} rank(s)')

    parser = argparse.ArgumentParser("Calculate sum and mean of run.")

    parser.add_argument("run", type=int, help='Run number.')
    parser.add_argument("--h5fname", default=None, help='Name of the input/output h5 file.')


    args = parser.parse_args()

    run = extra_data.open_run(proposal=cnst.PROPOSAL_NUM, run=args.run)
    sel = run.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data')


    if args.h5fname is None:
        args.h5fname =f'{cnst.H5OUT_DIR}/r{args.run:04}_ana.h5'
    assert os.path.exists(args.h5fname), f'h5 file {args.h5fname} does not exist'


    with h5py.File(f'{args.h5fname}', 'r') as f:
        f_train_ids = f['/train_ids'][...]
        f_n_trains = f['/n_trains'][...]
        f_n_pulses = f['/n_pulses'][...]
        f_run =  f['/run'][...]
        f_mean_im = f['/mean_im'][...]

    assert f_run == args.run, f'cmd input run {args.run} is different from file run {f_run}'
    assert f_n_trains >= mpi_size, 'TOO FEW TRAINS OR TOO MANY MPI RANKS'


    if mpi_rank ==0:
        print(f'Calculating average for run {args.run} with {f_n_trains} trains')



    worker_train_ids = np.array_split(f_train_ids, mpi_size)[mpi_rank]


    run = extra_data.open_run(proposal=cnst.PROPOSAL_NUM, run=args.run)

    worker_run = run.select_trains(extra_data.by_id[worker_train_ids])

    worker_sel = worker_run.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data')

    worker_train_ids = np.array(worker_sel.train_ids)
    worker_ntrains = worker_train_ids.size

    worker_sum_im = np.zeros(cnst.DET_SHAPE)

    print(f'Worker {mpi_rank} handling {worker_ntrains} trains.')

    t_loop0 = time.perf_counter()
    for i_train, (train_id, train_data) in enumerate(worker_sel.trains()):

        try:
            stack = extra_data.stack_detector_data(train_data, 'image.data')[:, 0,...] #pulses, ??, modules, fast scan, slow scan 
        except ValueError:
            print(f'Rank {mpi_rank}: Generating stack failed.\n\t{i_train=}, {train_id}: mean calc')
            continue


        train_min_mean_sq = (stack - f_mean_im)**2

        train_sum = train_min_mean_sq.sum(axis=0)

        worker_sum_im += train_sum


        if mpi_rank==0 and i_train%10==0:
            t_loop1 = time.perf_counter() - t_loop0
            print(f'Loop {i_train} took {round(t_loop1)} seconds.')
            t_loop0 = time.perf_counter()


    n_pulses = stack.shape[0]

    if mpi_rank ==0:
        run_sum_im = np.zeros(cnst.DET_SHAPE)
    else:
        run_sum_im = None

    mpi_comm.Reduce(
            [worker_sum_im, MPI.FLOAT],
            [run_sum_im, MPI.FLOAT],
            op=MPI.SUM,
            root=0
            )



    all_worker_train_ids = mpi_comm.gather(worker_train_ids, root=0)


    if mpi_rank==0:

        run_train_ids = []
        for worker in all_worker_train_ids:
            run_train_ids += list(worker)


        run_var_im = run_sum_im/np.sum(args.n_trains*n_pulses)

        t1 = time.perf_counter() - t0
        print(f'Time: {round(t1, 2)}')


        with h5py.File(args.h5fname, 'a') as h5out:
            h5out['/var_im'] = run_var_im











