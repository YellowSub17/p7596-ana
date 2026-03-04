

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
        print('calc_mask.py')
        print(f'Running MPI with {mpi_size} rank(s).')

    parser = argparse.ArgumentParser("Calculate a mask for a run")

    parser.add_argument("run", type=int, help='Run number.')
    parser.add_argument("--h5fname", default=None, help='Name of the output h5 file.')
    parser.add_argument("--n-trains", type=int, default=-1, help='Number of trains to analyse.')

    args = parser.parse_args()

    if args.h5fname is None:
        args.h5fname =f'{cnst.H5OUT_DIR}/r{args.run:04}_mask.h5'



    ### For whatever reason, dark runs only use raw data
    run_raw = extra_data.open_run(proposal=cnst.PROPOSAL_NUM, run=args.run, data='raw')
    sel_raw = run_raw.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data', require_all=True)


    #get the train ids of the run (and limit them if args.n_trains was set
    run_train_ids = sel_raw.train_ids[:]
    n_total_trains = len(run_train_ids)
    if args.n_trains ==-1:
        args.n_trains = n_total_trains
    run_train_ids = run_train_ids[:args.n_trains]

    if mpi_rank ==0:
        print(f'Calculating mask for run {args.run} with {args.n_trains} trains.')


    #split the train ids to the mpi ranks, and get this rank's ids
    worker_train_ids = np.array_split(run_train_ids, mpi_size)[mpi_rank]
    print(f'Worker {mpi_rank}: First/Last train id is {worker_train_ids[0]}, {worker_train_ids[-1]}, with {len(worker_train_ids)} trains.')
    worker_sel_raw = sel_raw.select_trains(extra_data.by_id[worker_train_ids]) #select the trains 

    #init the sum and sum sq of the intensity
    worker_sum_im = np.zeros(cnst.DET_SHAPE)
    worker_sumsq_im = np.zeros(cnst.DET_SHAPE)



    t_loop0 = time.perf_counter()
    for i_train, (train_id, train_data_raw) in enumerate(worker_sel_raw.trains()): #for each train

        stack = extra_data.stack_detector_data(train_data_raw, 'image.data')[:,0,:,:] #pulses, ??, modules, fast scan, slow scan 

        worker_sum_im += np.nansum(stack, axis=0) # add all of the pulses to the running total
        worker_sumsq_im += np.nansum(stack**2, axis=0) # add all the pulses (squared) to the running total

        #log the time
        if mpi_rank==0 and i_train%10==0:
            t_loop1 = time.perf_counter() - t_loop0
            print(f'Loop {i_train} took {round(t_loop1)} seconds.')
            t_loop0 = time.perf_counter()

    # number of pulses in a train (should be 202 or 101)
    n_pulses_per_train = stack.shape[0]


    #mpi reduction, combine all the ranks back into rank 0
    if mpi_rank ==0:
        run_sum_im = np.zeros(cnst.DET_SHAPE)
        run_sumsq_im = np.zeros(cnst.DET_SHAPE)
    else:
        run_sum_im = None
        run_sumsq_im = None

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






    #only rank 0 should save the data
    if mpi_rank==0:

        #mean pulse (sum of the all the pulses divided by the number of pulses in the run [ntrains*npulses_in_train])
        run_mean_im = run_sum_im/np.sum(args.n_trains*n_pulses_per_train)
        #variance
        run_vari_im = run_sumsq_im/np.sum(args.n_trains*n_pulses_per_train) - run_mean_im**2

        t1 = time.perf_counter() - t0
        print(f'Total calculation time: {round(t1/60, 2)} minutes')

        with h5py.File(args.h5fname, 'w') as h5out:
            h5out['/mean_im'] =run_mean_im
            h5out['/sum_im'] = run_sum_im
            h5out['/sumsq_im'] = run_sumsq_im
            h5out['/vari_im'] = run_vari_im
            h5out['/train_ids'] = run_train_ids
            h5out['/n_pulses_per_train'] = n_pulses_per_train
            h5out['/n_trains'] = args.n_trains
            h5out['/run'] = args.run
            h5out['/calc_time'] = round(t1/60, 2)


















