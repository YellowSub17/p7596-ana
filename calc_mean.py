

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
    parser.add_argument("--high-i-thresh", type=float, default=0, help='Name of the mask h5 file.')


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
        mask = np.ones(cnst.DET_SHAPE)





    run_proc = extra_data.open_run(proposal=cnst.PROPOSAL_NUM, run=args.run, data='proc')
    sel_proc = run_proc.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data', require_all=True)


    run_raw = extra_data.open_run(proposal=cnst.PROPOSAL_NUM, run=args.run, data='raw')
    sel_raw = run_raw.select('SPB_XTD9_XGM/XGM/DOOCS', require_all=True)


    run_train_ids = sel_proc.train_ids[:]
    n_total_trains = len(run_train_ids)
    if args.n_trains ==-1:
        args.n_trains = n_total_trains
    run_train_ids = run_train_ids[:args.n_trains]

    if mpi_rank ==0:
        print(f'Calculating average for run {args.run} with {args.n_trains} trains.')
        print(f'Intensity threshold: {args.high_i_thresh}')



    worker_train_ids = np.array_split(run_train_ids, mpi_size)[mpi_rank]
    print(f'Worker {mpi_rank}: First/Last train id is {worker_train_ids[0]}, {worker_train_ids[-1]}, with {len(worker_train_ids)} trains.')
    worker_sel_proc = sel_proc.select_trains(extra_data.by_id[worker_train_ids])
    worker_sel_raw = sel_raw.select_trains(extra_data.by_id[worker_train_ids])

    # worker_sum_im = np.zeros(cnst.DET_SHAPE)
    # worker_sumsq_im = np.zeros(cnst.DET_SHAPE)
    worker_pulse_inten = np.zeros( (worker_train_ids.size, 202) )


    worker_high_i_sum_im = np.zeros(cnst.DET_SHAPE)
    worker_low_i_sum_im = np.zeros(cnst.DET_SHAPE)

    worker_high_i_sum_sq_im = np.zeros(cnst.DET_SHAPE)
    worker_low_i_sum_sq_im = np.zeros(cnst.DET_SHAPE)

    worker_n_high_i_pulses = 0
    worker_n_low_i_pulses = 0



    t_loop0 = time.perf_counter()
    for i_train, (train_id, train_data_proc) in enumerate(worker_sel_proc.trains()):


        stack = extra_data.stack_detector_data(train_data_proc, 'image.data') #pulses, modules, fast scan, slow scan 

        stack[:, mask>1] = np.nan

        pulse_inten = np.nanmean(stack, axis=(1,2,3))
        worker_pulse_inten[i_train, :stack.shape[0]] = pulse_inten

        high_i_pulses = np.where(pulse_inten>=args.high_i_thresh)
        low_i_pulses = np.where(pulse_inten<args.high_i_thresh)

        worker_n_high_i_pulses += high_i_pulses[0].size
        worker_n_low_i_pulses += low_i_pulses[0].size

        worker_high_i_sum_im +=np.nansum(stack[high_i_pulses], axis=0)
        worker_high_i_sum_sq_im +=np.nansum(stack[high_i_pulses]**2, axis=0)
        worker_low_i_sum_im +=np.nansum(stack[low_i_pulses], axis=0)
        worker_low_i_sum_sq_im +=np.nansum(stack[low_i_pulses]**2, axis=0)




        # worker_sum_im += np.nansum(stack, axis=0)
        # worker_sumsq_im += np.nansum(stack**2, axis=0)

        if mpi_rank==0 and i_train%10==0:
            t_loop1 = time.perf_counter() - t_loop0
            print(f'Loop {i_train} took {round(t_loop1)} seconds.')
            t_loop0 = time.perf_counter()

    worker_sum_im = worker_high_i_sum_im + worker_low_i_sum_im
    worker_sum_sq_im = worker_high_i_sum_sq_im + worker_low_i_sum_sq_im

    n_pulses_per_train = stack.shape[0]

    worker_xgm = np.zeros( worker_train_ids.size )
    for i_train, (train_id, train_data_raw) in enumerate(worker_sel_raw.trains()):
        xgm_val = train_data_raw['SPB_XTD9_XGM/XGM/DOOCS']['pulseEnergy.photonFlux.value']
        worker_xgm[i_train] = xgm_val


    if mpi_rank ==0:
        # run_sum_im = np.zeros(cnst.DET_SHAPE)
        run_high_i_sum_im = np.zeros(cnst.DET_SHAPE)
        run_low_i_sum_im = np.zeros(cnst.DET_SHAPE)
        run_high_i_sum_sq_im = np.zeros(cnst.DET_SHAPE)
        run_low_i_sum_sq_im = np.zeros(cnst.DET_SHAPE)
        # run_sumsq_im = np.zeros(cnst.DET_SHAPE)
        run_pulse_inten = np.zeros(args.n_trains)
    else:
        # run_sum_im = None
        run_high_i_sum_im = None
        run_low_i_sum_im = None
        run_high_i_sum_sq_im = None
        run_low_i_sum_sq_im = None
        # run_sumsq_im = None
        run_pulse_inten = None

    mpi_comm.Reduce(
            [worker_high_i_sum_im, MPI.FLOAT],
            [run_high_i_sum_im, MPI.FLOAT],
            op=MPI.SUM,
            root=0
            )

    mpi_comm.Reduce(
            [worker_low_i_sum_im, MPI.FLOAT],
            [run_low_i_sum_im, MPI.FLOAT],
            op=MPI.SUM,
            root=0
            )

    mpi_comm.Reduce(
            [worker_high_i_sum_sq_im, MPI.FLOAT],
            [run_high_i_sum_sq_im, MPI.FLOAT],
            op=MPI.SUM,
            root=0
            )

    mpi_comm.Reduce(
            [worker_low_i_sum_sq_im, MPI.FLOAT],
            [run_low_i_sum_sq_im, MPI.FLOAT],
            op=MPI.SUM,
            root=0
            )


    # mpi_comm.Reduce(
            # [worker_sum_im, MPI.FLOAT],
            # [run_sum_im, MPI.FLOAT],
            # op=MPI.SUM,
            # root=0
            # )

    # mpi_comm.Reduce(
            # [worker_sumsq_im, MPI.FLOAT],
            # [run_sumsq_im, MPI.FLOAT],
            # op=MPI.SUM,
            # root=0
            # )


    run_pulse_inten_gathered = mpi_comm.gather(worker_pulse_inten, root=0)
    run_xgm_gathered = mpi_comm.gather(worker_xgm, root=0)

    run_n_low_i_pulses = mpi_comm.reduce(worker_n_low_i_pulses, root=0)
    run_n_high_i_pulses = mpi_comm.reduce(worker_n_high_i_pulses, root=0)






    if mpi_rank==0:

        run_sum_im = run_low_i_sum_im + run_high_i_sum_im
        run_sum_sq_im = run_low_i_sum_sq_im + run_high_i_sum_sq_im


        run_pulse_inten = np.concatenate(run_pulse_inten_gathered, axis=0)
        run_xgm = np.concatenate(run_xgm_gathered, axis=0)

        run_mean_im = run_sum_im/np.sum(args.n_trains*n_pulses_per_train)
        run_vari_im = run_sum_sq_im/np.sum(args.n_trains*n_pulses_per_train) - run_mean_im**2

        run_high_i_mean_im = run_high_i_sum_im/run_n_high_i_pulses
        run_high_i_vari_im = run_sum_sq_im/run_n_high_i_pulses - run_high_i_mean_im**2

        run_low_i_mean_im = run_low_i_sum_im/run_n_low_i_pulses
        run_low_i_vari_im = run_sum_sq_im/run_n_low_i_pulses - run_low_i_mean_im**2

        t1 = time.perf_counter() - t0
        print(f'Total calculation time: {round(t1/60, 2)} minutes')

        with h5py.File(args.h5fname, 'w') as h5out:
            h5out['/calc_time'] = round(t1/60,2)
            
            h5out['/mean_im'] =run_mean_im
            h5out['/mask'] = mask
            h5out['/sum_im'] = run_sum_im
            h5out['/sum_sq_im'] = run_sum_sq_im
            h5out['/vari_im'] = run_vari_im

            h5out['/train_ids'] = run_train_ids
            h5out['/xgm'] = run_xgm
            h5out['/pulse_inten'] = run_pulse_inten[:,:n_pulses_per_train]

            h5out['/n_pulses_per_train'] = n_pulses_per_train
            h5out['/n_trains'] = args.n_trains
            h5out['/run'] = args.run


            h5out['/high_i_thresh'] = args.high_i_thresh
            h5out['/low_i/sum_im'] = run_low_i_sum_im
            h5out['/low_i/sum_sq_im'] = run_low_i_sum_sq_im
            h5out['/low_i/mean_im'] = run_low_i_mean_im
            h5out['/low_i/vari_im'] = run_low_i_vari_im
            h5out['/low_i/n_pulses'] = run_n_low_i_pulses

            h5out['/high_i/sum_im'] = run_high_i_sum_im
            h5out['/high_i/sum_sq_im'] = run_high_i_sum_sq_im
            h5out['/high_i/mean_im'] = run_high_i_mean_im
            h5out['/high_i/vari_im'] = run_high_i_vari_im
            h5out['/high_i/n_pulses'] = run_n_high_i_pulses



















