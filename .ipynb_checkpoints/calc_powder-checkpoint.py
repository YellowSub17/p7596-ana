

import h5py
import numpy as np
import cnst
import argparse
import extra_data
import extra_geom
from extra_data.components import AGIPD1M
import time
from pyFAI.integrator.azimuthal import AzimuthalIntegrator



import os
from mpi4py import MPI






if __name__ == '__main__':

    t0  = time.perf_counter()

    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    if mpi_rank ==0:
        print('calc_powder.py')
        print(f'Running MPI with {mpi_size} rank(s).')

    parser = argparse.ArgumentParser("Calculate sum and mean of run.")

    parser.add_argument("run", type=int, help='Run number.')
    parser.add_argument("--nq", type=int, default=256, help='Number of q bins.')
    parser.add_argument("--h5fname", default=None, help='Name of the output h5 file.')
    parser.add_argument("--n-trains", type=int, default=-1, help='Number of trains to analyse.')
    parser.add_argument("--h5mask", default=None, help='Name of the mask h5 file.')
    parser.add_argument("--geomfname", default=None, help='Name of the geom file.')
    parser.add_argument("--rmin", default=5.728, help='Minimum r distance (mm) on detector')
    parser.add_argument("--rmax", default=101.9, help='Maxium r distance (mm) on detector')

    parser.add_argument("--high-i-thresh", type=float, default=0, help='')

    args = parser.parse_args()

    if args.h5fname is None:
        args.h5fname =f'{cnst.H5OUT_DIR}/r{args.run:04}_powder.h5'

    if args.h5mask is None:
        args.h5mask =f'{cnst.MASK_PATH}'

    if args.geomfname is None:
        args.geomfname = f'{cnst.GEOM_PATH}'
    geom = extra_geom.AGIPD_1MGeometry.from_crystfel_geom(args.geomfname)




    if os.path.exists(args.h5mask):
        with h5py.File(args.h5mask, 'r') as f:
            mask = f['/mask'][...]
    else:
        if mpi_rank ==0:
            print('Warning: No mask file found. Using empty mask.')
        mask = np.ones(cnst.DET_SHAPE)



    if args.high_i_thresh is None:
        args.high_i_thresh = 0


    run_proc = extra_data.open_run(proposal=cnst.PROPOSAL_NUM, run=args.run, data='proc')
    sel_proc = run_proc.select('SPB_DET_AGIPD1M-1/DET/*CH0:xtdf', 'image.data', require_all=True)


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

    worker_pulse_inten = np.zeros( (worker_train_ids.size, 202) )

    worker_train_powder = np.zeros( (worker_train_ids.size, args.nq) )
    worker_high_i_powder = np.zeros(  args.nq)
    worker_low_i_powder = np.zeros(  args.nq)

    worker_n_high_i_pulses = 0
    worker_n_low_i_pulses = 0



    ai = AzimuthalIntegrator(
            detector=geom.to_pyfai_detector(),
            dist = cnst.GEOM_CLEN,
            wavelength = (12.3984/ (cnst.GEOM_EV*1e-3)) *1e-10
            )



    t_loop0 = time.perf_counter()
    for i_train, (train_id, train_data_proc) in enumerate(worker_sel_proc.trains()):


        stack = extra_data.stack_detector_data(train_data_proc, 'image.data') #pulses, modules, fast scan, slow scan 

        stack[:, mask>1] = np.nan

        stack_sum = np.nansum(stack, axis=0)

        rint, I  = ai.integrate1d(
                stack_sum.reshape(16*512, 128),
                npt = args.nq,
                polarization_factor=None,
                unit='r_mm',
                radial_range = (args.rmin, args.rmax)
                )

        worker_train_powder[i_train, :] = I

        pulse_inten = np.nanmean(stack, axis=(1,2,3))
        worker_pulse_inten[i_train, :stack.shape[0]] = pulse_inten

        high_i_pulses = np.where(pulse_inten>=args.high_i_thresh)
        low_i_pulses = np.where(pulse_inten<args.high_i_thresh)
        worker_n_high_i_pulses += high_i_pulses[0].size
        worker_n_low_i_pulses += low_i_pulses[0].size


        high_i_sum = np.nansum(stack[high_i_pulses], axis=0)
        low_i_sum = np.nansum(stack[low_i_pulses], axis=0)

        _, high_I = ai.integrate1d(high_i_sum.reshape(16*512, 128),
                npt = args.nq,
                polarization_factor=None,
                unit='r_mm',
                radial_range = (args.rmin, args.rmax)
                )

        _, low_I = ai.integrate1d(low_i_sum.reshape(16*512, 128),
                npt = args.nq,
                polarization_factor=None,
                unit='r_mm',
                radial_range = (args.rmin, args.rmax)
                )

        worker_high_i_powder += high_I
        worker_low_i_powder += low_I








        if mpi_rank==0 and i_train%10==0:
            t_loop1 = time.perf_counter() - t_loop0
            print(f'Loop {i_train} took {round(t_loop1)} seconds.')
            t_loop0 = time.perf_counter()

    n_pulses_per_train = stack.shape[0]


    if mpi_rank ==0:
        run_pulse_inten = np.zeros(args.n_trains)
        run_train_powder = np.zeros((args.n_trains, args.nq))
        run_high_i_powder = np.zeros( args.nq)
        run_low_i_powder = np.zeros( args.nq)
    else:
        run_pulse_inten = None
        run_train_powder = None
        run_high_i_powder =None
        run_low_i_powder =None


    mpi_comm.Reduce(
            [worker_high_i_powder, MPI.FLOAT],
            [run_high_i_powder, MPI.FLOAT],
            op=MPI.SUM,
            root=0
            )

    mpi_comm.Reduce(
            [worker_low_i_powder, MPI.FLOAT],
            [run_low_i_powder, MPI.FLOAT],
            op=MPI.SUM,
            root=0
            )



    run_n_low_i_pulses = mpi_comm.reduce(worker_n_low_i_pulses, root=0)
    run_n_high_i_pulses = mpi_comm.reduce(worker_n_high_i_pulses, root=0)

    run_pulse_inten_gathered = mpi_comm.gather(worker_pulse_inten, root=0)
    run_train_powder_gathered = mpi_comm.gather(worker_train_powder, root=0)








    if mpi_rank==0:

        run_pulse_inten = np.concatenate(run_pulse_inten_gathered, axis=0)
        run_train_powder = np.concatenate(run_train_powder_gathered, axis=0)

        run_train_powder *= 1/(args.n_trains*n_pulses_per_train)


        if run_n_high_i_pulses >0:
            run_high_i_powder *= 1/(run_n_high_i_pulses)

        if run_n_low_i_pulses >0:
            run_low_i_powder *= 1/(run_n_low_i_pulses)





        t1 = time.perf_counter() - t0
        print(f'Total calculation time: {round(t1/60, 2)} minutes')

        with h5py.File(args.h5fname, 'w') as h5out:
            h5out['/calc_time'] = round(t1/60,2)

            h5out['/mask'] = mask

            h5out['/train_ids'] = run_train_ids
            h5out['/pulse_inten'] = run_pulse_inten[:,:n_pulses_per_train]

            h5out['/n_pulses_per_train'] = n_pulses_per_train
            h5out['/n_trains'] = args.n_trains
            h5out['/run'] = args.run

            h5out['/nq'] = args.nq
            h5out['/rint'] = rint

            h5out['/rmin'] = args.rmin
            h5out['/rmax'] = args.rmax


            h5out['/train_powder'] = run_train_powder
            h5out['/high_i_thresh'] = args.high_i_thresh
            h5out['/high_i_powder'] = run_high_i_powder
            h5out['/low_i_powder'] = run_low_i_powder

            h5out['/n_high_i_pulses'] = run_n_high_i_pulses
            h5out['/n_low_i_pulses'] = run_n_low_i_pulses



















