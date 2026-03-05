

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

    parser = argparse.ArgumentParser("Calculate the powder pattern for every pulse.")

    parser.add_argument("run", type=int, help='Run number.')
    parser.add_argument("--nq", type=int, default=256, help='Number of q bins.')
    parser.add_argument("--h5fname", default=None, help='Name of the output h5 file.')
    parser.add_argument("--n-trains", type=int, default=-1, help='Number of trains to analyse.')
    parser.add_argument("--h5mask", default=None, help='Name of the mask h5 file.')
    parser.add_argument("--geomfname", default=None, help='Name of the geom file.')
    parser.add_argument("--qmin", default=2.17, type=float, help='')
    parser.add_argument("--qmax", default=35, type=float, help='')

    # parser.add_argument("--qreg1", type=float, nargs=2, default=[2, 4], help='')
    # parser.add_argument("--qreg2", type=float, nargs=2, default=[6.5, 8.5], help='')
    # parser.add_argument("--ratio-thresh", default=10, type=float,  help='')



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

    worker_pulse_inten = np.zeros( (worker_train_ids.size, 202) )

    worker_powders = []
    worker_powder_coords = []

    worker_n_conden = 0


    ai = AzimuthalIntegrator(
            detector=geom.to_pyfai_detector(),
            dist = cnst.GEOM_CLEN,
            wavelength = (12.3984/ (cnst.GEOM_EV*1e-3)) *1e-10
            )






    t_loop0 = time.perf_counter()
    for i_train, (train_id, train_data_proc) in enumerate(worker_sel_proc.trains()):


        stack = extra_data.stack_detector_data(train_data_proc, 'image.data') #pulses, modules, fast scan, slow scan 

        stack[:, mask>1] = np.nan


        for i_pulse, pulse in enumerate(stack):

            qint, I  = ai.integrate1d(
                    pulse.reshape(16*512, 128),
                    npt = args.nq,
                    polarization_factor=None,
                    unit='q_nm^-1',
                    radial_range = (args.qmin, args.qmax)
                    )

            
            worker_powders.append(I)
            worker_powder_coords.append( [train_id, i_pulse] )




        if mpi_rank==0 and i_train%10==0:
            t_loop1 = time.perf_counter() - t_loop0
            print(f'Loop {i_train} took {round(t_loop1)} seconds.')
            t_loop0 = time.perf_counter()

    n_pulses_per_train = stack.shape[0]


    if mpi_rank ==0:
        run_powders = []
        run_powder_coords = []
    else:
        run_powders = None




    run_powders_gathered = mpi_comm.gather(worker_powders, root=0)
    run_powder_coords_gathered = mpi_comm.gather(worker_powder_coords, root=0)




    if mpi_rank==0:


        # run_powders_gathered[:] = [worker for worker in run_powders_gathered if worker] #remove empty workers before concatenation
        run_powders = np.concatenate(run_powders_gathered, axis=0)

        # run_powder_coords_gathered[:] = [worker for worker in run_powder_coords_gathered if worker] #remove empty workers before concatenation
        run_powder_coords = np.concatenate(run_powder_coords_gathered, axis=0)




        t1 = time.perf_counter() - t0
        print(f'Total calculation time: {round(t1/60, 2)} minutes')

        with h5py.File(args.h5fname, 'w') as h5out:
            h5out['/calc_time'] = round(t1/60,2)

            h5out['/mask'] = mask

            h5out['/train_ids'] = run_train_ids

            h5out['/n_pulses_per_train'] = n_pulses_per_train
            h5out['/n_trains'] = args.n_trains
            h5out['/run'] = args.run

            h5out['/nq'] = args.nq
            h5out['/qint'] = qint


            h5out['/powders'] = run_powders


            h5out['/qmin'] = args.qmin
            h5out['/qmax'] = args.qmax
            h5out['/powder_coords'] = run_powder_coords





















