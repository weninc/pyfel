import psana
import argparse
import processing
import numpy as np
from analysis import QuadAnodeDLD
from algorithms import find_blobs
from mpi4py import MPI

experiment = 'amox27716'
ffb = False
batch_size = 100

parser = argparse.ArgumentParser()
parser.add_argument('-r','--run', help='run number', required=True, type=int)
args = parser.parse_args()

exp_string = 'exp=%s:run=%d:smd' %(experiment, args.run)
if ffb:
    exp_string += ':dir=/reg/d/ffb/%s/%s/xtc:live' %(experiment[:3], experiment)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def missing_values(batch):
        batch.add_data('nblobs', 0)
        batch.add_data('x', np.zeros(0, dtype=np.float32))
        batch.add_data('y', np.zeros(0, dtype=np.float32))
        batch.add_data('adu_sum', np.zeros(0, dtype=np.float32))

if rank == 0:
    processing.master('/reg/d/psdm/amo/amox27716/results/weninc/run-%03d.h5' %args.run)
else:
    batch = processing.Batch(batch_size)
    ds = psana.DataSource(exp_string)
    epics = ds.env().epicsStore()
    
    dld_config = {'det': psana.Detector('ACQ1'), 'channels': [2, 3, 4, 5], 'fraction': 0.8, 'delay': 15, 'threshold': 0.04}
    mcp_config = {'det': psana.Detector('ACQ1'), 'channels': [6], 'fraction': 0.8, 'delay': 15, 'threshold': 0.04}
    quad_anode = QuadAnodeDLD(dld_config, mcp_config)
    det = psana.Detector('OPAL3')
    
    events = processing.events(ds, batch)
    for i, evt in enumerate(events):
        quad_anode.process_event(evt, batch)
        img = det.raw(evt)
        if img is None:
            missing_values(batch)
        nblobs, x, y, adu_sum = find_blobs(img, 50, 150)
        if nblobs > 0:
            batch.add_data('nblobs', nblobs)
            batch.add_data('x', x)
            batch.add_data('y', y)
            batch.add_data('adu_sum', adu_sum)
        else:
            missing_values(batch)
        