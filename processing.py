from __future__ import absolute_import, division, print_function

import time
import psana
import tables
import numpy as np
from mpi4py import MPI

class Batch():
    def __init__(self, batch_size):
        self.data = {}
        self.counter = 0
        self.batch_size = batch_size
        self.comm = MPI.COMM_WORLD
        
    def add_data(self, key, value):
        if key in self.data:
            l = self.data[key]
            assert type(value) == type(l[-1])
            l.append(value)
        else:
            self.data[key] = [value]   
            
    def event_complete(self):
        self.counter += 1
        if self.counter == self.batch_size:
            self.send()
            self.counter = 0
                
    def send(self):
        d = {}
        for key, value in self.data.items():
            t = type(value[0])
            if t is np.ndarray:
                d[key] = np.concatenate(value)
            elif t is int:
                d[key] = np.array(value, dtype=np.int64)
            elif t is float:
                d[key] = np.array(value, dtype=np.float32)
            else:
                raise TypeError('value must the np.ndarray or int or float')
        self.comm.send(d, dest=0, tag=1)
        self.data.clear()
               
def worker(rank, nworkers, batch_size, exp_string, analysis):
    comm = MPI.COMM_WORLD
    ds = psana.DataSource(exp_string)
    for a in analysis:
        a.get_detectors()
    batch = Batch(batch_size)
    for i, evt in enumerate(ds.events()):
        if i%nworkers != rank:
            continue 
        evt_id = evt.get(psana.EventId)
        if evt_id is None:
            continue
        batch.add_data('time', evt_id.time()[0] << 32 | evt_id.time()[1])
        for a in analysis:
            a.process_event(evt, batch)
        batch.event_complete()
    batch.send()
    comm.send('worker finished', dest=0, tag=0)

def master(run, nworkers):
    comm = MPI.COMM_WORLD
    fh = tables.open_file('/reg/d/psdm/xpp/xpptut15/scratch/weninc/run-%03d.h5' %run, 'w')
    dsets = {}
    active_workers = nworkers
    status = MPI.Status()
    nevents = 0
    print_events = 0
    while active_workers > 0:
        data = comm.recv(source=MPI.ANY_SOURCE, status=status)
        tag = status.Get_tag()
        if tag == 0:
            active_workers -= 1
            print('Worker finished')
        else:
            nevents += data['time'].size
            if nevents >= print_events:
                print_events += 1000
                print('%s %d events processed' %(time.ctime(), nevents))
            for key, value in data.items():
                if key not in dsets:
                    a = tables.Atom.from_dtype(value.dtype)
                    dsets[key] = fh.create_earray(fh.root, key, a, (0,))
                dsets[key].append(value)
    fh.close()
    

def main(exp, run, ffb, batch_size, analysis):  
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    nworkers = size - 1
    exp_string = 'exp=%s:run=%d:smd' %(exp, run)
    if ffb:
        exp_string += ':dir=/reg/d/ffb/%s/%s/xtc:live' %(exp[:3], exp)

    if rank == 0:    
        master(run, nworkers)
    else:
        worker(rank-1, nworkers, batch_size, exp_string, analysis)
