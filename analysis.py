import psana
import numpy as np
from algorithms import cfd

def process_waveform(evt, config):
    det = config['det']
    wf = det.waveform(evt)
    if wf is None:
        return [np.zeros(0, dtype=np.float32) for c in config['channels']]
    taxis = det.wftime(evt)
    peaks = [cfd(taxis[c], wf[c], config['fraction'], config['delay'], config['threshold'], 61).astype(np.float32) for c in config['channels']]
    return peaks

class QuadAnodeDLD:
    def __init__(self, dld_config, mcp_config):
        self.dld_config = dld_config
        self.mcp_config = mcp_config
        
    def get_detectors(self):
        name = self.dld_config['det']
        self.dld_config['det'] = psana.Detector(name)
        name = self.mcp_config['det']
        self.mcp_config['det'] = psana.Detector(name)
        
    def process_event(self, evt, batch):
        peaks = process_waveform(evt, self.dld_config)
        keys = ['x1_peaks', 'x2_peaks', 'y1_peaks', 'y2_peaks']
        for i, k in enumerate(keys):
            batch.add_data(k, peaks[i])
        
        keys = ['x1_counts', 'x2_counts', 'y1_counts', 'y2_counts']
        for i, k in enumerate(keys):
            batch.add_data(k, peaks[i].size)
        
        peaks, = process_waveform(evt, self.mcp_config)
        batch.add_data('mcp_counts', peaks.size)
        batch.add_data('mcp_peaks', peaks)
        
'''
class VonHamos:
    def __init__(self, config):
        self.config = config
        
    def get_detectors(self):
        self.det = psana.Detector(config['det'])
        
    def process_event(self, evt, batch):
        img = self.det.calib(evt)
        if img is None:
            
        ndroplets, x, y, adu = find_droplets(img, self.config['seed_threshold'], self.config['join_threshold'])
        if ndroplets > 0:
            batch.add_data('ndroplets', ndroplets)
            batch.add_data('x', x)
            batch.add_data('y', y)
            batch.add_data('adu', adu)
'''             