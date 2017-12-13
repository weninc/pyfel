import argparse
import processing
from analysis import QuadAnodeDLD

parser = argparse.ArgumentParser()
parser.add_argument('-r','--run', help='run number', required=True, type=int)
args = parser.parse_args()

### start of experiment specific stuff ###
exp = 'xpptut15'
ffb = False
batch_size = 100
dld_config = {'det': 'ACQ1', 'channels': [0, 1, 2, 3], 'fraction': 0.8, 'delay': 25, 'threshold': 0.035}
mcp_config = {'det': 'ACQ2', 'channels': [0], 'fraction': 0.8, 'delay': 25, 'threshold': 0.02}
analysis = []
analysis.append(QuadAnodeDLD(dld_config, mcp_config))
exp_string = 'exp=xpptut15:run=280'
### end of specific stuff ###
    
processing.main(exp, args.run, ffb, batch_size, analysis)
   