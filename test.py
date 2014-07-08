from pykCSD.pykCSD import KCSD3D
from pylab import *

elec_pos = np.array([(0,0,0), (0,0,1), (0,1,0), (1,0,0), (0,1,1), (1,1,0), (1,0,1), (1,1,1), (0.5,0.5,0.5), (1.2,1.2,1.2) ])
pots = np.array([-1,0,1,1,0,-1,0,0,-1, -1])

k = KCSD3D(elec_pos, pots, params={'gdX':0.02, 'gdY': 0.02, 'gdZ': 0.02, 'n_sources':64})
k.calculate_matrices()

k.estimate_pots()
k.estimate_csd()