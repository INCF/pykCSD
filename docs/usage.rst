========
Usage
========

To use pykCSD in a project::

	import pykCSD.pykCSD import KCSD
	import numpy as np
	
	elec_pos = np.array([[0, 0], [0, 1], [1, 0], [1,1], [0.5, 0.5]])
	pots = np.array([0, 0, 0, 0, 1])
	params = {'gdX': 0.05, 'gdY': 0.05}
	
	k = KCSD(elec_pos, pots, params)
	
	k.estimate_pots()
	k.estimate_csd()
	
	k.plot_all()

