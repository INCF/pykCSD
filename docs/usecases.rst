==========
Use Cases
==========

With the pykCSD toolbox you can estimate 1D, 2D and 3D potentials and CSD based on your input data.
Here are the basic examples for each of the reconstructions.

Sample 1D reconstruction::

	from pykCSD.pykCSD import KCSD
	import numpy as np

	#the most inner list corresponds to a position of one electrode
	elec_pos = np.array([[0], [1], [2], [3], [4]])

	#the most inner list corresponds to a time recording made with one electrode
	pots = np.array([[0], [1], [-1], [0], [0]])

	k = KCSD(elec_pos, pots, params)
	
	k.estimate_pots()
	k.estimate_csd()
	
	k.plot_all()

//image//

You can use cross validation to validate your results::

	from pykCSD import cross_validation as cv
	from sklearn.cross_validation import LeaveOneOut

	index_generator = KFold(len(k.k_pot.shape[0]), indices=True)
	lambdas = [10/x for x in xrange(0, 10)]
	
	k.lambd = cv.choose_lambda(lambdas, k.sampled_pots, k.k_pot, k.elec_pos, index_generator)

	k.estimate_pots()
	k.estimate_csd()
	
	k.plot_all()

//image after CV//

Sample 2D reconstruction::

	from pykCSD.pykCSD import KCSD
	import numpy as np
	
	elec_pos = np.array([[0, 0], [0, 1], [1, 0], [1,1], [0.5, 0.5]])
	pots = np.array([[0], [0], [0], [0], [1]])
	params = {'gdX': 0.05, 'gdY': 0.05}
	
	k = KCSD(elec_pos, pots, params)
	
	k.estimate_pots()
	k.estimate_csd()
	
	k.plot_all()

//image//


Sample 3D reconstruction::

	from pykCSD.pykCSD import KCSD
	import numpy as np

//image//








