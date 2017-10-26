
# This is the code package to train a CNN on simulations of the X^3 response function.

The goal is to understand the 2D spectra of CdSe nanoplatelets, which have a stable biexciton
	at room temperature, 1 SO phonon and possibly up to 3 LO phonons.  The potential ways 
	these could couple make a Monte Carlo fit prohibitive. 

The general outline for performing the analysis goes as follows:

1) Build a list of several simulation parameters you want to use to train the CNN
		I specified in this list:
		
			- biexciton binding energies
			- phonon energies
			- number of LO phonons that can be involved in a single transition
			- phonon coupling scheme (single, double uncoupled, or double coupled)
			
	* This is the SimsList.py script, it will build a csv to serve as the simulations queue

2) For each entry on that list, the code will do 30 simulations, with bounded randomly 
		selected values for:
		
			- dipole coupling strengths in the coupling matrix
			- initial state distribution
			- the specific exciton frequency
			- homogenous dephasing parameter
			- inhomogenous dephasing parameter
			
	* The 30 simulations will be split evenly with 15 assuming there are up to 3 LO phonon
		states and 15 assuming there are up to 4 LO phonon states.
	
	* These simulations are stored in hdf5 format, with the simulation parameters stored
		the metadata.
		
	* This is Simulations_CdSe2.py and CdSeModule.py scripts.  Hdf5 files will be stored
		in batches of 30 simulations to make file transfer easier.

3)  After a sufficient training set is simulated, the CNN will be trained on ~90% of the
		simulations with the remaining simulations used for validation.
		
			- The CNN as initially set up has 6 convolutional layers, 2 pooling layers, 
				3 drop out layers, a dense layer and a softmax dense layer
			- The structure is Conv-Conv-Pool-Drop-Conv-Conv-Pool-Drop-Conv-Conv-Dense-Drop-Softmax
			- The structure was inspired by Tuccillo et al.  arXiv: 1701.05917v1 [astro-ph.IM],
				which used a similar scheme for training a CNN on simulations for Galaxy
				Morphology
			- The classes to classify the simulations into are built from the hdf5 metadata
				encoded in the simulation.  Depending on which parameters you are interested
				in classifying, you can get several hundred class labels.  For the code deposited
				here, I determined that I wanted to classify on the model and number of phonons
				coupled per transition. As there are 3 models each simulated with 3 different
				conditions for the number of phonons / transition, that makes 9 classes.  The
				number of potential classes gets combinatorially huge.

Some notes about the methodology:

1) The CNN is looking at the data as if it were an RGB image.  Instead of RGB layers, however,
		they are the S1 (rephasing), S2 (non-rephasing), and S3 (two-quantum) scans of 
		different quantum mechanical pathways.  The same physics governs all it and this is
		attempting to analyze them all together for the first time.

2) The actual models are in the CdSeModule.py file.  These are simply the models I deemed most
		likely for CdSe nanoplatelets based on my data and Raman spectra available in the 
		literature.
3) How well this approach works is difficult to determine.  It does great against other 
		simulated data. Ideally, we would have some test dataset of labeled experimental data.  
		Of course, if we had that, we wouldn't need the CNN, now would we?  

Notes about the implementation:

1) The simulations are done using a package called theano.  It is an optimization package.
		The theano version ran a simulation 40-50% faster than the numpy version on my MacBook Air.
		The numpy version itself ran a simulation twice as fast as the Matlab version used
		in the lab.  All in all, it now takes about 25s per simulation on my laptop.
		One very nice feature of theano is that it can do calculations on an NVidia GPU, in addition
		to the CPU, using a package called Cuda.  I don't have an NVidia GPU on my computer,
		but using one would speed both the simulations and training the CNN (which is in keras,
		and uses a theano backend on my set up).

2) Theano is apparently going to stop being supported.  Keras (the module used for the CNN)
		can run on a tensorflow backend, so that won't be a problem.  I'm not sure what to
		do about the simulations, though.
