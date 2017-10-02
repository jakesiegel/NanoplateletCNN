<!-- 
# This is the code package to train a CNN on simulations of the X^3 response function.

The goal is to understand the spectra of CdSe nanoplatelets, which have a stable biexciton
	at room temperature, 1 SO phonon and possibly up to 3 LO phonons.  The potential ways 
	these could couple make a Monte Carlo fit prohibitive. 

The general outline for performing the analysis goes as follows:
1 - build a list of several simulation parameters you want to use to train the CNN
		I specified in this list:
			- biexciton binding energies
			- phonon energies
			- number of LO phonons that can be involved in a single transition
			- phonon coupling scheme (single, double uncoupled, or double coupled)
2 - for each entry on that list, the code will do 30 simulations, with bounded randomly 
		selected values for:
			- dipole coupling strengths in the coupling matrix
			- initial state distribution
			- the specific exciton frequency
			- homogenous dephasing parameter
			- inhomogenous dephasing parameter
		The 30 simulations will be split evenly with 15 assuming there are up to 3 LO phonon
		states and 15 assuming there are up to 4 LO phonon states.
 -->