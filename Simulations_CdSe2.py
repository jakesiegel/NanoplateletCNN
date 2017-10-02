# CdSe 2D simulations in python / theano
import numpy as np
import time
import math
from numpy import pi
import matplotlib.pyplot as plt
import scipy.linalg as linalg
from theano import *
import theano.tensor as TT
import theano.tensor.slinalg as TTslinalg
import theano.tensor.nlinalg as TTnlinalg
import h5py
import CdSeTheanoModule as CdSe

##########################
# Define constants & parameters
filenameoutroot = 'CdSeSim'
paramlist = 'CdSeList.csv'

h = 4.135 #meV*ps
c = 2.998e5 #nm/ps
Ec = 2374 # carrier frequency meV
wl = -2*pi*(2295-Ec)/h # central laser frequency (THz)
tau = 0.05 # Laser pulse width (ps)
T = 298 # temperature (K)

for i in xrange(0,1):
	
	dE,model,dV1,dV2,n_coup,batch = CdSe.getParamsFromFile(paramlist)
	fileout = filenameoutroot+str(batch)+'.hdf5'
	if model=='single':dV=dV1
	else: dV=[dV1,dV2]

	# Exciton Frequency
	Ex = np.random.uniform(2250,2260)
	# initial state distribution
	a = np.random.uniform(0.6,0.9)
	b = ((1-a)/(np.random.uniform(1,1.5)))
	init = np.asarray([a, b, (1-a-b)])

	# Homogeneous dephasing parameter
	Gamma = np.random.uniform(.125,.75)

	# Inhomogeneous dephasing parameter
	sparam = np.random.uniform(0,0.05)

	# Simulation time settings
	NAbs = 301
	NEmi = 301

	# Set up time parameters
	t1S3 = np.linspace(0.,-7.5,num=NAbs)
	t2S3 = t1S3
	t3 = 0
	t2 = 0 #arrival of second pulse
	t1 = np.linspace(t2,-15,num=NAbs) #arrival of first pulse
		
	t = np.linspace(t3,15,num=NEmi) # arrival of signal pulse
	Et = np.exp(-(t*t)/(.3607*tau*tau))*np.cos(wl*t) # laser pulse for time t
	Et1 = np.exp(-t1*t1/(.3607*tau*tau))*np.cos(wl*t1) # laser pulse for time t1
	Et1S3 = np.exp(-t1S3*t1S3/(.3607*tau*tau))*np.cos(wl*t1S3)
	Ew = np.fft.fftshift(np.fft.fft(Et))
	Ew1 = np.fft.fftshift(np.fft.fft(Et1))
	Ew1S3 = np.fft.fftshift(np.fft.fft(Et1S3))
	testtimeend = time.time()
	r = np.zeros((NAbs,NEmi), dtype=complex) # initialized time domain response function
	rS3 = r
	for q in xrange(0,30):
		if q<15: n_LO=3
		else: n_LO = 4
		n_states = CdSe.numStates(model,n_LO)
		#############################
		# Define theano shared variables
		# site frequency matrix
		w = theano.shared(CdSe.buildstates(dE,dV,(Ex-Ec)/h,model,n_LO), name="w")
		# initial density matrix
		p0 = theano.shared(CdSe.buildp0(init,model,n_LO), name="p0")
		# dipole operator matrix
		# m = theano.shared(np.array([[0,0,0,0,j,k,0,0,0],
		# 						[0,0,0,0,k,j,k,0,0],
		# 						[0,0,0,0,0,k,j,0,0],
		# 						[0,0,0,0,0,0,k,0,0],
		# 						[j,k,0,0,0,0,0,j,k],
		# 						[k,j,k,0,0,0,0,k,j],
		# 						[0,k,j,k,0,0,0,0,k],
		# 						[0,0,0,0,j,k,0,0,0],
		# 						[0,0,0,0,k,j,k,0,0]]), name="m")
		m = theano.shared(CdSe.buildm(model,n_LO,n_coup), name="m")
		# homogenous dephasing operator
		# g = theano.shared(np.array([[0,G0,G0,G0,G1,G1,G1,G2,G2],
		# 						[G0,0,G0,G0,G1,G1,G1,G2,G2],
		# 						[G0,G0,0,G0,G1,G1,G1,G2,G2],
		# 						[G0,G0,G0,0,G1,G1,G1,G2,G2],
		# 						[G1,G1,G1,G1,0,G0,G0,G1,G1],
		# 						[G1,G1,G1,G1,G0,0,G0,G1,G1],
		# 						[G1,G1,G1,G1,G0,G0,0,G1,G1],
		# 						[G2,G2,G2,G2,G1,G1,G1,0,G0],
		# 						[G2,G2,G2,G2,G1,G1,G1,G0,0]]), name="g")
		g = theano.shared(CdSe.buildGmatrix(Gamma,model,n_LO), name="g")
		# inhomogenous dephasing function
		s = theano.shared(math.pow(sparam,2)/2)
			
		iscan = theano.shared(np.linspace(0,(NEmi-1),NEmi, dtype=int))

		#############################
		# Declare theano symbolic input variables
		x = TT.dscalar("x") # for t
		xvec = TT.dvector("xvec") # for t 
		y = TT.dscalar("y") # for t1
		h1 = TT.dscalar("h1") # for heaviside t2-t1
		h2 = TT.dscalar("h2") # for heaviside t3-t2
		h3vec = TT.dvector("h3vec")
		U1 = TT.cmatrix("U1")
		U2 = TT.cmatrix("U2")
		U3ten = TT.ctensor3("U3ten")
		conjU1 = TT.cmatrix("conjU1")
		conjU2 = TT.cmatrix("conjU2")
		conjU3ten = TT.ctensor3("conjU3ten")
		i = TT.ivector("i")

		#############################
		# Theano expression graph
		# # Explicit calculation of a given point in the 2D time domain response function
		# # Easier to follow the physics:
		# #dephasing at each step
		# deph1 = TT.exp(-g*(t2-y))
		# deph2 = TT.exp(-g*(t3-t2))
		# deph3 = TT.exp(-g*(x-t3))	
		# # r1 ket/ket/ket interactions
		#r1a = (TT.dot(U1,TT.dot(m,TT.dot(p0,conjU1))))*deph1
		#r1b = (TT.dot(U2,TT.dot(m,TT.dot(r1a,conjU2))))*deph2
		#r1c = (TT.dot(U3,TT.dot(m,TT.dot(r1b,conjU3))))*deph3
		#r1 = TTnlinalg.trace(TT.dot(m,r1c))
		# # r2 bra/ket/bra interactions
		#r2a = (TT.dot(U1,TT.dot(p0,TT.dot(m,conjU1))))*deph1
		#r2b = (TT.dot(U2,TT.dot(m,TT.dot(r2a,conjU2))))*deph2
		#r2c = (TT.dot(U3,TT.dot(r2b,TT.dot(m,conjU3))))*deph3
		#r2 = TTnlinalg.trace(TT.dot(m,r2c))			
		# # r3 bra/bra/ket interactions
		#r3a = (TT.dot(U1,TT.dot(p0,TT.dot(m,conjU1))))*deph1
		#r3b = (TT.dot(U2,TT.dot(r3a,TT.dot(m,conjU2))))*deph2
		#r3c = (TT.dot(U3,TT.dot(m,TT.dot(r3b,conjU3))))*deph3
		#r3 = TTnlinalg.trace(TT.dot(m,r3c))
		# # r4 ket/bra/bra interactions
		#r4a = (TT.dot(U1,TT.dot(m,TT.dot(p0,conjU1))))*deph1
		#r4b = (TT.dot(U2,TT.dot(r4a,TT.dot(m,conjU2))))*deph2
		#r4c = (TT.dot(U3,TT.dot(r4b,TT.dot(m,conjU3))))*deph3
		#r4 = TTnlinalg.trace(TT.dot(m,r4c))
		# # inhomogeneous broadening
		#r1 = r1*TT.exp(-s*(( x - t3 + t2 - y)**2))
		#r2 = r2*TT.exp(-s*(((x - t3) - (t2 - y))**2))
		#r3 = r3*TT.exp(-s*(( (x - t3) - (t2 - y))**2))
		#r4 = r4*TT.exp(-s*(( x - t3 + t2 - y)**2))
		#response = (1j*1j*1j)*h1*h2*h3*(r1+r2+r3+r4-TT.conj(r1)-TT.conj(r2)-TT.conj(r3)-TT.conj(r4))

		# Theano construction of the same calculation as above, runs 2-3X faster
		# Calculates 1 column of the response function
		# Internal function within the theano scan to eliminate redundant calculations
		def resfunc(i, xvec, y, h1, h2, h3vec, U1, U2, U3ten, conjU1, conjU2, conjU3ten):
			deph1 = TT.exp(-g*(t2-y))
			deph2 = TT.exp(-g*(t3-t2))
			deph3 = TT.exp(-g*(xvec[i]-t3))
			inhom14 = TT.exp(-s*(( xvec[i] - t3 + t2 - y)**2))
			inhom23 = TT.exp(-s*(((xvec[i] - t3) - (t2 - y))**2))
			r14a = (TT.dot(U1,TT.dot(m,TT.dot(p0,conjU1))))*deph1
			r23a = (TT.dot(U1,TT.dot(p0,TT.dot(m,conjU1))))*deph1
			r1 = TTnlinalg.trace(TT.dot(m,((TT.dot(U3ten[:,:,i],TT.dot(m,
				TT.dot(((TT.dot(U2,TT.dot(m,TT.dot(r14a,conjU2))))*deph2),conjU3ten[:,:,i]))))*deph3)))*inhom14
			r2 = (TTnlinalg.trace(TT.dot(m,((TT.dot(U3ten[:,:,i],TT.dot(((TT.dot(U2,
				TT.dot(m,TT.dot(r23a,conjU2))))*deph2),TT.dot(m,conjU3ten[:,:,i]))))*deph3))))*inhom23	
			r3 = (TTnlinalg.trace(TT.dot(m,((TT.dot(U3ten[:,:,i],TT.dot(m,
				TT.dot(((TT.dot(U2,TT.dot(r23a,TT.dot(m,conjU2))))*deph2),conjU3ten[:,:,i]))))*deph3))))*inhom23
			r4 = (TTnlinalg.trace(TT.dot(m,((TT.dot(U3ten[:,:,i],TT.dot(((TT.dot(U2,
				TT.dot(r14a,TT.dot(m,conjU2))))*deph2),TT.dot(m,conjU3ten[:,:,i]))))*deph3))))*inhom14
		   	return (1j*1j*1j)*h1*h2*h3vec[i]*(r1+r2+r3+r4-TT.conj(r1)-TT.conj(r2)-TT.conj(r3)-TT.conj(r4))
	
		response, updates = theano.scan(fn=resfunc,
									outputs_info=None, sequences=iscan, 
									non_sequences=[xvec,y,h1,h2,h3vec,U1,U2,U3ten,conjU1,conjU2,conjU3ten])

		# Theano time propagator expression
		timeprop = TTslinalg.expm(-1j*w*(x-y))

		#############################
		# Declare theano functions
		f_response = theano.function([xvec,y,h1,h2,h3vec,U1,U2,U3ten,conjU1,conjU2,conjU3ten], response, allow_input_downcast=True, on_unused_input='warn')
		f_timeprop = theano.function([x,y],timeprop)

		#############################
		# Calculate the time domain response function for single quantum excitation scans
		U3array = np.zeros((n_states,n_states,NEmi), dtype=complex)
		conjU3array = np.zeros((n_states,n_states,NEmi), dtype=complex)
		h3temparray = np.zeros(NEmi, dtype=int)

		for l in xrange(0,NEmi):
			temp = f_timeprop(t[l],t3)
			U3array[:,:,l] = temp
			conjU3array[:,:,l] = np.conj(temp)
			h3temparray[l] = CdSe.Heaviside(t[l]-t3)

		U2temp = f_timeprop(t3,t2)
		conjU2temp = np.conj(U2temp)

		h1S3 = CdSe.Heaviside(0)
		for k in xrange(0,NAbs):
			U1temp = f_timeprop(t2,t1[k])
			conjU1temp = np.conj(U1temp)
			h1temp = CdSe.Heaviside(t2-t1[k])
			h2temp = CdSe.Heaviside(t3-t2)
			r[k,:] = f_response(t,t1[k],h1temp,h2temp,h3temparray,U1temp,U2temp,U3array,conjU1temp,conjU2temp,conjU3array)

		# Normalize, Fourier transform to frequency domain, apply the laser fields
		r = 1j*(r/np.max(np.abs(r)))
		R = np.fft.fftshift(np.fft.fft2(r))
		for k in xrange(1,NAbs):
			R[k,:] = R[k,:]*Ew

		for k in xrange(1,NEmi):
			R[:,k] = R[:,k]*Ew1
	
		# Isolate the relevant 1 Quantum Scans
		RS1Out = np.flip(R[15:NAbs/2-15,NEmi/2+15:NEmi-16],axis=0)
		RS2Out = R[NAbs/2+15:NAbs-16,NEmi/2+15:NEmi-16]

		# Calculate the time domain response function for 2 quantum excitation scan
		for k in xrange(0,NAbs):
			U2tempS3 = f_timeprop(t3,t2S3[k])
			conjU2tempS3 = np.conj(U2tempS3)
			U1tempS3 = f_timeprop(t2S3[k],t1S3[k])
			conjU1tempS3 = np.conj(U1tempS3)
			h2tempS3 = CdSe.Heaviside(t3-t2S3[k])
			rS3[k,:] = f_response(t,t1S3[k],h1S3,h2tempS3,h3temparray,U1tempS3,U2tempS3,U3array,conjU1tempS3,conjU2tempS3,conjU3array)

		# Normalize, Fourier transform to frequency domain, apply the laser fields
		rS3 = 1j*(rS3/np.max(np.abs(rS3)))
		RS3 = np.fft.fftshift(np.fft.fft2(rS3))

		# multiply fields by response function
		for k in xrange(1,NAbs):
			RS3[k,:] = RS3[k,:]*Ew
	
		for k in xrange(1,NEmi):
			RS3[:,k] = RS3[:,k]*Ew1S3

		# Isolate the relevant 2 Quantum Scans
		RS3Out = RS3[NAbs/2+31:,NEmi/2+15:NEmi-16]

		# define frequency axes
		f1 = (pi*h)/(t1[1]-t1[0])*np.linspace(-1,1,num=NAbs)+Ec
		f1S3 = (pi*h)/(t1S3[1]-t1S3[0])*np.linspace(-1,1,num=NAbs)+2*Ec
		ft = -(pi*h)/(t[1]-t[0])*np.linspace(-1,1,num=NEmi)+Ec

# 		fig = plt.gcf()
# 		fig.set_size_inches(15,4)
# 		ax1 = fig.add_subplot(1,3,1)
# 		plt.imshow(np.absolute(RS1Out),extent=(ft[NEmi-16],ft[NEmi/2+15],f1[NAbs-16],f1[NAbs/2+15]),cmap='jet')
# 		ax1.plot([ft[NEmi-16],ft[NEmi/2+15]],[ft[NEmi-16],ft[NEmi/2+15]],color='w')
# 		plt.colorbar(ax=ax1)
# 		ax1.set_title('S1')
# 		ax1.set_xlabel('Emission Energy (meV)')
# 		ax1.set_ylabel('Absorption Energy (meV)')
# 		ax2 = fig.add_subplot(1,3,2)
# 		plt.imshow(np.absolute(RS2Out),extent=(ft[NEmi-16],ft[NEmi/2+15],f1[NAbs-16],f1[NAbs/2+15]),cmap='jet')
# 		ax2.plot([ft[NEmi-16],ft[NEmi/2+15]],[ft[NEmi-16],ft[NEmi/2+15]],color='w')
# 		plt.colorbar(ax=ax2)
# 		ax2.set_title('S2')
# 		ax2.set_xlabel('Emission Energy (meV)')
# 		ax2.set_ylabel('Absorption Energy (meV)')
# 		ax3 = fig.add_subplot(1,3,3)
# 		plt.imshow(np.absolute(RS3Out),extent=(ft[NEmi-16],ft[NEmi/2+15],f1S3[NAbs-1],f1S3[NAbs/2+30]),aspect=0.65,cmap='jet')
# 		ax3.plot([ft[NEmi-16],ft[NEmi/2+30]],[2*ft[NEmi-16],2*ft[NEmi/2+30]],color='w')
# 		plt.colorbar(ax=ax3)
# 		ax3.set_title('S3')
# 		ax3.set_xlabel('Emission Energy (meV)')
# 		ax3.set_ylabel('Two Quantum Energy (meV)')
# 		plt.show()

		# Save scans & metadata to hdf5 file
		with h5py.File(fileout,'a') as f:
			grp = f.create_group(('Sim'+str(q)))
			grp.attrs['dE'] = dE
			grp.attrs['dV'] = dV
			grp.attrs['model']=model
			grp.attrs['n_LO']=n_LO
			grp.attrs['n_coup']=n_coup
			grp.attrs['init']=init
			grp.attrs['Gamma']=Gamma
			grp.attrs['Ex']=Ex
			grp.attrs['sparam']=sparam
			dset = grp.create_dataset('S1',data = RS1Out, chunks = RS1Out.shape, compression = 'gzip', compression_opts=9)
			dset = grp.create_dataset('S2',data = RS2Out, chunks = RS2Out.shape, compression = 'gzip', compression_opts=9)
			dest = grp.create_dataset('S3',data = RS3Out, chunks = RS3Out.shape, compression = 'gzip', compression_opts=9)
