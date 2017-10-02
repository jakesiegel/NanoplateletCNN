# module for building dipole moment and Dephasing matrices for CdSeTheano Simulations
import numpy as np
import csv
import shutil
from tempfile import NamedTemporaryFile

h = 4.135 #meV*ps

def buildstates(dE, dV, a0, phonon_scheme, n_LO):
	if isinstance(dV, list):
		phS = dV[0]/h
		phL = dV[1]/h
	else:
		ph = dV/h
	
	bx = dE/h
	b0 = 2*a0-dE
	if (phonon_scheme=='single')&(n_LO==3):
		res = np.diagflat([0,ph, 2*ph, 3*ph, a0, a0+ph, a0+2*ph, a0+3*ph, b0, b0+ph, b0+2*ph, b0+3*ph])
	elif (phonon_scheme=='single')&(n_LO==4):
		res = np.diagflat([0,ph, 2*ph, 3*ph, 4*ph, a0, a0+ph, a0+2*ph, a0+3*ph, a0+4*ph,
					b0, b0+ph, b0+2*ph, b0+3*ph, b0+4*ph])
	elif (phonon_scheme=='uncoupled')&(n_LO==3):
		res = np.diagflat([0, phS, phL, 2*phL, 3*phL, a0, a0+phS, a0+phL, a0+2*phL, a0+3*phL,
					b0, b0+phS, b0+phL, b0+2*phL, b0+3*phL])
	elif (phonon_scheme=='uncoupled')&(n_LO==4):
		res = np.diagflat([0, phS, phL, 2*phL, 3*phL, 4*phL, a0, a0+phS, a0+phL, a0+2*phL, 
					a0+3*phL, a0+4*phL, b0, b0+phS, b0+phL, b0+2*phL, b0+3*phL, b0+4*phL])
	elif (phonon_scheme=='coupled')&(n_LO==3):
		res = np.diagflat([0, phS, phL, phS+phL, 2*phL, phS+2*phL, 3*phL, phS+3*phL,
					a0, a0+phS, a0+phL, a0+phS+phL, a0+2*phL, a0+phS+2*phL, a0+3*phL, a0+phS+3*phL,
					b0, b0+phS, b0+phL, b0+phS+phL, b0+2*phL, b0+phS+2*phL, b0+3*phL, b0+phS+3*phL])
	elif (phonon_scheme=='coupled')&(n_LO==4):
		res = np.diagflat([0, phS, phL, phS+phL, 2*phL, phS+2*phL, 3*phL, phS+3*phL,
					4*phL, phS+4*phL, a0, a0+phS, a0+phL, a0+phS+phL, a0+2*phL,
					a0+phS+2*phL, a0+3*phL, a0+phS+3*phL, a0+4*phL, a0+phS+4*phL,
					b0, b0+phS, b0+phL, b0+phS+phL, b0+2*phL, b0+phS+2*phL, b0+3*phL,
					b0+phS+3*phL, b0+4*phL, b0+phS+4*phL])	
	return res
	
def buildm(phonon_scheme, n_LO, n_coup):
	dSL = 0.2 # local variable for how close different types of dipoles with = phonon number must be
	# if a +SO+2LO can be totally independent, change e.g. l2 +=...(-dSL,dSL) to l2=...(0.5,1.1)
	j = np.random.uniform(0.7, 1.3)
	o=o1=o2=o3=q1=q2=0. # these are the dipoles for 4 & 5 transitions, currently allow up to 3
	if (phonon_scheme=='single')&(n_LO==3):
		zer = np.zeros((4,4),dtype=float)
		k = np.random.uniform(0.6, 1.2)
		l=n=0.
		if n_coup>1:
			l = np.random.uniform(0.5,1.1)
			if n_coup>2:
				n = np.random.uniform(0.4, 1)
		mSin = np.array([[j,k,l,n],[k,j,k,l],[l,k,j,k],[n,l,k,j]])
	elif (phonon_scheme=='single')&(n_LO==4):
		zer = np.zeros((5,5),dtype=float)
		k = np.random.uniform(0.6, 1.2)
		l=n=0.
		if n_coup>1:
			l = np.random.uniform(0.5,1.1)
			if n_coup>2:
				n = np.random.uniform(0.4, 1)
		mSin = np.array([[j,k,l,n,o],[k,j,k,l,n],[l,k,j,k,l],[n,l,k,j,k],[o,n,l,k,j]])
	elif (phonon_scheme=='uncoupled')&(n_LO==3):
		zer = np.zeros((5,5),dtype=float)	
		k1 = k2 = np.random.uniform(0.6, 1.2)
		k2 += np.random.uniform(-dSL, dSL)# COMMENT OUT if all 1 phonon dipoles are equal
		l1 = l2 = n1= n2 = 0.
		if n_coup>1:
			l1 = l2 = np.random.uniform(0.5,1.1)
			l2 +=np.random.uniform(-dSL,dSL) # COMMENT OUT if all 2 phonon dipoles are equal
			if n_coup>2:
				n1 = n2 = np.random.uniform(0.4, 1)
				n2 +=np.random.uniform(-dSL,dSL)# COMMENT OUT if all 3 phonon dipoles are equal
		mSin = np.array([[j,k1,k2,l1,n1],[k1,j,l2,n2,o1],[k2,l2,j,k2,l1],
					[l1,n2,k2,j,k2],[n1,o1,l1,k2,j]])
	elif (phonon_scheme=='uncoupled')&(n_LO==4):	
		zer = np.zeros((6,6),dtype=float)	
		k1 = k2 = np.random.uniform(0.6, 1.2)
		k2 += np.random.uniform(-dSL, dSL)# COMMENT OUT if all 1 phonon dipoles are equal
		l1 = l2 = n1 = n2 = o1 = o2 = q1 = 0.
		if n_coup>1:
			l1 = l2 = np.random.uniform(0.5,1.1)
			l2 +=np.random.uniform(-dSL,dSL) # COMMENT OUT if all 2 phonon dipoles are equal
			if n_coup>2:
				n1 = n2 = np.random.uniform(0.4, 1)
				n2 +=np.random.uniform(-dSL,dSL)# COMMENT OUT if all 3 phonon dipoles are equal		
		mSin = np.array([[j,k1,k2,l1,n1,o2],[k1,j,l2,n2,o1,q1],[k2,l2,j,k2,l1,n1],
				[l1,n2,k2,j,k2, l1],[n1,o2,l1,k2,j,k2],[o1,q1,n1,l1,k2,j]])
	elif (phonon_scheme=='coupled')&(n_LO==3):
		zer = np.zeros((8,8),dtype=float)	
		k1 = k2 = np.random.uniform(0.6, 1.2)
		k2 += np.random.uniform(-dSL, dSL)# COMMENT OUT if all 1 phonon dipoles are equal
		l1 = l2 = l3 = n1 = n2 = n3 = o1 = o3 = 0.
		if n_coup>1:
			l1 = l2 = l3 = np.random.uniform(0.5,1.1)
			l2 +=np.random.uniform(-dSL,dSL) # COMMENT OUT if all 2 phonon dipoles are equal
			l3 +=np.random.uniform(-dSL,dSL) # COMMENT OUT if all 2 phonon dipoles are equal
			if n_coup>2:
				n1 = n2 = n3 = np.random.uniform(0.4, 1)
				n2 +=np.random.uniform(-dSL,dSL)# COMMENT OUT if all 3 phonon dipoles are equal
				n3 +=np.random.uniform(-dSL,dSL)# COMMENT OUT if all 3 phonon dipoles are equal
		mSin = np.array([[j, k1, k2, l2, l1, n2, n1, o3],[k1, j, l3, k2, n3, l1, o1, n1],
				[k2, l3, j, k1, k2, l2, l1, n2],[l2, k2, k1, j, l3, k2, n3, l1],
				[l1, n3, k2, l3, j, k1, k2, l2],[n2, l1, l2, k2, k1, j, l3, k2],
				[n1, o1, l1, n3, k2, l3, j, k1],[o3, n1, n2, l1, l2, k2, k1, j]])
	elif (phonon_scheme=='coupled')&(n_LO==4):			
		zer = np.zeros((10,10),dtype=float)	
		k1 = k2 = np.random.uniform(0.6, 1.2)
		k2 += np.random.uniform(-dSL, dSL)# COMMENT OUT if all 1 phonon dipoles are equal
		l1 = l2 = l3 = n1 = n2 = n3 = o1 = o2 = o3 = q1 = q2 = 0.
		if n_coup>1:
			l1 = l2 = l3 = np.random.uniform(0.5,1.1)
			l2 +=np.random.uniform(-dSL,dSL) # COMMENT OUT if all 2 phonon dipoles are equal
			l3 +=np.random.uniform(-dSL,dSL) # COMMENT OUT if all 2 phonon dipoles are equal
			if n_coup>2:
				n1 = n2 = n3 = np.random.uniform(0.4, 1)
				n2 +=np.random.uniform(-dSL,dSL)# COMMENT OUT if all 3 phonon dipoles are equal
				n3 +=np.random.uniform(-dSL,dSL)# COMMENT OUT if all 3 phonon dipoles are equal	
		mSin = np.array([[j, k1, k2, l2, l1, n2, n1, o3, o2, q2],[k1, j, l3, k2, n3, l1, o1, n1, q1, o2],
				[k2, l3, j, k1, k2, l2, l1, n2, n1, o3],[l2, k2, k1, j, l3, k2, n3, l1, o1, n1],
				[l1, n3, k2, l3, j, k1, k2, l2, l1, n2],[n2, l1, l2, k2, k1, j, l3, k2, n3, l1],
				[n1, o1, l1, n3, k2, l3, j, k1, k2, l2],[o3, n1, n2, l1, l2, k2, k1, j, l1, k2],
				[o2, q1, n1, o1, l1, n3, k2, l3, j, k1],[q2, o2, o3, n1, n2, l1, l2, k2, k1, j]])
	# k1: +S, k2: +L, l1: +2L, l2: -S +L, l3: -S +L, 
	# n1: +3L, n2: -S +2L, n3: -S +2L, o1: -S +3L, o2: +4L, o3: -S +3L, q1: -S +4L q2: +S +4L		
	mout = np.concatenate([np.concatenate([zer,mSin,zer],axis=1),np.concatenate([
					mSin,zer,mSin],axis=1),np.concatenate([zer,mSin,zer],axis=1)],axis=0)
	return mout
			
def buildGmatrix(Gamma, phonon_scheme, n_LO):
	# it will similar to above where for each condition you make a G0 matrix, a G1 and G2 and then concatenate them
	G0 = 0.5/Gamma
	G1 = 0.9/Gamma
	G2 = 0.9/Gamma
	if (phonon_scheme=='single')&(n_LO==3):
		n_sub = 4
	elif (phonon_scheme=='single')&(n_LO==4):
		n_sub = 5
	elif (phonon_scheme=='uncoupled')&(n_LO==3):
		n_sub = 5
	elif (phonon_scheme=='uncoupled')&(n_LO==4):
		n_sub = 6
	elif (phonon_scheme=='coupled')&(n_LO==3):
		n_sub = 8
	elif (phonon_scheme=='coupled')&(n_LO==4):
		n_sub = 10	
	
	popC = G0*(np.ones([n_sub,n_sub],dtype=int)-np.diagflat(np.ones(n_sub,dtype=int)))
	ex1C = G1*np.ones([n_sub,n_sub],dtype=int)
	ex2C = G2*np.ones([n_sub,n_sub],dtype=int)
	GamOut = np.concatenate([np.concatenate([popC,ex1C,ex2C],axis=1),np.concatenate([
					ex1C,popC,ex1C],axis=1),np.concatenate([ex2C,ex1C,popC],axis=1)],axis=0)
	return GamOut

def buildp0(init,phonon_scheme,n_LO):
	if (phonon_scheme=='single')&(n_LO==3):
		l = 12
	elif (phonon_scheme=='single')&(n_LO==4):
		l = 15
	elif (phonon_scheme=='uncoupled')&(n_LO==3):
		l = 15
	elif (phonon_scheme=='uncoupled')&(n_LO==4):
		l = 18
	elif (phonon_scheme=='coupled')&(n_LO==3):
		l = 24
	elif (phonon_scheme=='coupled')&(n_LO==4):
		l = 30
	
	return np.diagflat(np.concatenate([init,np.zeros((l-len(init)),dtype=np.int)]))

def numStates(phonon_scheme,n_LO):
	if (phonon_scheme=='single')&(n_LO==3):
		l = 12
	elif (phonon_scheme=='single')&(n_LO==4):
		l = 15
	elif (phonon_scheme=='uncoupled')&(n_LO==3):
		l = 15
	elif (phonon_scheme=='uncoupled')&(n_LO==4):
		l = 18
	elif (phonon_scheme=='coupled')&(n_LO==3):
		l = 24
	elif (phonon_scheme=='coupled')&(n_LO==4):
		l = 30
	
	return l

	
# Heaviside step function
def Heaviside(arg1):
	if np.size(arg1)==1:
		if arg1<0:
			step = 0;
		else:
			step = 1;
	
	else:
		temp = np.where(arg1<0)
		if np.size(temp[0])==0:
			step = np.ones(len(arg1))
		else:
			step = np.concatenate([np.zeros(temp[0][-1]),np.ones(len(arg1)-temp[0][-1])])
			
	return step;

def getParamsFromFile(filename):
	tempfile = NamedTemporaryFile(delete=False)
	simparams = np.genfromtxt(filename,dtype=None,delimiter=",")
	for i in range(0,len(simparams)-1):
		if simparams[i][-1]=='next':
			paramsout = (simparams[i][0],simparams[i][1],simparams[i][2],simparams[i][3],simparams[i][4],simparams[i][5])
			simparams[i][-1]='done'
			if i<(len(simparams)-1):
				simparams[i+1][-1]='next'
				break
			
	with open(filename) as f:
		writer = csv.writer(tempfile)
		for row in simparams:
			writer.writerow(row)
		
	shutil.move(tempfile.name,filename)
	return paramsout

def getParams(filename):
	simparams = np.genfromtxt(filename,dtype=None,delimiter=",")
	for i in range(0,len(simparams)-1):
		if simparams[i][-1]=='next':
			paramsout = (simparams[i][0],simparams[i][1],simparams[i][2],simparams[i][3],simparams[i][4],simparams[i][5])
			simparams[i][-1]='done'
			if i<(len(simparams)-1):
				simparams[i+1][-1]='next'
				break
			
	with open(filename) as f:
		writer = csv.writer(tempfile)
		for row in simparams:
			writer.writerow(row)
		
	shutil.move(tempfile.name,filename)
	return paramsout


# to test whether it's single or double phonon use: isinstance(dV,list)