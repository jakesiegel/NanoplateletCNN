# script to make the csv
import numpy as np
import csv

first = True
fileout = 'CdSeList.csv'
count = 1
with open(fileout,'wb') as f:
	writer = csv.writer(f)
	for i in xrange(0,3):
		dE = 48+i*3
		for j in xrange(0,3):
			if j==0: scheme = 'single'			
			elif j==1: scheme = 'uncoupled'
			elif j==2: scheme = 'coupled'
			for k in xrange(0,3):
				n_LO = k+1
				for l in xrange(0,4):
					dV1 = 20.5+1.5*l
					repeat = False
					for m in xrange(0,4):
						dV2 = 20.5+1.5*m
						if j==0: 
							dV2=0
							if repeat == False:
								if first:
									list = [dE,scheme,dV1,dV2,n_LO,count,'next']
									first=False
								else:
									list = [dE,scheme,dV1,dV2,n_LO,count,'undone']
								print list
								writer.writerow(list)
								count+=1
								repeat = True
						else:
							if first:
								list = [dE,scheme,dV1,dV2,n_LO,count,'next']
								first=False
							else:
								list = [dE,scheme,dV1,dV2,n_LO,count,'undone']
							print list
							writer.writerow(list)
							count+=1
								

test = np.genfromtxt('CdSeList.csv',dtype=None,delimiter=",")
print len(test)
# needs to go dE, scheme, dV1, dV2, n_LO