import numpy as np
import numpy.random as nr
from math import log
import mixed 

def main():
	n = 1000
	cov = [[1,0.9],[0.9,1]]
	beta = 0.9
	p_con, p_dis = 0.5, 0.5
	gt = -p_con*0.5*log(np.linalg.det(cov))+p_dis*(log(2)+beta*log(beta)+(1-beta)*log(1-beta))-p_con*log(p_con)-p_dis*log(p_dis)

	x_con,y_con = nr.multivariate_normal([0,0],cov,int(n*p_con)).T
	x_dis = nr.binomial(1,0.5,int(n*p_dis))
	y_dis = (x_dis + nr.binomial(1,1-beta,int(n*p_dis)))%2
	x_dis, y_dis = 2*x_dis-np.ones(int(n*p_dis)), 2*y_dis-np.ones(int(n*p_dis))
	x = np.concatenate((x_con,x_dis)).reshape((n,1))
	y = np.concatenate((y_con,y_dis)).reshape((n,1))

	print "Ground Truth = ", gt
	print "Mixed KSG: I(X:Y) = ", mixed.Mixed_KSG(x,y)
	print "Partitioning: I(X:Y) = ", mixed.Partitioning(x,y)
	print "Noisy KSG: I(X:Y) = ", mixed.Noisy_KSG(x,y)
	print "KSG: I(X:Y) = ", mixed.KSG(x,y)

if __name__ == '__main__':
	main()
	
