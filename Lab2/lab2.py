import numpy as np

def montecarlo(func,array_of_limits,points):
    if (len(array_of_limits)/2)==1:

        X=np.random.uniform(array_of_limits[0],array_of_limits[1],points)
        constant=(array_of_limits[1]-array_of_limits[0])/points
        sum=np.zeros(points)

        for i in range(points):
            sum[i]=func(X[i])

    elif (len(array_of_limits)/2)==2:

        X=np.random.uniform(array_of_limits[0],array_of_limits[1],points)
        Y=np.random.uniform(array_of_limits[2],array_of_limits[3],points)
        constant=(array_of_limits[1]-array_of_limits[0])*(array_of_limits[3]-array_of_limits[2])/points
        sum=np.zeros((points,2))

        for j in range(2):
            for i in range(points):
                sum[i,j]=func(X[i],Y[j])
    
    return constant*np.sum(sum)


def a(x):
    return 2


def b(x):
    return -x


def c(x):
    return x**2


def d(x,y):
    return (x*y)+x


print(montecarlo(d,[0,1,0,1],10000))
