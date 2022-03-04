import numpy as np
from numba import jit
import matplotlib.pyplot as plt


###only for the 1-D integral###
def montecarlo_single(func,array_of_limits,points):

    X=np.random.uniform(array_of_limits[0],array_of_limits[1],points)
    constant=(array_of_limits[1]-array_of_limits[0])/points
    sum=np.zeros(points)

    for i in range(points):
        sum[i]=func(X[i])
   
    answer=constant*np.sum(sum)
    answer_squared=constant*np.sum(sum**2)
    variance=(1/points)*(answer_squared-(answer**2))
    ### do root-mean-squared at some point
    return X,answer,np.sqrt(abs(variance))


def scatter_points(func,limits,points,montecarlo_randomvals):
    
    x_vals=np.linspace(limits[0],limits[1],points)
    set=func(x_vals)
    rand_set=func(montecarlo_randomvals)
    upper=np.empty([2,1])
    lower=np.empty([2,1])

    for i in range(len(x_vals)):
        if rand_set[i]>=set[i]:
            upper=np.append(upper,np.array([[rand_set[i]],[x_vals[i]]]),1)
  
        else:
            lower=np.append(lower,np.array([[rand_set[i]],[x_vals[i]]]),1)

    return upper,lower
###---###
###---###

def montecarlo_multi(func,array_of_limits,points):
    global area
    dimension=int(len(array_of_limits)/2)
    X=np.zeros((dimension,points))
    const=0
    sum=np.zeros(int(points))
    area=1

    for i in range(dimension):
        X[i,:]=np.random.uniform(array_of_limits[2*i],array_of_limits[2*i+1],points)
        const+=(array_of_limits[2*i]-array_of_limits[(2*i)+1])
        area=area*abs(array_of_limits[2*i]-array_of_limits[(2*i)+1])

    for j in range(points):
        sum[j]=func(X,j)

    const=const/(points*dimension)

    answer=(const*np.sum(sum))

    var1=np.sum(sum**2)
    var2=np.sum(sum)**2

    variance=(1/points)*(var1-var2)
    
    average=(1/points)*np.sum(sum)

    rms=np.sqrt(abs((1/points)*np.sum(sum**2)))

    return sum,answer,variance,rms,average,np.sqrt(abs(variance))

###
#Functions-start
###
### axis are denoted with indices, i.e. x[0,i]=x, x[1,i]=y, ... and so on.
def func_a(x,i):
    return 2


def func_b(x,i):
    return -x[0,i]


def func_c(x,i):
    return x[0,i]**2


def func_d(x,i):
    return (x[0,i]*x[1,i])+x[0,i]


def func_e(x,i):
    return (x[0,i]*x[1,i]*x[2,i])+(x[1,i]*x[2,i])


def circle(x,i):
    radius=1.5
    r=np.sqrt(x[0,i]**2+x[1,i]**2+x[2,i]**2+x[3,i]**2+x[4,i]**2)
    if r>=radius:
        return 0
    else:
        return 1

def circle_answer(params):
    sum,answer,variance,rms,average,standard_deviation=montecarlo_multi(circle,params[0],params[1])
    answer=(np.count_nonzero(sum)/params[1])*area
    return answer,variance,rms,average,standard_deviation
###
#Functions-end
###

'''
x_rand_vals,answer,variance=montecarlo_single(func_c,params[0],params[1])
print(np.round(answer,2),np.round(variance,5))
upper,lower=scatter_points(func_c,params[0],params[1],x_rand_vals)
x_vals=np.linspace(params[0][0],params[0][1],params[1])

plt.scatter(upper[1],upper[0],marker='+',color='orange',alpha=0.3)
plt.scatter(lower[1],lower[0],marker='+',alpha=0.3,label='Area under the graph = {} $units^2$'.format(np.round(answer,1)))
plt.plot(x_vals,func_c(x_vals),'r')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Monte-Carlo Integration of a Function')
plt.legend()
plt.tight_layout()
#plt.ylim(-2,6)
plt.show()
'''

params=[[2,-2,2,-2,2,-2,2,-2,2,-2],int(1e6)]

#sum,answer,variance,rms,average,standard_deviation=montecarlo_multi(func_e,params[0],params[1])
###circle answers only
answer,variance,rms,average,standard_deviation=circle_answer(params)
###circle answers only
print('Integral output = [{}] units.\nSample Size = [{}].\n----\nVariance = [{}].\nRoot-Mean-Square = [{}].\nAverage = [{}].\nStandard Deviation = [{}].'.format(format(answer,".3"),format(params[1]),format(variance,".3"),format(rms,".3"),format(average,".3"),format(standard_deviation,".3")))
