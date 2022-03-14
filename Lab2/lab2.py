import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time 

start=time.perf_counter()

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

    var1=(1/points)*np.average(np.sum(sum**2))
    var2=(1/points)*np.average(np.sum(sum))**2

    variance=(1/points)*(var1-var2)
    
    average=(1/points)*np.sum(sum)

    rms=np.sqrt(abs((1/points)*np.sum(sum**2)))
    
    standard_deviation=np.sqrt(abs(variance))

    sd_of_dist=standard_deviation/np.sqrt(points)

    return sum,answer,abs(variance),rms,average,standard_deviation,sd_of_dist

###
#Functions-start
###
### axis are denoted with indices, i.e. x[0,i]=x, x[1,i]=y, ... and so on.

@jit(nopython=True)
def func_a(x,i):
    return 2


@jit(nopython=True)
def func_b(x,i):
    return -x[0,i]


@jit(nopython=True)
def func_c(x,i):
    return x[0,i]**2


@jit(nopython=True)
def func_d(x,i):
    return (x[0,i]*x[1,i])+x[0,i]

@jit(nopython=True)
def func_e(x,i):
    return (x[0,i]*x[1,i]*x[2,i])+(x[1,i]*x[2,i])


@jit(nopython=True)
def circle(x,i):
    radius=1.5
    r=np.sqrt(x[0,i]**2+x[1,i]**2+x[2,i]**2+x[3,i]**2+x[4,i]**2)
    if r>=radius:
        return 0
    else:
        return 1

def circle_answer(params):
    sum,answer,variance,rms,average,standard_deviation,sd_of_dist=montecarlo_multi(circle,params[0],params[1])
    answer=(np.count_nonzero(sum)/params[1])*area
    return sum,answer,variance,rms,average,standard_deviation,sd_of_dist

@jit(nopython=True)
def task4(x,i):
    return 1/abs((x[0,i]*x[0,i])+x[0,i])

@jit(nopython=True)
def task5a(x):
    return 2*np.exp(-x**2)

def task5a_sample(x):
    return np.exp(-abs(x))

def task5b(x,i):
    return 1.5*np.sin(x)

def task5b_sample(x):
    return (4/np.pi**2)*x*(np.pi-x)

def metropolis(func,N):
    delta=0.1
    x_i=5
    x_logged=np.zeros(N)
    for i in range(N):
        rand_delta=np.random.uniform(-delta,delta,1)
        x_trial=x_i+rand_delta
        w=func(x_trial)/func(x_i)
        x_logged[i]=x_i
        if w>=1:
            x_i=x_trial
            x_logged[i]=x_trial
        elif w>=np.random.ranf(1):
            x_i=x_trial
            x_logged[i]=x_trial
    return x_logged

start=time.perf_counter()
x_logged=metropolis(task5a,int(1e5))
stop=time.perf_counter()
print(x_logged,format((stop-start),".2"))
plt.hist(x_logged)
plt.show()


###
#Functions-end
###
###
#for 1-D integrals
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
###
#1-D integrals end 
###
###
#Task 2-4
###
'''
params=[[1,0,1,0,1,0],int(1e4)]

sum,answer,variance,rms,average,standard_deviation,sd_of_dist=montecarlo_multi(task4,params[0],params[1])
###circle answers only
#sum,answer,variance,rms,average,standard_deviation,sd_of_dist=circle_answer(params)
###circle answers only
stop=time.perf_counter()
time_taken=stop-start

confidence=1.645*(standard_deviation/params[1])
print('Integral output (to a 90% confidence) = [{}] +- {} units.\nSample Size = [{}].\nTime Taken = [{}]s.\n----\nVariance = [{}].\nRoot-Mean-Square = [{}].\nAverage = [{}].\nStandard Deviation (Of Distrobution/X-axis) = [{}]/[{}].'.format(format(answer,".3"),format(confidence,".3"),format(params[1]),format(time_taken,".3"),format(variance,".3"),format(rms,".3"),format(average,".3"),format(sd_of_dist,".3"),format(standard_deviation,".3")))

plt.hist(sum,200)
plt.vlines(standard_deviation,0,params[1]/10,color='r',linestyles='dashed',label='Standard Deviation')
plt.legend()
plt.show()
'''
###
#Task2-4 end 
###
###
#Task 5-6
###



###
#Task 5-6 end 
###
