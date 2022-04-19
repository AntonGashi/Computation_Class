import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import time 


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
    

    rms=np.sqrt(abs((1/points)*np.sum(sum**2)))
    
    standard_deviation=np.sqrt(abs(variance))

    sd_of_dist=standard_deviation/np.sqrt(points)

    return sum,answer,abs(variance),rms,standard_deviation,sd_of_dist

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
def circle(x,i):
    radius=1.5
    r=np.sqrt(x[0,i]**2+x[1,i]**2+x[2,i]**2+x[3,i]**2+x[4,i]**2)
    if r>=radius:
        return 0
    else:
        return 1

def circle_answer(range,points):
    sum,answer,variance,rms,standard_deviation,sd_of_dist=montecarlo_multi(circle,range,points)
    answer=(np.count_nonzero(sum)/points)*area
    return sum,answer,variance,rms,standard_deviation,sd_of_dist

@jit(nopython=True)
def task4(x,i):
    return 1/abs((x[0,i]*x[0,i])+x[0,i])

@jit(nopython=True)
def task5a(x):
    return 2*np.exp(-x**2)

def task5a_sample(x):
    return (1/2)*np.exp(-abs(x))

def task5b(x):
    return 1.5*np.sin(x)

def task5b_sample(x):
    return (3/(2*np.pi))*(4/np.pi**2)*x*(np.pi-x)


###
#Functions-end
###
def metropolis(func,N):
    delta=2.3
    x_i=0.1
    x_logged=np.zeros(N)
    trials=np.ones(N)

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
        else:
            trials[i]=0
        

    return x_logged,trials


def importance_sampling(input_function,input_sample_function,N):
    x_logged,trials=metropolis(input_sample_function,N)
    divided_funcs=input_function(x_logged)/input_sample_function(x_logged)
    answer=np.sum(divided_funcs)/N

    var1=(1/N)*np.average(np.sum(divided_funcs**2))
    var2=(1/N)*np.average(np.sum(divided_funcs))**2

    variance=(1/N)*(var1-var2)
    
    rms=np.sqrt(abs((1/N)*np.sum(divided_funcs**2)))
    
    standard_deviation=np.sqrt(abs(variance))

    sd_of_dist=standard_deviation/np.sqrt(N)
    
    confidence=1.645*(standard_deviation/N)

    print('Integral output (to a 90% confidence) = [{}] +- {} units.\nSample Size = [{}].\nTime Taken = [{}]s.\n----\nVariance = [{}].\nRoot-Mean-Square = [{}].\nStandard Deviation (Of Distribution/X-axis) = [{}]/[{}].'.format(format(answer,".4"),format(confidence,".3"),format(N,".0e"),format(5),format(variance,".3"),format(rms,".3"),format(sd_of_dist,".3"),format(standard_deviation,".3")))
    return x_logged,trials,answer,variance,rms,standard_deviation,sd_of_dist

start=time.perf_counter()
x_logged,trials,answer,variance,rms,standard_deviation,sd_of_dist=importance_sampling(task5b,task5b_sample,int(1e5))
stop=time.perf_counter()

print(format((stop-start),".3"))


###
#Task 2-4
###

def convergance_test(low,high,steps,answer,input_func):
    
    point_array=np.linspace(low,high,steps,dtype=int)
    params=[10,-10]
    results=np.zeros([6,steps])
    for i in range(steps):
        sum,calc_answer,variance,rms,standard_deviation,sd_of_dist=montecarlo_multi(input_func,params,(point_array[i]))
        #sum,calc_answer,variance,rms,standard_deviation,sd_of_dist=circle_answer(params,point_array[i])
        #x_logged,trials,trials_graph,calc_answer,variance,rms,standard_deviation,sd_of_dist=importance_sampling(task5a,task5a_sample,point_array[i])
        results[0,i]=calc_answer
        results[1,i]=variance
        results[2,i]=rms
        results[3,i]=standard_deviation
        results[4,i]=sd_of_dist
    
    yerr=results[4]
    plt.errorbar(point_array,results[0,:],yerr=yerr,color='r',fmt="k.",ecolor='k',capsize=2,label='Calculated Answer with Standard Deviation')
    plt.hlines(answer,low,high,color='tab:orange',label='Known Answer {}'.format(answer,".2"))
    plt.legend()
    plt.tight_layout()
    #plt.show()
    plt.savefig("task_6_a.png",dpi=300)
    pass


answer_array_task2=[2,-0.5,5.333333,0.75]
#convergance_test(1,1e6,100,3.5449,task5a)

'''
params=[[10,-10],int(1e5)]

start=time.perf_counter()
sum,answer,variance,rms,standard_deviation,sd_of_dist=montecarlo_multi(task5a,params[0],params[1])
###circle answers only
#sum,answer,variance,rms,standard_deviation,sd_of_dist=circle_answer(params[0],params[1])
###circle answers only
stop=time.perf_counter()
time_taken=stop-start

confidence=1.645*(standard_deviation/params[1])
print('Integral output (to a 90% confidence) = [{}] +- {} units.\nSample Size = [{}].\nTime Taken = [{}]s.\n----\nVariance = [{}].\nRoot-Mean-Square = [{}].\nStandard Deviation (Of Distribution/X-axis) = [{}]/[{}].'.format(format(answer,".4"),format(confidence,".3"),format(params[1],".0e"),format(time_taken,".3"),format(variance,".3"),format(rms,".3"),format(sd_of_dist,".3"),format(standard_deviation,".3")))

#plt.hist(sum,200)
#plt.vlines(standard_deviation,0,params[1]/10,color='r',linestyles='dashed',label='Standard Deviation')
#plt.legend()
#plt.show() 
'''
