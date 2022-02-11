import numpy as np
import matplotlib.pyplot as plt
from numba import jit

def Task1(seed,N,M):
    np.random.seed(seed)
    ###Task1

    E=N/M
    start_stop_length=[0,1,N]

    uni_rand=np.random.uniform(start_stop_length[0],start_stop_length[1],start_stop_length[2])
    random_choice=np.random.choice(uni_rand,start_stop_length[2],replace=True) # if replacement is not allowed, 'False', then the chi squared of both random sequences becomes equal.

    hist1=np.histogram(uni_rand,M)
    hist2=np.histogram(random_choice,M)
    x1=0
    x2=0

    for i in hist1[0]:
        x1+=(i-E)**2/E
    for i in hist2[0]:
        x2+=(i-E)**2/E

    correl=np.round(np.corrcoef(random_choice,uni_rand),3)
    correl_perc=np.diag(correl,1)*100
    print('Correlation between the two arrays of random numbers with the np.corrcoef function: {} \n With a {}% Pearson product-moment correlation'.format(correl,correl_perc))
    print('The Chi squared for the uniform distribution is {} and for the random choice its {}'.format(np.round(x1,3),np.round(x2,3)))

    shifted_array=np.roll(uni_rand,10)
    plt.rcParams['text.usetex'] = True
    plt.scatter(uni_rand,shifted_array,marker='x')
    plt.title('Scatter plot of Uniform random distribution $x_n$ against $x_{n+10}$ for 10000 points')
    plt.xlabel('Uniform distribution')
    plt.ylabel('Uniform distribution + 10')
    #plt.hlines(0,0,1,'r',':')
    #plt.hlines(1,0,1,'r',':')
    #plt.vlines(0,0,1,'r',':')
    #plt.vlines(1,0,1,'r',':')
    plt.show()

#Task1(10,1000,100)

@jit
def Task2(seed,N,time_steps):
    np.random.seed(seed)
    ###Task2

    left_side=np.ones(N,dtype=int)
    right_side=np.zeros(N,dtype=int)
    left_sum=np.zeros(time_steps)
    right_sum=np.zeros(time_steps)
    plot_time=np.linspace(0,time_steps,time_steps)
    random_int_array=np.zeros(time_steps)
    dndx=np.zeros(time_steps)

    for i in range(time_steps):
        left_sum[i]=np.sum(left_side)
        right_sum[i]=N-left_sum[i]
        rand_index=np.random.randint(0,N)
        random_int_array[i]=rand_index
        dndx[i]=left_sum[i]-right_sum[i]

        if left_side[rand_index]==1:
            right_side[rand_index]=1
            left_side[rand_index]=0
        elif right_side[rand_index]==1:
            left_side[rand_index]=1
            right_side[rand_index]=0

    #plt.plot(plot_time,left_sum,label='Left Side of Box')
    #plt.plot(plot_time,right_sum,label='Right Side of Box')
    #plt.title('Dispersion of {} Particles in a Box'.format(N))
    #plt.xlabel('Time Steps')
    #plt.ylabel('Particles on the Left and Right Side of the Box')
    #plt.hlines(N/2,0,time_steps,'r',linestyles=':',label='Half of All Particles')
    #plt.vlines(500,0,300,color='k',linestyles=':',label='Equilibrium point')
    #plt.legend()

    #shifted_array=np.roll(random_int_array,10)
    #correl=np.round(np.corrcoef(random_int_array,shifted_array),3)
    #print(correl)
    #plt.scatter(random_int_array,shifted_array,marker='x')
    #plt.title('Scatter plot of a Random Integer distribution $x_n$ against $x_n+10$ for {} points. \n Correlation between the two arrays of random numbers with the np.corrcoef function: {}'.format(time_steps,np.diag(correl,1)))
    #plt.xlabel('Random Integer distribution')
    #plt.ylabel('Random Integer distribution + 10')
    #plt.tight_layout()
    #plt.show()

#Task2(1,70000,int(1e6))

@jit
def Task4(seed,N,time_steps):
    np.random.seed(seed)
    ###Task4

    left_side,right_side=np.ones(N,dtype=int),np.zeros(N,dtype=int) #initialising the left side full of particles and the right side that's empty
    left_sum,right_sum=np.zeros(time_steps),np.zeros(time_steps) #initialising the left and right side sums
    plot_time=np.linspace(0,time_steps,time_steps) #defining the time array for plotting

    for i in range(time_steps):
        side_pick=np.random.randint(1,5) #generates int 1-4
        rand_index=np.random.randint(0,N) #index's the array of particles
        left_sum[i]=np.sum(left_side) #sum the left side of array
        right_sum[i]=N-left_sum[i]
        if side_pick==1: #conditional statement choosing a particle to switch
            left_side[rand_index]=0
            right_side[rand_index]=1
        else:
            left_side[rand_index]=1
            right_side[rand_index]=0
    #plotting the data collected
    plt.plot(plot_time,left_sum,label='Left Side of Box')
    plt.plot(plot_time,right_sum,label='Right Side of Box')
    plt.title('Dispersion of {} Particles in a Box With a Barrier Only Allowing 25% Through'.format(N))
    plt.xlabel('Time Steps')
    plt.ylabel('Particles on the Left and Right Side of the Box')
    plt.hlines(N*0.75,0,time_steps,'r',linestyles=':',label='Three Quarters of All Particles')
    plt.hlines(N*0.25,0,time_steps,'k',linestyles=':',label='One Quarters of All Particles')
    plt.legend(loc='center left')
    plt.show()

Task4(1,1000,10000)
