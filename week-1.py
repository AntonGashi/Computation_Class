from cProfile import label
from operator import le
import numpy as np
import matplotlib.pyplot as plt

def Task1(seed,N):
    np.random.seed(seed)
    ###Task1

    start_stop_length=[0,1,N]
    uni_rand=np.random.uniform(start_stop_length[0],start_stop_length[1],start_stop_length[2])
    random_choice=np.random.choice(uni_rand,start_stop_length[2])
    x_sq_1=np.zeros(start_stop_length[2])
    x_sq_2=np.zeros(start_stop_length[2])


    for i in range(start_stop_length[2]):
        x_sq_1[i]=(random_choice[i]-1)**2
        x_sq_2[i]=(uni_rand[i]-1)**2

    print('Correlation between the two arrays of random numbers with the np.correlate function: {}'.format(np.correlate(random_choice,uni_rand)))
    
    plt.scatter(random_choice,x_sq_1)
    plt.scatter(uni_rand,x_sq_2)
    plt.show()

#Task1(1,2000)

def Task2(seed,N,time_steps):
    np.random.seed(seed)
    ###Task2

    left_side,right_side=np.ones(N,dtype=int),np.zeros(N,dtype=int)
    left_sum,right_sum=np.zeros(time_steps),np.zeros(time_steps)
    plot_time=np.linspace(0,time_steps,time_steps)


    for i in range(time_steps):
        left_sum[i]=np.sum(left_side)
        right_sum[i]=N-left_sum[i]
        rand_index=np.random.randint(0,N)
        if left_side[rand_index]==1:
            right_side[rand_index]=1
            left_side[rand_index]=0
        elif right_side[rand_index]==1:
            left_side[rand_index]=1
            right_side[rand_index]=0


    plt.plot(plot_time,left_sum,label='Left Side of Box')
    plt.plot(plot_time,right_sum,label='Right Side of Box')
    plt.xlabel('Time Steps')
    plt.ylabel('Particles on the Left and Right Side of the Box')
    plt.hlines(N/2,0,time_steps,'r',linestyles=':',label='Half of All Particles')
    plt.ylim((80,310))
    plt.legend()
    plt.show()

#Task2(1,300,5000)

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
    plt.xlabel('Time Steps')
    plt.ylabel('Particles on the Left and Right Side of the Box')
    plt.hlines(N*0.75,0,time_steps,'r',linestyles=':',label='Three Quarters of All Particles')
    plt.hlines(N*0.25,0,time_steps,'k',linestyles=':',label='One Quarters of All Particles')
    plt.legend(loc='center left')
    plt.show()

Task4(1,300,5000)