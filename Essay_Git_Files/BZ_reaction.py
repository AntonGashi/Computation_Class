import numpy as np
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
from matplotlib import animation
from numba import jit
import time

# Width, height of the image.
nx, ny = 300,300
# Reaction parameters.


def update(p,arr,alpha,beta,gamma):
    """Update arr[p] to arr[q] by evolving in time."""
    global s
    # Count the average amount of each species in the 9 cells around each cell
    # by convolution with the 3x3 array m.
    q = (p+1) % 2
    s = np.zeros((3, ny,nx))
    m = np.ones((3,3)) / 9
    for k in range(3):
        s[k] = convolve2d(arr[p,k], m, mode='same', boundary='wrap')
    # Apply the reaction equations
    arr[q,0] = s[0] + s[0]*(alpha*s[1] - gamma*s[2])
    arr[q,1] = s[1] + s[1]*(beta*s[2] - alpha*s[0])
    arr[q,2] = s[2] + s[2]*(gamma*s[0] - beta*s[1])
    # Ensure the species concentrations are kept within [0,1].
    np.clip(arr[q], 0, 1, arr[q])
    return arr

# Initialize the array with random amounts of A, B and C.
arr = np.random.random(size=(2, 3, ny, nx))
# Set up the image
fig, ax = plt.subplots()
im = ax.imshow(arr[0,0], cmap=plt.cm.cool)
ax.axis('off')

def animate(i, arr):
    """Update the image for iteration i of the Matplotlib animation."""
    alpha,beta,gamma=1,1,1 
    arr = update(i % 2, arr,alpha,beta,gamma)
    im.set_array(arr[i % 2, 0])
    return [im]

anim = animation.FuncAnimation(fig, animate, frames=1000, interval=10, blit=False, fargs=(arr,))
#fig = plt.figure()
#ax = plt.axes(projection='3d')

def time_evol(N):
    
    x_array=np.zeros(N)
    y_array=np.zeros(N)
    z_array=np.zeros(N)

    alpha,beta,gamma=3,1,1   
    for i in range(N):
        x_array[i]=np.sum(update(i%2,arr,alpha,beta,gamma)[0,0])
        y_array[i]=np.sum(update(i%2,arr,alpha,beta,gamma)[0,1])
        z_array[i]=np.sum(update(i%2,arr,alpha,beta,gamma)[0,2])

    return x_array,y_array,z_array
N=300
'''
start=time.perf_counter()
x_array,y_array,z_array=time_evol(N)
stop=time.perf_counter()
print(stop-start)


fig, (ax1, ax2) = plt.subplots(1,2)

time_=np.linspace(0,N,N)
x_array_tau=np.roll(x_array,1)
#ax1.scatter(y_array,x_array,marker='x')
ax1.plot(x_array,y_array)
ax1.set_title('A')
ax1.set_xlabel('Amount of Chemical Y')
ax1.set_ylabel('Amount of Chemical X')

ax2.set_title('B')
ax2.plot(time_,x_array)
ax2.plot(time_,y_array)

ax2.set_ylabel('Amount of Each Chemical')
ax2.set_xlabel('Time')
ax2.set_xlim(0,500)

plt.tight_layout()
#ax2.plot(time_,z_array)

#plt.plot(x_array,y_array)
#plt.plot(y_array,z_array)
#plt.plot(z_array,x_array)
ax.plot3D(x_array, y_array, z_array, 'gray')
ax.set_xlabel('X data')
ax.set_ylabel('Y data')
ax.set_zlabel('Z data')
#ax.view_init(-140, -10)


#plt.show()
#plt.savefig('3d2.png',dpi=300)
# To view the animation, uncomment this line
anim.save(filename='bz_3_1_1.mp4', fps=30)
'''
