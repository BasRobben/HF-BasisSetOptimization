import numpy as np
import matplotlib.pyplot as plt

def main():
    print('Hello world :))')
    print("Bobber kurwa")

    x = np.linspace(0,1,1000)
    y = np.sin(2*np.pi*x)
    plt.plot(x,y)
    plt.show()

    return 

if __name__ == '__main__':
    main()
