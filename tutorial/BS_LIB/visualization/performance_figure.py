import numpy as np 
import matplotlib.pyplot as plt

# Tutorial
'''
x = np.linspace(-1, 1, 50)
y = 2 * x + 1

plt.figure()
plt.plot(x, y)
plt.show()
'''

x = np.linspace(-3, 3, 50)
y1 = 2*x + 1
y2 = x**2

plt.figure()
plt.plot(x, y1)
plt.show()

plt.figure(num=3, figsize=(8, 5),)
plt.plot(x, y2)
plt.plot(x, y1, color='red', linewidth=1.0, linestyle='--')
plt.show()



'''
filepath = ""
file = open(filepath, "r")
f1 = file.readlines()
for x in f1:
    print(x)
'''
