__author__ = 'XYF'

import numpy as np
import random
import matplotlib.pyplot as plt


def density(x, y):
    # 该函数充当密度函数，来生成模拟数据集。但函数未进行归一化，需要后面进行一些特殊处理。
    # 该函数最大值略小于4，最小值略大于-0.2
    # 该函数的负数绝对值都很小，绝对值大于0.1的数主要集中在-3到3之间
    # 这个函数的图像见Simulation-Function.png文件

    x2 = (x/1) ** 2
    y2 = (y/5) ** 2
    z = np.sin(x2 + 3*y2) / (0.1 + x2 + y2) + (x2 + 5*y2) * np.exp(1 - x2 - y2) / 2
    return z


#----------Generate Data Begins:----------
Np = 0
coordinates = []
grid = np.linspace(-3.5, 3.5, 150)  # 这个参数很重要

for i in range(0, len(grid)):
    for j in range(0, len(grid)):
        tmp1 = density(grid[i], 5*grid[j])
        tmp2 = random.random() * 7 - 0.1  # 这里的参数也很重要
        if tmp1 > tmp2:
            coordinates.append([grid[i], 5*grid[j]])
            Np += 1
print(Np)
print(len(coordinates))


#----------Analysis Begins:----------
Xs = []
Ys = []
for i in range(0, Np):
    Xs.append(coordinates[i][0])
    Ys.append(coordinates[i][1])
xcoor_min = min(Xs)
xcoor_max = max(Xs)
ycoor_min = min(Ys)
ycoor_max = max(Ys)
x_min = xcoor_min - (xcoor_max - xcoor_min) * 0.1
x_max = xcoor_max + (xcoor_max - xcoor_min) * 0.1
y_min = ycoor_min - (ycoor_max - ycoor_min) * 0.1
y_max = ycoor_max + (ycoor_max - ycoor_min) * 0.1

plt.figure(0)
plt.plot(Xs, Ys, 'yo')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simulation Points')
plt.axis([x_min, x_max, y_min, y_max])
plt.savefig('Output/Simulation Points.png')
plt.show()


#----------Write into File:----------
fDistance = open('Distance-Simulation.txt', 'w')
for i in range(0, Np - 1):
    for j in range(i + 1, Np):
        tmp = np.sqrt((coordinates[i][0]-coordinates[j][0])*(coordinates[i][0]-coordinates[j][0])
                      + (coordinates[i][1]-coordinates[j][1])*(coordinates[i][1]-coordinates[j][1]))
        fDistance.writelines(str(i+1) + '\t' + str(j+1) + '\t' + str(tmp) + '\r\n')
fDistance.close()

