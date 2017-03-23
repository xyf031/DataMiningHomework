__author__ = 'XYF'

import numpy as np
import matplotlib.pyplot as plt

# 处理Wine.txt中的数据
# 该数据集下载自http://cs.joensuu.fi/sipu/datasets/
# 数据集中包含178种酒，每种酒包含13个特征。所以共178行，13列。每个特征上的取值都是0到65535，所以不需要剔除量纲影响
# 根据网站介绍，这些酒来自3个品种，应该被分为3类


#----------Read Data Begins:----------
originData = [line.split() for line in open('Wine.txt')]
print(originData[0:5])

data = []
for line in originData:
    tmp = list(map(int, line))
    data.append(tmp)
Np = len(data)  # 表示酒的个数
print(data[0:5])


def get_distance(x, y):
    if len(x) != len(y):
        print('x和y长度不相同')
        return
    # 使用绝对值距离
    temp = [abs(x[ii] - y[ii]) for ii in range(0, len(x))]
    return sum(temp)


fWriteData = open('Distance-Wine.txt', 'w')
dist = np.zeros((Np, Np))
for i in range(0, Np - 1):
    for j in range(i + 1, Np):
        distance = get_distance(data[i], data[j])
        dist[i, j] = distance
        dist[j, i] = distance
        print(str(i) + ',\t' + str(j) + ':\t\t' + str(distance))
        fWriteData.writelines(str(i+1) + ' ' + str(j+1) + ' ' + str(distance) + '\r\n')
fWriteData.close()
print('----------Write Finished!----------')


#----------Analysis Begins:----------
def mds(d, p=2):
    """
        mdscale_eigh(D, p=2)
        in: D pairwise distances, |Xi - Xj|
        out: M pvecs N x p  with |Mi - Mj| ~ D (mod flip, rotate)
        uses np.linalg.eigh
    """
    d2 = d ** 2
    av = d2.mean(axis=0)
    b = -.5 * ((d2 - av) .T - av + d2.mean())
    evals, evecs = np.linalg.eigh(b)
    pvals = evals[-p:] ** .5
    pvecs = evecs[:, -p:]
    m = pvecs * pvals
    return m - m.mean(axis=0)

coordinates = mds(dist, 2)
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
plt.plot(Xs, Ys, 'yo', ms=12)
plt.xlabel('X-Wine')
plt.ylabel('Y-Wine')
plt.title('Wine Points')
plt.axis([x_min, x_max, y_min, y_max])
plt.savefig('Output/Wine Points.png')
plt.show()
