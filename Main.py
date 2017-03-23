__author__ = 'XYF'

import numpy as np
import matplotlib.pyplot as plt


# ----------Important Variables:----------
# data      :矩阵，N_pair行，3列。原始数据，保存任意两点之间的距离。
# N_pair    :数字，表示原始数据中有多少个点对。
# Np        :数字，一共有多少个点。
# dist      :方形矩阵，Np阶。点与点之间的距离矩阵。
# dc        :数字，判断邻居与否的界限值
# rho       :一维，长度Np。每个点的rho值
# delta     :一维，长度Np。每个点的delta值
# deltaJ    :一维，长度Np。最近的、更高的rho对应的点序号，全局最高rho对应的点用-1表示
# N_grid    :数字，自动寻找类中心点时分割的网格密度。
# N_center  :数字，表示聚成的类的个数。
# cluster   :一维，长度Np。每个点的聚类结果。0表示未分类，用正整数表示类序号，从1开始标号。
# outlier   :一维，长度Np。标记每个点是否是Outlier，是则标记为1，其余是0。
# core      :一维，长度Np。标记每个点是否是对应类的核心部分，还是边界部分。
# clusterCenter :一维，长度N_center。保存每个类的中心点的编号。编号从1开始，结束于Np
# rho_min   :数字，表示类中心的rho下界。
# delta_min :数字，表示类中心的delta下界。
# rhoC      :数字，表示Outlier的rho阀值
# resultPoint   :一维，统计总点数、未分配点个数、Outlier个数、各类的点个数。
# resultCluster :矩阵，N_center行，4列。行表示各Cluster，4列分别表示：总点数、Core内点数、层边界点数、层内Outlier个数。


# Write into File
# 需要提前新建文件夹：Output
fReadme = open('Output/Readme.txt', 'w')
fDecision = open('Output/Decision Rho and Delta.txt', 'w')
fCluster = open('Output/Cluster Result.txt', 'w')


#----------Read Data Begins:----------
# 重要！！原始数据的格式仅支持一种：每行3个数“点A标号 点B标号 点AB间的距离”。而且所有点的标号必须从1开始连续整数标号。
originData = [line.split() for line in open('Distance-Simulation.txt')]
print(originData[0:5])

data = []
for line in originData:
    tmp = list(map(int, line[0:2]))
    tmp.append(float(line[2]))
    data.append(tmp)
N_pair = len(data)  # 表示数据对的个数，即距离记录表的行数，从1开始计数
print(data[0:5])

N1 = max([line[0] for line in data])
N2 = max([line[1] for line in data])
Np = max(N1, N2)  # 表示点的个数，即距离矩阵的阶数，从1开始计数

dist = np.zeros((Np, Np))
for i in data:
    dist[i[0] - 1, i[1] - 1] = i[2]
    dist[i[1] - 1, i[0] - 1] = i[2]
print()
print(dist[0:6, 0:6])
print('\n-----Read finished: ' + str(N_pair) + ' records read, ' + str(Np) + ' points in total.-----\n')
fReadme.writelines('Read finished: ' + str(N_pair) + ' records read, ' + str(Np) + ' points in total.\r\n\r\n')


#----------Decision Parameter Calculation Begins:----------
percent = 2.0
position = round(N_pair * percent / 100)
distSorted = [i[2] for i in data]
distSorted.sort()
dc = distSorted[position]
print('Neighbor cutoff dc is : ' + str(dc) + ', which is the ' + str(percent) + '% of all distances.')
fReadme.writelines('Neighbor cutoff dc is : ' + str(dc) + ', which is the ' + str(percent)
                   + '% of all distances.\r\n\r\n')

#-----阀值核-----
# rho = []
# for i in dist:
#     tmp = 0
#     for j in i:
#         if dc > j > 0:
#             tmp += 1
#     rho.append(tmp)

#-----高斯核-----
rho = [0] * Np
for i in range(Np - 1):
    for j in range(i + 1, Np):
        tmp = np.exp(0 - (dist[i, j] / dc) * (dist[i, j] / dc))
        rho[i] = rho[i] + tmp
        rho[j] = rho[j] + tmp

delta = []
deltaJ = []
for i in range(Np):
    tmp = []
    for j in range(Np):
        if rho[j] > rho[i]:
            tmp.append(j)
    if len(tmp) > 0:
        tmp1 = tmp[0]
        tmp2 = dist[i, tmp1]
        for k in tmp:
            if dist[i, k] < tmp2:
                tmp1 = k
                tmp2 = dist[i, tmp1]
        delta.append(tmp2)
        deltaJ.append(tmp1)
    else:
        delta.append(max(dist[i]))
        deltaJ.append(-1)

print('-----The rho and delta achieved.-----\n')
fReadme.writelines('The rho and delta achieved. See them in the "Decision Rho and Delta.txt".\r\n')
fReadme.writelines('(In the "Decision Rho and Delta.txt", the first is rho, and the second is delta.)\r\n\r\n')
tmp = []
for i in range(Np):
    tmp.append(str(i + 1) + ':\t' + str(rho[i]) + '\t' + str(delta[i]) + '\r\n')
fDecision.writelines('The columns are: point id, rho, delta.\r\n')
fDecision.writelines(tmp)
fDecision.close()


#----------Search for centers Begins:----------
N_grid = 20  # This number depends on the situation.

rho_grid = np.linspace(min(rho), max(rho), N_grid + 1)
delta_grid = np.linspace(min(delta), max(delta), N_grid + 1)


def find_center(rho_bottom, delta_bottom):
    center = 0
    for ii in range(Np):
        if rho[ii] >= rho_bottom and delta[ii] >= delta_bottom:
            center += 1
    return center


center_try = []
for i in range(N_grid, 0, -1):
    center_try.append(find_center(rho_grid[i], delta_grid[i]))

tmp = -1
N_center = 0
for i in range(0, N_grid):
    if center_try[i] == 1:
        tmp = i
        N_center = 1
    if 2 <= center_try[i] <= 6:
        tmp = i
        N_center = center_try[i]
        break

if tmp < 0:
    print('Search for centers FAILED!!! 需要手动修改 N_grid 变量（第100行附近），可依次尝试20、30、……100等')
    fReadme.writelines('Search for centers FAILED!!! 需要手动修改 N_grid 变量（第100行附近），可依次尝试20、30、……100等')
    fReadme.close()
    fCluster.close()
    print()
    exit(0)
rho_min = rho_grid[N_grid - tmp]
delta_min = delta_grid[N_grid - tmp]

print('The number of Clusters is: ' + str(N_center))
print("The min of the center's rho is: " + str(rho_min) + ', min of delta is: ' + str(delta_min))
fReadme.writelines('The number of Clusters is: ' + str(N_center) + '\r\n')
fReadme.writelines("The min of the center's rho is: " + str(rho_min)
                   + ', min of delta is: ' + str(delta_min) + '\r\n\r\n')


#----------Cluster Each Point Begins:----------
cluster = np.zeros(Np)  # 每个点的聚类结果，类序号从1开始。
outlier = np.zeros(Np)  # 使用rho阀值法标记Outlier，是Outlier用1表示，其余用0。
core = np.zeros(Np)  # 每个类的核心群组，核心点用自身的类序号表示，非核心的点用自身类序号的相反数表示。
clusterCenter = np.zeros(N_center)  # 保存每个类的中心点的编号。编号从1开始，结束于Np

percentRho = 1.0
positionRho = round(Np * percentRho / 100)
rhoSorted = sorted(rho)
rhoC = rhoSorted[positionRho]
print('Our Outlier cutoff rho is : ' + str(rhoC) + ', come from the ' + str(percentRho) + '% of all points.')
print("(The rho no-greater-than 'cutoff' will be set as Outliers.)")
fReadme.writelines('Outlier cutoff rho is : ' + str(rhoC)
                   + ', which is the ' + str(percentRho) + '% of all points.\r\n')
fReadme.writelines("(The rho no-greater-than 'cutoff' will be set as Outliers.)\r\n\r\n")

tmp = 0
for i in range(Np):
    if rho[i] >= rho_min and delta[i] >= delta_min:
        clusterCenter[tmp] = i + 1
        tmp += 1
        cluster[i] = tmp
    if rho[i] <= rhoC:
        outlier[i] = 1


def get_cluster(point):
    if cluster[point] > 0:
        return cluster[point]
    else:
        temp = get_cluster(deltaJ[point])
        cluster[point] = temp
        return temp


for i in range(Np):
    if cluster[i] == 0:
        cluster[i] = get_cluster(i)

for i in range(Np):
    core[i] = cluster[i]

for i in range(Np - 1):
    if core[i] >= 0:
        for j in range(i + 1, Np):
            if cluster[i] != cluster[j] and dist[i, j] <= dc:
                core[i] = 0 - cluster[i]
                core[j] = 0 - cluster[j]
                break

print()
print('----------The Clustering is finished! Cheers!----------')
fReadme.writelines('The Clustering result is written into the "Cluster Result.txt".\r\n')
fReadme.writelines('(Cluster number begins from 1.'
                   + ' And the columns are: the point id, cluster number, core number, outlier flag.)\r\n')
fReadme.writelines('\r\n----------The Clustering is finished! Cheers!----------\r\n\r\n')
print()

tmp = []
for i in range(Np):
    tmp.append(str(i + 1) + ':\t' + str(int(cluster[i])) + '\t'
               + str(int(core[i])) + '\t' + str(int(outlier[i])) + '\r\n')
fCluster.writelines('The columns are: the point id, cluster number, core number, outlier flag.\r\n')
fCluster.writelines(tmp)
fCluster.close()


#----------Analysis of the result Begins:----------
# Analise the result of every point.
resultPoint = [0, 0, 0]
for i in range(N_center):
    resultPoint.append(0)

for i in cluster:
    resultPoint[int(i) + 1] += 1
    resultPoint[N_center + 2] += 1

tmp = resultPoint[N_center + 2]
for i in range(len(resultPoint) - 1, 0, -1):
    resultPoint[i] = resultPoint[i - 1]
resultPoint[0] = tmp
tmp = resultPoint[1]
resultPoint[1] = resultPoint[2]
resultPoint[2] = tmp

tmp = 0
for i in outlier:
    tmp += i
resultPoint[2] = tmp
print('\nThe whole situation of all points are: \n[Total, Not Assigned, Outliers; Cluster 1, ...]')
print(resultPoint)
fReadme.writelines('The whole situation of all points are: \r\n[Total, Not Assigned, Outliers, Cluster 1, ...]\r\n')
fReadme.writelines(str(resultPoint))


# Analise the result of every Cluster
resultCluster = np.zeros((N_center, 4))
print('\nThe situation of each cluster is: ')
fReadme.writelines('\r\n\r\nThe situation of each cluster is: \r\n')
for i in range(N_center):
    for j in range(Np):
        if cluster[j] == i + 1:
            resultCluster[i, 0] += 1
            if outlier[j] == 1:
                resultCluster[i, 3] += 1
            if core[j] > 0:
                resultCluster[i, 1] += 1
            else:
                resultCluster[i, 2] += 1

    print('Cluster ' + str(i) + ': Total points--> ' + str(resultCluster[i, 0]) + '\tPoints in the core--> '
          + str(resultCluster[i, 1]) + '\tPoints not in the core--> ' + str(resultCluster[i, 2])
          + '\tPoints are outlier--> ' + str(resultCluster[i, 3]) + '\tCenter of the cluster--> '
          + str(clusterCenter[i]))
    fReadme.writelines('Cluster ' + str(int(i)) + ': Total points--> ' + str(int(resultCluster[i, 0]))
                       + '\tPoints in the core--> ' + str(int(resultCluster[i, 1]))
                       + '\tPoints not in the core--> ' + str(int(resultCluster[i, 2]))
                       + '\tPoints are outlier--> ' + str(int(resultCluster[i, 3])) + '\tCenter of the cluster--> '
                       + str(int(clusterCenter[i])) + '\r\n')


# Plot 1: Rho and Delta
colors = ['r', 'y', 'b', 'm', 'g', 'w']
plt.figure(1)
plt.plot(rho, delta, 'co', ms=8)
x_min = rho_grid[0] * 1.0
x_max = (rho_grid[N_grid] - x_min) * 1.1 + x_min
y_min = delta_grid[0] * 1.0
y_max = (delta_grid[N_grid] - y_min) * 1.1 + y_min
for i in range(N_center):
    plt.plot(rho[int(clusterCenter[i]) - 1], delta[int(clusterCenter[i]) - 1], colors[i % 6] + 'o', ms=12, hold='on')
    plt.text(rho[int(clusterCenter[i]) - 1] + 0.02*(x_max-x_min),
             delta[int(clusterCenter[i]) - 1] + 0.02*(y_max-y_min), str(i + 1))
plt.plot([rho_min, rho_min], [delta_min, 2*delta_grid[N_grid]], 'k--')
plt.plot([rho_min, 2*rho_grid[N_grid]], [delta_min, delta_min], 'k--')
plt.xlabel('Rho')
plt.ylabel('Delta')
plt.title('Rho and Delta of each point')
plt.axis([x_min, x_max, y_min, y_max])
plt.savefig("Output/Rho and Delta.png")
# plt.show()


# MDS
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


# Plot 2: Cluster Result
plt.figure(2)
xcoor_min = min(Xs)
xcoor_max = max(Xs)
ycoor_min = min(Ys)
ycoor_max = max(Ys)
x_min = xcoor_min - (xcoor_max - xcoor_min) * 0.1
x_max = xcoor_max + (xcoor_max - xcoor_min) * 0.1
y_min = ycoor_min - (ycoor_max - ycoor_min) * 0.1
y_max = ycoor_max + (ycoor_max - ycoor_min) * 0.1
for i in range(Np):
    if outlier[i] > 0:
        plt.plot(coordinates[i][0], coordinates[i][1], 'wo', hold=True)
    else:
        if core[i] > 0:
            plt.plot(coordinates[i][0], coordinates[i][1], colors[int(cluster[i]-1) % 6] + 'o', hold=True)
        else:
            plt.plot(coordinates[i][0], coordinates[i][1], 'ko', hold=True)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Cluster Result')
plt.axis([x_min, x_max, y_min, y_max])
plt.savefig('Output/Cluster Result.png')
plt.show()


print('\n\n------------------Program Finishes Successfully!-------------------------')
print('肖一凡 计算机系 2014210871')
fReadme.writelines('\r\n\r\n\r\n-----------------------------------------------------\r\n')
fReadme.writelines('肖一凡 计算机系 2014210871\r\n')
fReadme.close()
