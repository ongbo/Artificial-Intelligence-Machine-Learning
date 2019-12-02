import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# Num = [1,2,3,4,5,6,7,8,9,10,
#        11,12,13,14,15,16,17,18,19,20,
#        21,22,23,24,25,26,27,28,29,30]
#
# density = [0.697,0.774,0.634,0.608,0.556,0.403,0.481,0.437,0.666,0.243,
#            0.245,0.343,0.639,0.657,0.360,0.593,0.719,0.359,0.339,0.282,
#            0.748,0.714,0.483,0.478,0.525,0.751,0.532,0.473,0.725,0.446]
#
# Sugar_content = [0.460,0.376,0.264,0.318,0.215,0.237,0.149,0.211,0.091,0.267,
#                  0.057,0.099,0.161,0.198,0.370,0.042,0.103,0.188,0.241,0.257,
#                  0.232,0.346,0.312,0.437,0.369,0.489,0.472,0.376,0.445,0.459]
# x=[]
# for i in range(0, len(density)):
#     a=[density[i],Sugar_content[i]]
#     x.append(a)
x = np.array([
    [0.697,0.460],
    [0.774,0.376],
    [0.634,0.264],
    [0.608,0.318],
    [0.556,0.215],
    [0.403,0.237],
    [0.481,0.149],
    [0.437,0.211],
    [0.666,0.091],
    [0.243,0.267],
    [0.245,0.057],
    [0.343,0.099],
    [0.639,0.161],
    [0.657,0.198],
    [0.360,0.370],
    [0.593,0.042],
    [0.719,0.103],
    [0.359,0.188],
    [0.339,0.241],
    [0.282,0.257],
    [0.748,0.232],
    [0.714,0.346],
    [0.483,0.312],
    [0.478,0.437],
    [0.525,0.369],
    [0.751,0.489],
    [0.532,0.472],
    [0.473,0.376],
    [0.725,0.445],
    [0.446,0.459]
])
plt.title("Melon Data   K-means")
plt.xlabel("Density",fontsize = 18)
plt.ylabel("Sugar_content",fontsize = 18)
plt.subplot(211)
plt.scatter(x,Sugar_content,c='b')

plt.subplot(212)


y_pred = KMeans(n_clusters=3,random_state=170).fit_predict(x)


plt.scatter(x[y_pred==0][:,0],x[y_pred==0][:,1],marker='*',c='b')
plt.scatter(x[y_pred==1][:,0],x[y_pred==1][:,1],marker='+',c='r')
plt.scatter(x[y_pred==2][:,0],x[y_pred==2][:,1],marker='1',c='g')


plt.show()