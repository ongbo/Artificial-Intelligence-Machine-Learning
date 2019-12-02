import matplotlib.pyplot as plt
input_values = [1,2,3,4,5]
squares = [1,4,9,16,25]
#plt.plot(input_values,squares,linewidth=10)
plt.title("square Numbers",fontsize=24)
plt.xlabel("Value",fontsize = 14)
plt.ylabel("Square of Value",fontsize = 14)
#plt.tick_params(axis='both',labelsize=14)
plt.scatter(1,20)
plt.scatter(2,10,s=90)
plt.scatter(input_values,squares,c='red',edgecolors='none',s=90)
plt.show()