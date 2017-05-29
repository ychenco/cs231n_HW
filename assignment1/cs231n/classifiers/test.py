import numpy as np

a = np.array([[4,5,6],[7,8,9],[5,6,7],[7,8,9]])
y = np.array([1,2,1,0])
x = np.array(range(4))
b = a[x,y]
print b
c = b[b>3]
print c
d = np.array([a.max(axis=1)])
d = np.reshape(d,[4,1])
print d

print sum(np.exp(a-d))
b[b==5] = 0
print b
e = a>2
print e
print e * np.ones(e.shape)
print np.ones([1,2,3],[])


a = np.array([[4,-1,6],[7,0,-1],[5,6,7],[7,8,-1]])
b = np.array([[4,5,6],[7,8,9],[5,6,7],[7,8,9]])
# b = b * (a>=0)
b[a<=0]=0
print b