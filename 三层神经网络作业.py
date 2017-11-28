import numpy as np
import matplotlib.pyplot as plt
np.random.seed(1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def sigmoid_prime(x):
    return x*(1-x)

x_data = np.float32(np.array([[ 0.36889357,  0.39115305,  0.46387817,  0.28504011,  0.25410423],
       [ 0.37422065,  0.39583106,  0.46460683,  0.28943161,  0.25752109],
       [ 0.37190992,  0.4041215 ,  0.46594294,  0.29645425,  0.26438966],
       [ 0.38197742,  0.41972943,  0.46484586,  0.30867157,  0.27267109],
       [ 0.39161011,  0.43083702,  0.46689615,  0.31980244,  0.2759615 ],
       [ 0.39922397,  0.43493494,  0.47028913,  0.32846788,  0.28453647],
       [ 0.40272884,  0.43763652,  0.46938773,  0.33201671,  0.28844498],
       [ 0.40541264,  0.43899082,  0.47095201,  0.33348557,  0.30624917],
       [ 0.40807383,  0.43895045,  0.47061936,  0.33444512,  0.30357338],
       [ 0.41153542,  0.43964748,  0.46648064,  0.33528383,  0.30904485],
       [ 0.4147732 ,  0.4417963 ,  0.46557501,  0.33740882,  0.31633791],
       [ 0.41772767,  0.44376015,  0.46599713,  0.33936805,  0.31988659],
       [ 0.42058807,  0.44726365,  0.46341698,  0.34186658,  0.3244141 ]]))

y_data = np.float32(np.array([[ 0.35121505,  0.35230958,  0.36315452,  0.37856145,  0.39004218,
         0.39354065,  0.39540494,  0.39618955,  0.39608512,  0.39838066,
         0.40124576,  0.40408396,  0.4051577 ]]).T)
x_test_data=np.array([[ 0.42967748,  0.45591762,  0.46716069,  0.34678152,  0.33687758],
       [ 0.43440384,  0.45960569,  0.46848651,  0.35125377,  0.3389221 ]])
y_test_data=np.array([ [0.40780942,  0.41054421]]).T
y_real=[ 10**(x*10) for x in y_test_data]

print(y_real)
weight0 = np.random.randn(5, 16)-0.63
weight1 = np.random.randn(16, 1)-0.63
bias1=np.random.randn(1,16)
bias2=np.random.randn()
error=[]
iter=[]
for j in range(200):
    l0 = x_data
    l1 = sigmoid(np.dot(l0, weight0)+bias1)
    l2 = sigmoid(np.dot(l1, weight1)+bias2)
    l2_error = l2-y_data
    if (j % 10) == 0:
        print("当前错误率为" + str(np.mean(np.abs(l2_error))))
        error.append(np.mean(np.abs(l2_error)))
        iter.append(j)
    l2_delta = l2_error * sigmoid_prime(l2)
    l1_error = l2_delta.dot(weight1.T)
    l1_delta = l1_error * sigmoid_prime(l1)
    bias1-=np.sum(l1_delta,axis=0)
    bias2-=np.sum(l2_delta,axis=0)
    weight1 -= l1.T.dot(l2_delta)
    weight0 -= l0.T.dot(l1_delta)


layer1=sigmoid(np.dot(x_test_data, weight0)+bias1)
layer2=sigmoid(np.dot(layer1, weight1)+bias2)
y_vitural=[ 10**(x*10) for x in layer2]
errorpercent=[np.abs(x-y)/y for x,y in zip(y_vitural,y_real)]
print(y_real)
print(y_vitural)
print(errorpercent)
plt.plot(iter,error)
plt.show()





