import numpy as np


# sigmoid function
def tan_sigmod(x):
    return (2 / (1 + np.exp(-2*x)))-1

def sigmoid_d(x):
     return x * (1 - x)

# input dataset
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
       [ 0.42058807,  0.44726365,  0.46341698,  0.34186658,  0.3244141 ],
       [ 0.42967748,  0.45591762,  0.46716069,  0.34678152,  0.33687758],
       [ 0.43440384,  0.45960569,  0.46848651,  0.35125377,  0.3389221 ]]))
y_data = np.float32(np.array([[ 0.35121505,  0.35230958,  0.36315452,  0.37856145,  0.39004218,
         0.39354065,  0.39540494,  0.39618955,  0.39608512,  0.39838066,
         0.40124576,  0.40408396,  0.4051577 ,  0.40780942,  0.41054421]]).T)

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)
syn0 = np.random.randn(5, 1) - 0.5

for iter in range(500):
    # forward propagation
        l0 = x_data
        l1 = sigmod(np.dot(l0, syn0))

        l1_error = l1-y_data

        # multiply how much we missed by the
        # slope of the sigmoid at the values in l1
        l1_delta = l1_error * sigmoid_d(l1)
        if iter%50==0:
            print(l1_error)
        syn0 -= np.dot(l0.T, l1_delta)


print(l1)