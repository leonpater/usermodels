
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, stats
import numpy.polynomial.polynomial as poly



def fix_blinks(y, trail_number, difficulty_cond):
    x = range(len(y))
    x_non_zero = []
    for i in range(1,len(y),1):
        if (y[i] == 0 or abs(y[i]-y[i-1]) > 0.005*y[i]):
            continue
        x_non_zero.append(i)

    # x_non_zero = np.nonzero(y)[0]

    y_non_zero = np.take(y,x_non_zero)
    x_new = np.arange(0,len(y),0.2)


    tck = interpolate.splrep(x_non_zero, y_non_zero, k = 3)  # a 3rd order spline
    # y_new = interpolate.splev(x_new, tck, der=0)
    # coefs = poly.polyfit(x_non_zero, y_non_zero, 3)
    # y_new = poly.polyval(x_new, coefs)
    y_new = interpolate.splev(x, tck, der=0)
    y_new = [int(round(i)) for i in y_new]
    return np.array(y_new)
    # plt.plot(x, y, 'o', x_new, y_new)
    # plt.title('Trial Number = ' + str(trail_number) + '  , Difficulty =' + str(difficulty_cond))
    # plt.savefig('Plots/Daan/Trial_' + str(trail_number) + '_' + str(difficulty_cond)+ '_' + '.png')
    # plt.clf()

def main():

    data = pd.read_csv("/Users/vashisthdalmia/Documents/GitHub/usermodels/subject-swaraj.csv")
    # data = pd.read_csv("/Users/vashisthdalmia/Documents/GitHub/usermodels/subject-daan.csv")

    ps = data.str_pupilsizes
    # print("Number of trials: ", len(ps_daan))
    difficuly_order = data.difficulty
    # print(difficuly_order)

    pupil_sizes = []
    slopes = []
    intercept = []
    for i in range(len(ps)):
        print(i)
        trial = [int(float(x)) for x in ps[i].split(',')]
        fixed_pupil_data = fix_blinks(trial, i, difficuly_order[i])
        pupil_sizes.append(fixed_pupil_data)
        x = np.arange(0,len(fixed_pupil_data),1)
        slope = stats.linregress(x, fixed_pupil_data)[0]
        slopes.append(slope)

    

    # pupil_sizes.append(fix_blinks(trial))


    # ps_arithmetic_task = np.zeros((3,len(pupil_sizes[1])), dtype=float)
    # ps_word_reversals = np.zeros((3,len(pupil_sizes[1])), dtype=float)


    
    for i in range(15):
        if difficuly_order[i] == "easy":
            print('reached easy')
            plt.plot(0,slopes[i],'bo', label = 1)
        elif difficuly_order[i] == "medium":
            plt.plot(1,slopes[i],'go', label = 2)
        else:
            plt.plot(2,slopes[i],'ro', label = 3) 
    plt.title("Swaraj : Slopes for Arithmetic tasks")
    plt.savefig('Plots/Swaraj_Slopes_Arithmetic.png')
    # plt.clf()

    # for i in range(15,30,1):
    #     if difficuly_order[i] == "easy":
    #         print('reached easy')
    #         ps_word_reversals[0,:] += pupil_sizes[i]
    #     elif difficuly_order[i] == "medium":
    #         ps_word_reversals[1,:] += pupil_sizes[i]
    #     else:
    #         ps_word_reversals[2,:] += pupil_sizes[i]
    


    # # pupilsizes_1 = 
    # # for i in range()

    # # print(len(pupil_swaraj[0]))
    # plt.plot(ps_arithmetic_task[0], 'b-', label = "easy")
    # plt.plot(ps_arithmetic_task[1], 'r-', label = "medium")
    # plt.plot(ps_arithmetic_task[2], 'g-', label = "hard")

    # plt.show()

    
if __name__ == '__main__':
    # fix_blinks()
    main()