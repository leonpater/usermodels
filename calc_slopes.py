
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, stats
import numpy.polynomial.polynomial as poly


# to do : Remove the trials which have a really high std. Calculate the slopes for various window sizes
# and plot the means and s.t.d. 


def pre_process(pupil_size, base_line):
    # subtracts the average value of the baseline from all the entries of pupil_size
    mean_base = np.mean(base_line)
    pupil_size = [i - mean_base for i in pupil_size]
    return pupil_size

def fix_blinks(y, trail_number, difficulty_cond, task_type):
    # interpolates the blinks
    x = range(len(y))
    x_non_zero = []

    # interpolation doesn't take into account these points. 
    # data points where is a jump in the pupil dilation are removed
    for i in range(1,len(y),1):
        if (y[i] == 0 or abs(y[i]-y[i-1]) > 0.005*y[i]):
            continue
        x_non_zero.append(i)

    # x_non_zero = np.nonzero(y)[0]

    # the y values for the x positions 
    y_non_zero = np.take(y,x_non_zero)

    x_new = np.arange(0,len(y),0.2)

    # interpolation with a 3-d spline
    tck = interpolate.splrep(x_non_zero, y_non_zero, k = 3)  # a 3rd order spline

    # the new y values after interpolation 
    y_new = interpolate.splev(x, tck, der=0)
    y_new = [int(round(i)) for i in y_new]

    # plt.plot(x, y, 'o', x, y_new)
    # plt.title('Difficulty = ' + str(difficulty_cond) + ' , Task = ' + str(task_type))
    # plt.savefig('Plots/Leon/Trial_' + str(trail_number) + '_' + str(difficulty_cond)+ '_' + '.png')
    # plt.clf()
    return np.array(y_new)

def mean_slopes(trial_pupil_sizes, fixation_pupil_sizes, difficuly_order, task_type):
    # calculates the mean, simple slopes and performs all the pre-processing steps
    pupil_sizes = []
    pupil_mean = [] 
    slopes = []  
    for i in range(len(trial_pupil_sizes)):
        # Reads the pupil sizes from an array of strings
        pupil_size = [int(float(x)) for x in trial_pupil_sizes[i].split(',')]
        base_line = [int(float(x)) for x in fixation_pupil_sizes[i].split(',')]

        # Interpolates the blinks with a 3rd order spline - same as paper
        pupil_size = fix_blinks(pupil_size, i, difficuly_order[i], task_type[i])
        
        base_line = fix_blinks(base_line, i, difficuly_order[i], task_type[i])

        # the pre-processing step.
        pupil_size = pre_process(pupil_size, base_line)

        x = np.arange(0,len(pupil_size),1)
        slope = stats.linregress(x, pupil_size)[0]

        pupil_sizes.append(pupil_size)
        pupil_mean.append(np.mean(pupil_size))
        slopes.append(slope)
    mean_slopes_data = []
    mean_slopes_data.append(pupil_sizes)
    mean_slopes_data.append(pupil_mean)
    mean_slopes_data.append(slopes)

    return mean_slopes_data

def find_condition(difficulty_order, j, task_type, i):
    # find the trials with difficulty = j and task type = i
    find_pos = []
    for x in range(len(difficulty_order)):
        if (difficulty_order[x] == j and task_type[x] == i):
            find_pos.append(x)
    return find_pos

def find_mean_areas(pupil_sizes, trials, window_size, window_slide):
    # for a specific set of trails finds the mean areas with the sliding wondow 
    mean_area = []
    for j in range(0, len(pupil_sizes[trials[0]]) -  window_size, window_slide):
        mean_trail = [] 
        for i in trials :
            ps = pupil_sizes[i]
            mean_trail.append(find_window_area(ps, j, window_size))
        mean_trial = remove_area_outliers(mean_trail)
        mean_area.append(np.mean(mean_trail))
    return mean_area
    
def find_window_area(pupil_size, start, window_size):
    # finds the area under the window for a given array of pupil sizes !
    sum = 0
    for i in range(start, start+window_size ,1):
        sum += pupil_size[i]
    return sum 

def remove_area_outliers(areas):
    # removes the area outliers that are more than 2 std over the mean
    new_areas = []
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    for i in areas:
        if(abs(mean_area - i) < 2*std_area):
            new_areas.append(i)
    return new_areas

def main(subject):
    # the data you want to read
    data = pd.read_csv("/Users/vashisthdalmia/Documents/GitHub/usermodels/subject-" + subject + ".csv")
    # data = pd.read_csv("/Users/vashisthdalmia/Documents/GitHub/usermodels/subject-daan.csv")

    # stores the pupil sizes for all the trial periods
    trial_pupil_sizes = data.str_pupilsizes
    # stores the pupil sizes for all the fixation cross periods
    fixation_pupil_sizes = data.str_baseline
    # stores the difficulty order and the task_type
    difficuly_order = data.difficulty
    task_type = data.tasktype

    mean_slopes_data = mean_slopes(trial_pupil_sizes, fixation_pupil_sizes, difficuly_order, task_type)
    
    pupil_sizes = np.array(mean_slopes_data[0])
    pupil_mean= np.array(mean_slopes_data[1])
    slopes = np.array(mean_slopes_data[2])

    arr_task = ["arithmetic", "reversal"]
    arr_condition = ["easy", "medium", "hard"]
    struct_condition = np.zeros(shape=(2,3)).tolist()
    area__mean_condition = np.zeros(shape=(2,3)).tolist()
    window_size = 375 # 1000ms since pupil size is stored every 10ms
    window_slide = 25 # 250ms
    for i in range(len(arr_task)):
        for j in range(len(arr_condition)):
            struct_condition[i][j] = find_condition(difficuly_order, arr_condition[j], task_type, arr_task[i])
            area__mean_condition[i][j] = find_mean_areas(pupil_sizes, struct_condition[i][j], window_size, window_slide)
    
    for i in range(len(arr_task)):
        x = range(len(area__mean_condition[i][0]))
        plt.plot(x,area__mean_condition[i][0],'bo', label = "easy")
        plt.plot(x,area__mean_condition[i][1],'ro', label = "medium")
        plt.plot(x,area__mean_condition[i][2],'go', label = "hard")
        plt.legend(loc='upper left')
        plt.title(subject + ", Task = " + str(arr_task[i]))
        plt.savefig('Plots/'+ subject + '_Areas_' + str(arr_task[i]) + '.png')
        plt.clf()


    # for i in range(1,len(pupil_mean),1):
    #     if(task_type[i] == "arithmetic"):
    #         if difficuly_order[i] == "easy":
    #             plt.plot(0,pupil_mean[i],'bo', label = 1)
    #         elif difficuly_order[i] == "medium":
    #             plt.plot(1,pupil_mean[i],'go', label = 2)
    #         else:
    #             plt.plot(2,pupil_mean[i],'ro', label = 3)
    #     else:
    #         if difficuly_order[i] == "easy":
    #             print('reached easy')
    #             plt.plot(3,pupil_mean[i],'bo', label = 4)
    #         elif difficuly_order[i] == "medium":
    #             plt.plot(4,pupil_mean[i],'go', label = 5)
    #         else:
    #             plt.plot(5,pupil_mean[i],'ro', label = 6)
    # plt.title("Swaraj : Means. Arthmetic[0-2], Reversals[3-5]")
    # plt.savefig('Plots/Swaraj_means.png')
    # plt.clf()

    # for i in range(15,30,1):
    #     if difficuly_order[i] == "easy":
    #         print('reached easy')
    #         ps_word_reversals[0,:] += pupil_sizes[i]
    #     elif difficuly_order[i] == "medium":
    #         ps_word_reversals[1,:] += pupil_sizes[i]
    #     else:
    #         ps_word_reversals[2,:] += pupil_sizes[i]
    
    
    # pos_to_del = []
    # for i in range(0,len(pupil_mean),1):
    #     if (abs(pupil_mean[i] - mean) > 2*std) :
    #         pos_to_del.append(i)

    # pupil_mean = np.delete(pupil_mean, (pos_to_del), axis=0)
    # pupil_sizes = np.delete(pupil_sizes, (pos_to_del), axis=0)
    # slopes = np.delete(slopes, (pos_to_del), axis=0)
    
if __name__ == '__main__':
    # fix_blinks()
    main("swaraj")
    main("leon")