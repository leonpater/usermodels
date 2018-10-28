
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate, stats
import numpy.polynomial.polynomial as poly
from scipy import interpolate

def pre_process(pupil_size, base_line):
    # subtracts the average value of the baseline from all the entries of pupil_size
    mean_base = np.mean(base_line)
    pupil_size = [i - mean_base for i in pupil_size]
    return pupil_size

# y is an array of pupil sizes 


def fix_blinks(y):
	# interpolates the blinks
    x = range(len(y))
    x_non_zero = []

	# interpolation doesn't take into account these points. 
	# data points where is a jump in the pupil dilation are removed
    for i in range(1,len(y),1):
	    if (y[i] == 0 or abs(y[i]-y[i-1]) > 0.005*y[i]):
		    continue
	    x_non_zero.append(i)

	# the y values for the x positions 
    y_non_zero = np.take(y,x_non_zero)

    x_new = np.arange(0,len(y),0.2)

	# interpolation with a 3-d spline
    tck = interpolate.splrep(x_non_zero, y_non_zero, k = 3)  # a 3rd order spline

	# the new y values after interpolation 
    y_new = interpolate.splev(x, tck, der=0)
    y_new = [int(round(i)) for i in y_new]
    return np.array(y_new)

def find_pupil_area(pupil_s):
    # finds the area for a given array of pupil sizes !
    area = 0
    for i in pupil_s:
	    area += i
    return area

def pupils_string_to_array(pupil_s):
    pupil_size = []
    for i in range(len(pupil_s)):
    # Reads the pupil sizes from an array of strings
        pupil_size = [int(float(x)) for x in pupil_s[i].split(',')]
    return pupil_size

def preprocess_answer_baseline(baselines):
    processed = []
    for i in range(60,80):
        temp = baselines.iloc[i]
        temp = temp.replace("[", "")
        temp = temp.replace("]", "")
        temp = np.fromstring(temp, dtype=float, sep=',')
        processed.append(temp)
    return processed

def preprocess_answer_pupils(answer_ps):
    processed = []
    for i in range (60,80):
        temp = answer_ps.iloc[i]
        temp = temp.replace("[", "")
        temp = temp.replace("]", "")
        temp = temp.replace(".",",")
        temp = np.fromstring(temp, dtype=int, sep=',')
        processed.append(temp)
    return processed

# return means in correct order 0-6 respectively in np array
def preprocess_means_array(finals,task,diff2):
    final = []
    finals = finals.replace("[", "")
    finals = finals.replace("]", "")
    finals = finals.split(',')

    for i in range(0,len(finals)):
        if i%2==0:
            final.append(float(finals[i]))

    # only take the relevant means
    relevant_means = []
    c=0;
    if(task=='reversal'):
        c=3;
    relevant_means.append(final[c])  # baseline easy
    relevant_means.append(final[c+int(diff2)-1])  # second task index

    return relevant_means

def run_pipeline(baselines, pupils):
	# first correct the blinks 
    areas = []
    for i in range(0,len(pupils)):
        baseline = fix_blinks(baselines[i])
        pupil = fix_blinks(pupils[i])	
        # subtract the baseline 
        pupil = pre_process(pupil,baseline)
        # find the sum of area for the conditions
        a = find_pupil_area(pupil)
        # dont remove outliers here right
        areas.append(a)
    return areas

def main(subject):
    # the data you want to read
    data = pd.read_csv("D:/usermodels/usermodels/Entire_Subject_Data/subject-" + subject + ".csv")

    
    # select and preprocess all relevant data
    question = data.question
    task = data.tasktype.get_values()
    task = task[70]
    diff2 = data.difficulty2[70]

    means_final = data.means_final[70]  # doesnt matter as long as it is in range of 60-80, duplicate array values
    means_final = preprocess_means_array(means_final,task,diff2)
    

    # preparing for pipeline, opensesame (yay) logs are different 
    answer_pupils = data.pupilsizes_answer
    answer_pupils = preprocess_answer_pupils(answer_pupils)  


    baselines = data.str_baseline_question
    baselines = preprocess_answer_baseline(baselines)

    # final pipeline
    areas = run_pipeline(baselines, answer_pupils)
    print(means_final)

    
    
    # For each test-question, print the question, and model-selected answer (closest to mean)
    for i in range(60,80):
        
        # 60 just because different indexing, clunky
        # in this case it is closer to the first mean
        if(abs(means_final[0] - areas[i-60]) < means_final[1] - areas[i-60]):
            
            answer="yes"
        else:
            # closer to the second condition mean
            answer="no"

       
        print(question[i])
        print("Model selected: ",answer)



        # just some spacing for readability
        print("")
    
    
    
    


    
if __name__ == '__main__':
    # fix_blinks()
    main("leon")