
from csv import reader
import numpy as np
# open file in read mode
with open('/home/nesli/keypointdetection/input/lane_change_23_88456.csv', 'r') as read_obj:
#with open('/nwstore/NGSIM/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    counter = 0
    new_list = []
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        print(row)
        if counter == 1:
            indexes = [i for i,x in enumerate(row) if x == '0']
        new_list.append(row)
        counter = counter+1
res_list
res_list = []
for j in range(4):
    res_list.append([new_list[j][i] for i in indexes])

# name of csv file  
filename = "/home/nesli/keypointdetection/input/lane_change_23_88456_filtered.csv"
import csv    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)
    # writing the data rows  
    csvwriter.writerows(res_list) 