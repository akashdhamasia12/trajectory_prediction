
from csv import reader
import numpy as np

# open file and filter the data for us-101
with open('/nwstore/NGSIM/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    counter = 0
    new_list = []
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        if counter == 0:
            new_list.append(row)
        elif counter > 0 and row[-1] == 'us-101':
            #indexes = [i for i,x in enumerate(row) if x == '0']
            new_list.append(row)
            print(row)
        counter = counter+1

# name of csv file  
filename = '/nwstore/NGSIM/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_us101.csv'
import csv    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)
    # writing the data rows  
    csvwriter.writerows(new_list) 

# read the csv file for us-101
import pandas
df = pandas.read_csv('/nwstore/NGSIM/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_i80.csv')
print(df)
#new_list = df.values.tolist()

# sort the data according to vehicle ID
new_np_array = df.values
print(new_np_array)
#new_array_sortedtime = new_np_array[new_np_array[:, 3].argsort()]
#print(new_array_sortedtime)
new_array_sortedvehicle = new_np_array[new_np_array[:, 0].argsort()]
print(new_array_sortedvehicle)
#new_list = new_array_sortedvehicle.tolist()
#new_array_sortedvehicle = new_array_sortedvehicle.tolist()

res_list_right = []
res_list_left = []
initial_index = 0

for vehicle_ID in range(1, 3367): #3110 for us-101 and 3367 for i-80
    #lst = [item[0] for item in new_array_sortedvehicle]
    indices = np.argwhere(new_array_sortedvehicle[:, 0] == vehicle_ID)
    if len(indices) > 0:
        # partition the part of the data for the corresponding vehicle ID and then sort this partition according to the timestamp
        new_partition = new_array_sortedvehicle[indices[0].item():indices[-1].item(), :]
        new_array_sortedtime = new_partition[new_partition[:, 3].argsort()]
        print(new_array_sortedtime)
        new_array_sortedtime = new_array_sortedtime.tolist()
        lane_ID_prev = new_array_sortedtime[0][-12]  #initial lane ID
        global_time_prev = new_array_sortedtime[0][3] # initial global time
        
        for i in range(1, len(new_array_sortedtime)):
            lane_ID = new_array_sortedtime[i][-12] # current lane ID
            global_time = new_array_sortedtime[i][3] # current global times

            if (lane_ID != lane_ID_prev) and ((global_time - global_time_prev) == 100):
                print("Lane change happened!")
                if (lane_ID > lane_ID_prev):    
                    print("Changed to right lane!")
                    res_list_right.append([new_array_sortedtime[i][:]])
                else:
                    print("Changed to left lane!")
                    res_list_left.append([new_array_sortedtime[i][:]])                       
            
            lane_ID_prev = lane_ID
            global_time_prev = global_time


# name of csv file  
filename = '/nwstore/NGSIM/res_list_right_i80.csv'
import csv    
# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)
    # writing the data rows  
    csvwriter.writerows(res_list_right) 


'''
res_list = []
lane_ID_prev = new_list[0][-12]
global_time_prev = new_list[0][3]
vehicle_ID_prev = new_list[0][0]
for i in range(1, len(new_list)):

    lane_ID = new_list[i][-12]
    global_time = new_list[i][3]
    vehicle_ID = new_list[i][0]
    if global_time == global_time_prev:
        print("Multiple vehicles in the scene!")
    if (lane_ID != lane_ID_prev) and (global_time != global_time_prev) and (vehicle_ID_prev == vehicle_ID):
        print("Lane changed happened!")
        res_list.append(new_list[i][:])
    
    lane_ID_prev = lane_ID
    global_time_prev = global_time    
    vehicle_ID_prev = vehicle_ID 


res_list = []
with open('/nwstore/NGSIM/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data_us101.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Iterate over each row in the csv using reader object
    counter = 0
    for row in csv_reader:
        # row variable is a list that represents a row in csv
        lane_ID = row[-12] #3 for us-101, 4 for i-80
        frame_ID = row[1]
        vehicle_ID = row[0]
        global_time = row[3]
        print(frame_ID)
        if counter == 0:
            res_list.append(row)
        elif counter == 1:
            lane_ID_prev = lane_ID 
        elif counter > 1 and (lane_ID != lane_ID_prev):
            print("Lane ID changed!!")
            res_list.append(row)
            lane_ID_prev = lane_ID
        counter = counter+1


'''
'''
for j in range(1, len(new_list)):
    lane_ID = new_list[j][-12]
    vehicle_ID = new_list[j][0]
    global_time = new_list[j][3]
    if lane_ID != lane_ID_prev:
        print("Lane change happened!!")
        if (lane_ID - lane_ID_prev) > 0:
            print("Changed to right lane!")
            res_list_right.append([new_list[j][i] for i in indexes])
        else:
            print("Changed to left lane!")

    res_list.append([new_list[j][i] for i in indexes])

'''