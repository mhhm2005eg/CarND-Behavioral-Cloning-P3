import csv
import numpy as np
import matplotlib.pyplot as plt

drive_log_file = "driving_log.csv"

def get_y():
    lines = []
    with open(drive_log_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)
    steering_measurements = []
    index = 0
    for line in lines:
        if not index:
            index += 1
            continue
        steering_measurement = float(line[3])
        steering_measurements.append(steering_measurement)
    train_y = np.array(steering_measurements)
    return train_y


def save_hist(train_y):
	n, bins, patches = plt.hist(train_y, 200 , facecolor='blue', alpha=0.5)
	#plt.show()
	plt.savefig('hist.png')


	
# -------------------------------------
# Entry point for the script
# -------------------------------------
if __name__ == '__main__':
    #main()
    save_hist(get_y())