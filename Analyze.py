import csv
import numpy as np
import math
import os
import matplotlib.pyplot as plt

def process_coord(projName, csv_file, projDirectory):

    # Open the CSV file for reading
    file = open(csv_file, 'r')

    # Create a CSV reader object
    reader = csv.reader(file)

    ### Generate Distance CSV File and Plot ###

    # 2D array to hold distance coordinates and time steps
    totalDistCoord = []  # accumulates -> used for Total Distance Plot

    # 2D array to hold landmark coordinates for simpler computation
    landmark_coords = []
    # Skip the header row
    next(reader)

    # Initialize previous row
    previous_row = None

    # Open a new CSV file for writing
    output_file = open(os.path.join(projDirectory, projName + "_Distance.csv"), 'w', newline='')
    writer = csv.writer(output_file)

    # Write header row
    landmark_positions = []
    for i in range(21):
        landmark_positions.extend(['Landmark ' + str(i + 1)])
    landmark_positions.extend(["Time"])
    writer.writerow(landmark_positions)

    # Read and process each row
    for row in reader:
        # Gets every landmark coordinate except for time
        for entry in range(len(row)-1):
                landmark_coords.append(row[entry])

        # Check if previous row exists
        if previous_row is not None:

            # Subtract respective values from each cell in the current row
            result = []
            distance = []
            for curr_cell, prev_cell in zip(row, previous_row):
                curr_values = curr_cell.strip('()').split(', ')
                prev_values = prev_cell.strip('()').split(', ')

                # Convert values to integers or floats
                curr_values = [int(val) if val.isdigit() else float(val) for val in curr_values]
                prev_values = [int(val) if val.isdigit() else float(val) for val in prev_values]

                # Perform subtraction based on value count
                if len(curr_values) == len(prev_values):
                    diff = [curr_val - prev_val for curr_val, prev_val in zip(curr_values, prev_values)]
                else:
                    diff = None  # Handle the case when value counts are different

                result.append(diff)  # Difference of values between rows

            for index, cell in enumerate(result):  # only apply distance formula to coordinates (not time)
                if index < 21:
                    distance.append(float(math.sqrt(cell[0] ** 2 + cell[1] ** 2 + cell[2] ** 2)))
                else:
                    distance.append(float(str(cell)[1:-1]))  # time difference
            
            # Write the distance to the CSV file
            writer.writerow(distance)

            # Add row to distCoord 2D array
            if len(totalDistCoord) == 0:  # Check if array is empty -> do not add previous row
                totalDistCoord.append(distance)
            else:
                totalDistCoord.append([x + y for x, y in zip(distance, totalDistCoord[-1])])  # Total distance accumulates
        # Assign the current row as the previous row for the next iteration
        previous_row = row

    # Close the files
    file.close()
    output_file.close()

    # Plot Total Distance Graph
    plt.figure(figsize=(10, 6))  # Adjust the width and height as desired

    for i in range(21):
        plt.plot(list(zip(*totalDistCoord))[21], list(zip(*totalDistCoord))[i], label="Landmark " + str(i + 1))

    plt.xlabel('Total Time (seconds)')
    plt.ylabel('Total Distance')
    plt.title('Total Distance per Landmark')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.7)
    save_path = os.path.join(projDirectory, projName + "_TotalDistancePlot.png")
    plt.savefig(save_path)  # Save plot
    plt.show()

    ### Generate Velocity CSV File and Plot ###

    # Calculate velocity
    time = list(zip(*totalDistCoord))[21]  # Extract time values
    total_distance = list(zip(*totalDistCoord))[:21]  # Extract total distance values for each landmark

    velocity = []
    for distances in total_distance:
        landmark_velocity = [distances[1]]  # Starting velocity is the second distance value
        for i in range(1, len(distances)):
            delta_distance = distances[i] - distances[i - 1]  # Calculate change in distance
            delta_time = time[i] - time[i - 1]  # Calculate change in time
            landmark_velocity.append(delta_distance / delta_time)  # Calculate velocity and append to list
        velocity.append(landmark_velocity)  # Append velocity list for each landmark

    # Open a new CSV file for writing velocity
    velocity_output_file = open(os.path.join(projDirectory, projName + "_Velocity.csv"), 'w', newline='')
    velocity_writer = csv.writer(velocity_output_file)

    # Write header row
    velocity_header = []
    for i in range(21):
        velocity_header.append("Landmark " + str(i + 1))
    velocity_header.extend(["Time"])
    velocity_writer.writerow(velocity_header)

    # Write velocity data to the CSV file
    for i in range(len(time)):
        velocity_row = []
        for j in range(21):
            if j < len(velocity):  # Add this condition to check if j is within the valid range
                velocity_row.append(velocity[j][i])
            else:
                print("Index out of range: j =", j)
        velocity_row.append(time[i])
        velocity_writer.writerow(velocity_row)

    # Close the file
    velocity_output_file.close()

    # Plot Velocity Graph
    plt.figure(figsize=(10, 6))  # Adjust the width and height as desired

    for i in range(21):
        plt.plot(time, velocity[i], label="Landmark " + str(i + 1))

    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity')
    plt.title('Velocity per Landmark')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.7)
    save_path = os.path.join(projDirectory, projName + "_VelocityPlot.png")
    plt.savefig(save_path)  # Save plot
    plt.show()

    ### Generate Acceleration CSV File and Plot ###

    # Calculate acceleration
    acceleration = []
    for velocities in velocity:
        landmark_acceleration = [velocities[1]]  # Starting acceleration is the second velocity value
        for i in range(1, len(velocities)):
            delta_velocity = velocities[i] - velocities[i - 1]  # Calculate change in velocity
            delta_time = time[i] - time[i - 1]  # Calculate change in time
            landmark_acceleration.append(delta_velocity / delta_time)  # Calculate acceleration and append to list
        acceleration.append(landmark_acceleration)  # Append acceleration list for each landmark

    # Open a new CSV file for writing acceleration
    acceleration_output_file = open(os.path.join(projDirectory, projName + "_Acceleration.csv"), 'w', newline='')
    acceleration_writer = csv.writer(acceleration_output_file)

    # Write header row
    acceleration_header = []
    for i in range(21):
        acceleration_header.append("Landmark " + str(i + 1))
    acceleration_header.extend(["Time"])
    acceleration_writer.writerow(acceleration_header)

    # Write acceleration data to the CSV file
    for i in range(len(time)):
        acceleration_row = []
        for j in range(21):
            if j < len(acceleration):  # Add this condition to check if j is within the valid range
                acceleration_row.append(acceleration[j][i])
            else:
                print("Index out of range: j =", j)
        acceleration_row.append(time[i])
        acceleration_writer.writerow(acceleration_row)

    # Close the file
    acceleration_output_file.close()

    # Plot Acceleration Graph
    plt.figure(figsize=(10, 6))  # Adjust the width and height as desired

    for i in range(21):
        plt.plot(time, acceleration[i], label="Landmark " + str(i + 1))

    plt.xlabel('Time (seconds)')
    plt.ylabel('Acceleration')
    plt.title('Acceleration per Landmark')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.7)
    save_path = os.path.join(projDirectory, projName + "_AccelerationPlot.png")
    plt.savefig(save_path)  # Save plot
    plt.show()
    
    ### Proprocessing on 2D Coordinate Array (time excluded):
    landmark_tuples = []

    # Send all the coordinate strings to a 1D array of tuples
    for i in range(len(landmark_coords)):
        # Convert string to tuple
        coord_vals_str = landmark_coords[i].strip('()').split(',')
        coord_vals = [float(val) for val in coord_vals_str]

        # Append tuple to 1D list
        landmark_tuples.append(tuple(coord_vals))

    # Reshape list to preserve order of the landmarks. Each entry
    # will be indexed by landmark_array[time][number][dimension]
    landmark_array = []
    for i in range(len(time)):
        landmark_array.append([])
        for j in range(21):
            landmark_array[i].append(landmark_tuples[21*i +j])

    # Initialize list of empty lists for all time
    joint_angs = [[] for i in range(len(time))]
    
    # Generate tuples for joint locations on fingers
    joint_tuples = [(2, 3, 4), (5, 6, 7), (6, 7, 8), (9, 10 , 12), (10, 11, 12), (13, 14, 15), (14, 15, 16), (17, 18, 19), (18, 19, 20)]

    # Compute joint angle for the given landmark sets. These only apply to joints in the fingers
    for t in range(len(time)):
        for joint in joint_tuples:
            # Generate first vector for line segment formed by landmarks
            vec1 = np.subtract(landmark_array[t][joint[0]], landmark_array[t][joint[1]])

            # Generate second vector for line segment formed by landmarks
            vec2 = np.subtract(landmark_array[t][joint[2]], landmark_array[t][joint[1]])

            # Compute dot product:
            X = np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)

            Y = np.sqrt(1- X **2)
            # Compute angle 
            joint_angs[t].append(np.arctan2(Y, X)* 180 / np.pi) 

    # Open a new CSV file for writing joint angle
    joint_angle_output_file = open(os.path.join(projDirectory, projName + "_JointAngles.csv"), 'w', newline='')
    joint_angle_writer = csv.writer(joint_angle_output_file)

    # Write header row
    joint_angle_header = []
    for i in range(len(joint_tuples)):
        joint_angle_header.append("Joint " + str(i + 1))
    joint_angle_header.extend(["Time"])
    joint_angle_writer.writerow(joint_angle_header)

    # Write joint angle data to the CSV file
    for i in range(len(time)):
        joint_angle_row = []
        for j in range(len(joint_tuples)):
            if j < len(joint_angs):  # Add this condition to check if j is within the valid range
                joint_angle_row.append(joint_angs[i][j])
            else:
                print("Index out of range: j =", j)
        joint_angle_row.append(time[i])
        joint_angle_writer.writerow(joint_angle_row)

    # Close the file
    joint_angle_output_file.close()
    # Plot Joint Angle Graph
    plt.figure(figsize=(10, 6))  # Adjust the width and height as desired

    for i in range(len(joint_tuples)):
        plt.plot(time, [item[i] for item in joint_angs], label="Joint " + str(i + 1))

    plt.xlabel('Time (seconds)')
    plt.ylabel('Joint Angle 1')
    plt.title('Joint Angles for Fingers')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.7)
    save_path = os.path.join(projDirectory, projName + "_JointAngPlot.png")
    plt.savefig(save_path)  # Save plot
    plt.show()

    #Compute the joint angles for the knuckles
    knuckle_tuples = [(1, 2, 3), (0, 5, 6), (0, 9, 10), (0, 13, 14), (0, 17, 18)]
    knuckle_angs = [[] for i in range(len(time))]
    # Compute joint angle for knuckles, a bit more complicated
    for t in range(len(time)):
        for joint in knuckle_tuples:
            # Generate first vector for line segment formed by landmarks (pointing towards the base of the hand)
            vec1 = np.subtract(landmark_array[t][joint[0]], landmark_array[t][joint[1]])

            # Generate second vector for line segment formed by landmarks
            vec2 = np.subtract(landmark_array[t][joint[2]], landmark_array[t][joint[1]])

            # Generate third vector for line segment formed by the next two landmarks after vec2
            vec3 = np.subtract(landmark_array[t][joint[2]+1], landmark_array[t][joint[2]])

            # Generate plane formed by vec2 and vec3
            normal_vec = np.cross(vec2, vec3)

            # Project vec1 onto the plane
            vec1_proj = vec1 - (np.dot(normal_vec, vec1)/(np.linalg.norm(normal_vec) ** 2) * normal_vec)

            # Compute dot product:
            X = np.dot(vec1_proj, vec2) / np.linalg.norm(vec1_proj) / np.linalg.norm(vec2)

            Y = np.sqrt(1- X **2)
            # Compute angle from dot product formula
            knuckle_angs[t].append(np.arctan2(Y, X)* 180 / np.pi)
    
    # Plot Knuckle Angle Graph for Thumb Knuckle
    plt.figure(figsize=(10, 6))  # Adjust the width and height as desired
    for i in range(len(knuckle_tuples)):
        plt.plot(time, [item[i] for item in knuckle_angs], label="Knuckle " + str(i + 1))

    plt.xlabel('Time (seconds)')
    plt.ylabel('Joint Angle 1')
    plt.title('Joint Angles for Index Finger Lower Joint')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.7)
    save_path = os.path.join(projDirectory, projName + "_KnuckleAngPlot.png")
    plt.savefig(save_path)  # Save plot
    plt.show()

    # Calulate angular displacement for joints
    ang_displacement = [[] for i in range(len(time)-1)]
    for i in range(1, len(time)):
        for j in range(len(joint_tuples)):
            # Subtract angles 
            ang_displacement[i-1].append(joint_angs[i][j] - joint_angs[i-1][j])
    
    # Calculate angular speed for joints
    ang_speed = [[] for i in range(len(time)-1)]
    for i in range(1, len(time)):
        for j in range(len(joint_tuples)):
            # Do linear approximation of angular speed
            ang_speed[i-1].append(ang_displacement[i-1][j]/(time[i]-time[i-1]))
    
    # Calculate angular acceleration for joints
    ang_acceleration = [[] for i in range(len(time)-2)]
    for i in range(2, len(time)):
        for j in range(len(joint_tuples)):
            # Do linear approximation of angular speed
            ang_acceleration[i-2].append((ang_speed[i-1][j]-ang_speed[i-2][j])/(time[i-1]-time[i-2]))

