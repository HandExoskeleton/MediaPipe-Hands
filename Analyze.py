import csv
import numpy as np
import math
import os
import matplotlib.pyplot as plt

def process_coord(projName, csv_file):

    # Open the CSV file for reading
    file = open(csv_file, 'r')

    # Create a CSV reader object
    reader = csv.reader(file)

    ### Generate Distance CSV File and Plot ###

    # 2D array to hold distance coordinates and time steps
    totalDistCoord = []  # accumulates -> used for Total Distance Plot

    # Skip the header row
    next(reader)

    # Initialize previous row
    previous_row = None

    # Open a new CSV file for writing
    output_file = open(os.path.join(projName, projName + "_Distance.csv"), 'w', newline='')
    writer = csv.writer(output_file)

    # Write header row
    landmark_positions = []
    for i in range(21):
        landmark_positions.extend(['Landmark ' + str(i + 1)])
    landmark_positions.extend(["Time"])
    writer.writerow(landmark_positions)

    # Read and process each row
    for row in reader:
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

    plt.xlabel('Total Time (milli-seconds)')
    plt.ylabel('Total Distance')
    plt.title('Total Distance per Landmark')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.7)
    save_path = os.path.join(projName, projName + "_TotalDistancePlot.png")
    plt.savefig(save_path)  # Save plot
    plt.show()

    ### Generate Velocity CSV File and Plot ###

    # Calculate velocity
    time = list(zip(*totalDistCoord))[21]  # Extract time values
    total_distance = list(zip(*totalDistCoord))[:21]  # Extract total distance values for each landmark

    velocity = []
    for distances in total_distance:
        landmark_velocity = [distances[0]]  # Starting velocity is the first distance value
        for i in range(1, len(distances)):
            delta_distance = distances[i] - distances[i - 1]  # Calculate change in distance
            delta_time = time[i] - time[i - 1]  # Calculate change in time
            landmark_velocity.append(delta_distance / delta_time)  # Calculate velocity and append to list
        velocity.append(landmark_velocity)  # Append velocity list for each landmark

    # Open a new CSV file for writing velocity
    velocity_output_file = open(os.path.join(projName, projName + "_Velocity.csv"), 'w', newline='')
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

    plt.xlabel('Time (milli-seconds)')
    plt.ylabel('Velocity')
    plt.title('Velocity per Landmark')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.7)
    save_path = os.path.join(projName, projName + "_VelocityPlot.png")
    plt.savefig(save_path)  # Save plot
    plt.show()

    ### Generate Acceleration CSV File and Plot ###

    # Calculate acceleration
    acceleration = []
    for velocities in velocity:
        landmark_acceleration = [velocities[0]]  # Starting acceleration is the first velocity value
        for i in range(1, len(velocities)):
            delta_velocity = velocities[i] - velocities[i - 1]  # Calculate change in velocity
            delta_time = time[i] - time[i - 1]  # Calculate change in time
            landmark_acceleration.append(delta_velocity / delta_time)  # Calculate acceleration and append to list
        acceleration.append(landmark_acceleration)  # Append acceleration list for each landmark

    # Open a new CSV file for writing acceleration
    acceleration_output_file = open(os.path.join(projName, projName + "_Acceleration.csv"), 'w', newline='')
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

    plt.xlabel('Time (milli-seconds)')
    plt.ylabel('Acceleration')
    plt.title('Acceleration per Landmark')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.subplots_adjust(right=0.7)
    save_path = os.path.join(projName, projName + "_AccelerationPlot.png")
    plt.savefig(save_path)  # Save plot
    plt.show()

# Run analysis
# process_coord("sample", os.path.join("sample", "sample" + "_Coordinates.csv"))
