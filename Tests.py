# import the necessary libraries
import math
import matplotlib.pyplot as plt
import csv
import time
import numpy as np

# gravitational constant used to find acceleration
gravitational_constant = 6.67e-11

# Allows us to set the number of steps and the amount of each time step skips
n_steps = 20000
time_step = 0


# defines arrays needed to be global
planet_x_positions = []
planet_y_positions = []
planets = []
tot_x = []
tot_y = []
total_energy = []
test_energy = []
ch = False
times = []

#steps = [500, 1000, 5000, 10000, 25000, 50000, 75000, 100000]
steps = [20000,40000,60000,80000]


# defines the class of a planet
class Planet:
    def __init__(self, name,  mass, x_pos, y_pos, x_vel, y_vel ):
        self.name = name
        self.mass = mass
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_vel = x_vel
        self.y_vel = y_vel


# method to just return c and y acceleration used when using the different schemes
def get_acc(planet, temp):
    accx = 0
    accy = 0
    for other in enumerate(planets):
        if planet.name == other[1].name:  # ensuring the planet isnt using itself to accelerate
            pass

        elif planet.name == "Sun":
            accx = 0
            accy = 0

        else:
            dist = distance(temp[0], temp[1], other[1].x_pos, other[1].y_pos)

            # calculates the change in x and y
            x_displacement = other[1].x_pos - temp[0]
            y_displacement = other[1].y_pos - temp[1]

            accx = calc_acc(accx, x_displacement, other[1].mass, dist)
            accy = calc_acc(accy, y_displacement, other[1].mass, dist)
    return accx, accy


# calculates the distance between two points in 2d space
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


# sums the acceleration of the planet
def calc_acc(acc, dis, mass, dist):
    acc += (gravitational_constant * dis * mass) / dist ** 3
    return acc

# computes the energy and acceleration of planets
def compute_acc(planet):
    accx = 0
    accy = 0

    potential_energy = 0

    for other in enumerate(planets):
        if planet[0] == other[0]:           # ensuring the planet is not using itself to accelerate
            pass

        elif planet[0] == 0:
            accx = 0
            accy = 0

        else:     # calculates the distance between two objects
            dist = distance(planet[1].x_pos, planet[1].y_pos, other[1].x_pos, other[1].y_pos)

            # calculates the change in x and y
            x_displacement = other[1].x_pos - planet[1].x_pos
            y_displacement = other[1].y_pos - planet[1].y_pos

            # calculates the acceleration in x and y direction
            accx = calc_acc(accx, x_displacement, other[1].mass, dist)
            accy = calc_acc(accy, y_displacement, other[1].mass, dist)

            potential_energy = potential_energy + (-gravitational_constant * planet[1].mass * other[1].mass) / dist

    velocity_squared = planet[1].x_vel ** 2 + planet[1].y_vel ** 2

    kinetic_energy = 0.5 * planet[1].mass * velocity_squared

    return accx, accy, potential_energy, kinetic_energy


# calculates the derivatives used when using the different schemes
def derive(planet, derivative, time_step_1):
    intermediate = [0, 0, 0, 0]  # intermediate will be overwritten
    intermediate[0] = planet.x_pos + derivative[0] * time_step_1
    intermediate[1] = planet.y_pos + derivative[1] * time_step_1
    intermediate[2] = planet.x_vel + derivative[2] * time_step_1
    intermediate[3] = planet.y_vel + derivative[3] * time_step_1
    x_acc_1, y_acc_1 = get_acc(planet, intermediate)
    return [intermediate[2], intermediate[3], x_acc_1, y_acc_1]


def calc_rk4(k1,k2,k3,k4, time_step):
    return 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * time_step

def run_step_rk4(planets):
    kinetic_energy = 0
    potential_energy = 0

    for planet in enumerate(planets):

        # computes the x and y accleration
        x_acc, y_acc, potential, kinetic = compute_acc(planet)

        global ch
        if planet[0] == 0 and ch:
            planet[1].x_pos = 1
            ch = False

        elif planet[0] != 0:
            # calculates the values for each k
            k1 = [planet[1].x_vel, planet[1].y_vel, x_acc, y_acc]
            k2 = derive(planet[1], k1, time_step * 0.5)
            k3 = derive(planet[1], k2, time_step * 0.5)
            k4 = derive(planet[1], k3, time_step)

            # adds it to the previous values multiplying by the time step
            planet[1].x_pos += calc_rk4(k1[0], k2[0], k3[0], k4[0], time_step)
            planet[1].y_pos += calc_rk4(k1[1], k2[1], k3[1], k4[1], time_step)
            planet[1].x_vel += calc_rk4(k1[2], k2[2], k3[2], k4[2], time_step)
            planet[1].y_vel += calc_rk4(k1[3], k2[3], k3[3], k4[3], time_step)
            # sums the kinetic and potential energy
            kinetic_energy = kinetic_energy + kinetic
            potential_energy = potential_energy + potential

    # appends the total energy to the array
    total_energy.append((kinetic_energy + potential_energy))


# gets accseleration 1 step in advance and returns the new accelerations and velocities
def derive_mod(planet, derivative, time_step_1):
    # intermediate will be overwritten
    intermediate = [0, 0, 0, 0]

    intermediate[0] = derivative[0] * time_step_1
    intermediate[1] = derivative[1] * time_step_1
    intermediate[2] = derivative[2] * time_step_1
    intermediate[3] = derivative[3] * time_step_1

    x_acc_1, y_acc_1 = get_acc(planet, intermediate)
    return [intermediate[2], intermediate[3], x_acc_1, y_acc_1]


# run a step of the euler-cromer method
def run_step_cromer(planets):
    kinetic_energy = 0
    potential_energy = 0

    for planet in enumerate(planets):

        # computes the x and y accleration
        x_acc, y_acc, potential, kinetic = compute_acc(planet)

        # keeps the sun central
        global ch
        if planet[0] == 0 and ch:
            planet[1].x_pos = 1
            ch = False

        else:
            planet[1].x_vel += x_acc * time_step
            planet[1].y_vel += y_acc * time_step
            planet[1].x_pos += planet[1].x_vel * time_step
            planet[1].y_pos += planet[1].y_vel * time_step

        kinetic_energy = kinetic_energy + kinetic
        potential_energy = potential_energy + potential

    total_energy.append((kinetic_energy + potential_energy))


# run a step of eulers method
def run_step_euler(planets):
    kinetic_energy = 0
    potential_energy = 0

    for planet in enumerate(planets):

        # computes the x and y accleration
        x_acc, y_acc, potential, kinetic = compute_acc(planet)

        planet[1].x_pos += planet[1].x_vel * time_step
        planet[1].y_pos += planet[1].y_vel * time_step
        planet[1].x_vel += x_acc * time_step
        planet[1].y_vel += y_acc * time_step

        # keeps the Sun central
        global ch
        if planet[0] == 0 and ch:
            planet[1].x_pos = 1
            ch = False

        # sums kinetic and potential energy
        kinetic_energy = kinetic_energy + kinetic
        potential_energy = potential_energy + potential

    # appends the total energy to the array
    total_energy.append((kinetic_energy + potential_energy))


# reads the csv file
def read_csv_file(x):
    with open(x) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            temp = []
            for y in row:
                temp.append(y)

            # creates a new planet and adds it to the array planets
            planets.append(Planet(x_pos=float(temp[1]), y_pos=float(temp[2]), mass=float(temp[3]), x_vel=float(temp[4]),
                                y_vel=float(temp[5]), name=temp[0]))


# plots the graph
def plot(xpos, ypos, dimension):
    # creates figure
    plt.figure("2d Plot")

    # puts limits on the axes
    plt.axes(xlim=(-dimension, dimension), ylim=(-dimension, dimension))

    # plots the points stored in two arrays
    for planet_pos in zip(xpos, ypos):
        plt.plot(planet_pos[0]["x"], planet_pos[1]["y"],
                label=planet_pos[0]["name"])

    plt.legend()


if __name__ == "__main__":

    # sets the schemes for comparison
    scheme = 0
    schemes = [[run_step_euler, "EULER"], [run_step_cromer, "EULER CROMER"], [run_step_rk4, "RK4"]]
    all_energy = []
    all_times = []

    # headers for table printed
    print("%-19s %-9s %-3s %-17s %-1s %-10s" % (
          "SCHEME", "time step", "   ", "energy", "    ", "time"))

    # loops through schemes
    while scheme < 3:
        print("")

        # loops through steps
        for t_step in steps:
            planets.clear()
            planet_x_positions.clear()
            planet_y_positions.clear()

            read_csv_file('SolarSystemData.csv')

            time_step = t_step

            # create the arrays storing the x and y positions
            for planet in planets:
                planet_x_positions.append({"x": [], "name": planet.name})
                planet_y_positions.append({"y": [], "name": planet.name})

            # set necessary variables
            dimension = 0
            i = 0

            prev_max = 0
            prev_min = 1000000
            testenergy = 0
            beg_energy = None

            start_time = time.time()
            negative = True

            # execute program
            while ((i)*time_step)/31556952 <=12 or i==0:
                schemes[scheme][0](planets)

                # store the x position
                for place, planet in enumerate(planet_x_positions):
                    planet["x"].append(planets[place].x_pos)

                    # to find maximum dimension
                    if planets[place].x_pos>dimension:
                        dimension = planets[place].x_pos

                # store the y position
                for place, planet in enumerate(planet_y_positions):
                    planet["y"].append(planets[place].y_pos)

                    #  to find maximum dimension
                    if planets[place].y_pos>dimension:
                        dimension = planets[place].y_pos
                i = i+1

            time_taken = time.time() - start_time
            times.append(time_taken)

            test_energy.append((max(total_energy) - min(total_energy)) / float(-1.9816601e+35))
            print("%-17s %10.1f %-3s %4.15f %-3s %4.2f" %(schemes[scheme][1], t_step , "   " , test_energy[-1]*100, "    ", time_taken))
            total_energy.clear()

        # goes to the next scheme and appends the values
        scheme = scheme + 1
        all_times.append(times)
        all_energy.append(test_energy)

        # clears the temporary lists
        times = []
        test_energy = []

    # increases the dimension so the planet isnt at the border
    dimension = dimension + dimension * 0.1
    plot(planet_x_positions, planet_y_positions, dimension)

    string_steps = []

    for step in steps:
        string_steps.append(str(step))

    # prints graph for time
    plt.figure("graph for time")
    plt.title("Time Taken for Each Time Step", size=12)
    plt.plot(string_steps, all_times[0], label="Euler's Method", marker='x')
    plt.plot(string_steps, all_times[1], '--', label="Euler-Cromer Method", marker='x')
    plt.plot(string_steps, all_times[2], label="RK4 Scheme", marker='x')
    plt.xlabel('Time Step', size=12)
    plt.ylabel('Time Taken (seconds)', size=12)
    plt.legend()
    plt.show()

    # prints graph for rk4 scheme
    plt.figure("graph for time")
    plt.title("Time Taken for Each Time Step", size=12)
    plt.bar(string_steps, all_times[2], label="RK4 Scheme")
    plt.xlabel('Time Step', size=12)
    plt.ylabel('Time Taken (seconds)', size=12)
    plt.legend()
    plt.show()

    # prints graph for energy
    plt.figure("graph for energy")
    plt.title("Energy change for Each Time Step", size=12)
    plt.scatter(string_steps, all_energy[0], label="Euler's Method")
    plt.scatter(string_steps, all_energy[1], label="Euler-Cromer Method")
    plt.scatter(string_steps, all_energy[2], label="RK4 Scheme")
    plt.xlabel('Time Step', size=12)
    plt.ylabel('Range in energy', size=12)
    plt.legend()
    plt.show()
