# import the necessary libraries
import math
import random
import matplotlib.pyplot as plt
import csv
import matplotlib.animation as animation
import numpy as np
import time


def Euler(x, f, h):         # deines how the euler method works
    return x+f(x)*h


def modified_euler(x,f,h):          # deines how the modified euler method works
    K1 = h*f(x)
    K2 = h*f(x+K1)
    return x+0.5*(K1+K2)


def RK4(x,f,h):          # deines how the rk4 scheme works
    K1 = h*f(x)
    K2 = h*f(x + 0.5*K1)
    K3 = h*f(x + 0.5*K2)
    K4 = h*f(x + K3)
    return x + (1/6)*(K1 + 2*K2 + 2*K3 + K4)


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

steps = [500]
#steps = [10000, 60000, 110000, 160000, 210000, 260000, 310000, 360000, 410000]
times = []


# defines the class of a planet
class Planet:
    def __init__(self, name,  mass, x_pos, y_pos, x_vel, y_vel ):
        self.name = name
        self.mass = mass
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_vel = x_vel
        self.y_vel = y_vel


# calculates the distance between two points in 2d space

def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist



def calc_acc(acc, dis, mass, dist):
    acc += (gravitational_constant * dis * mass) / dist ** 3
    return acc



def compute_acc(planet):
    accx = 0
    accy=0
    float(accx)
    float(accy)
    potential_energy = 0
    kinetic_energy = 0

    for other in enumerate(planets):
        if planet[0] == other[0]:           # ensuring the planet isnt using itself to accelerate
            pass

        else:
            # calculates the distance between two objects
            dist = distance(planet[1].x_pos, planet[1].y_pos, other[1].x_pos, other[1].y_pos)

            # calculates the change in x and y
            x_displacement = other[1].x_pos - planet[1].x_pos
            y_displacement = other[1].y_pos - planet[1].y_pos

            # calculates the acceleration in x and y direction
            accx = calc_acc(accx, x_displacement, other[1].mass, dist)
            accy = calc_acc(accy, y_displacement, other[1].mass, dist)


            potential_energy =  potential_energy + (-gravitational_constant * planet[1].mass * other[1].mass) / dist

            velocity_squared = planet[1].x_vel ** 2 + planet[1].y_vel ** 2

            kinetic_energy = kinetic_energy +  0.5 * planet[1].mass * velocity_squared
    return accx, accy, potential_energy, kinetic_energy


def run_step(planets):
    kinetic_energy = 0
    potential_energy = 0
    for planet in enumerate(planets):


        act_time_step = time_step

        # computes the x and y acceleration
        x_acc, y_acc, potential, kinetic = compute_acc(planet)

        # calculates the velocity by differentiating the acceleration
        planet[1].x_vel = Euler(planet[1].x_vel, lambda x: x_acc, act_time_step)
        planet[1].y_vel = Euler(planet[1].y_vel, lambda y: y_acc, act_time_step)

        # calculates the displacment by differentiating the velocity
        planet[1].x_pos = Euler(planet[1].x_pos, lambda x: planet[1].x_vel, act_time_step)
        planet[1].y_pos = Euler(planet[1].y_pos, lambda y: planet[1].y_vel, act_time_step)

        kinetic_energy = kinetic_energy + kinetic
        potential_energy = potential_energy + potential

    tot_energy = 2 * kinetic_energy / abs(potential_energy)
    total_energy.append(tot_energy)


def read_csv_file(x):               # reads the csv file
    with open(x) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            temp = []
            for y in row:
                temp.append(y)

            # creates a new planet and adds it to the array planets
            planets.append(Planet(x_pos=float(temp[1]), y_pos=float(temp[2]), mass=float(temp[3]), x_vel=float(temp[4]),
                                y_vel=float(temp[5]), name=temp[0]))



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


def update(i):          # this is used for the animation and gets the points we need to plot in real time

    # gets the latest values
    newXvalues, newYvalues = getNewValues(i)

    # adds the latest to the array with every value before it
    tot_x.extend(newXvalues)
    tot_y.extend(newYvalues)

    # plots the animation
    scatter.set_sizes([0.2,])
    scatter.set_offsets(np.c_[tot_x,  tot_y])


def getNewValues(i):
    new_x = []
    new_y = []
    anim_speed = 100

    for place, planet in enumerate(zip(planet_x_positions, planet_y_positions)):

        # gets the current values at position i * animation speed (otherwise too slow)
        new_x.append(planet[0]["x"][i*anim_speed])
        new_y.append(planet[1]["y"][i*anim_speed])

    return new_x, new_y


if __name__ == "__main__":
    # get data from csv file


    # execute each step one at a time
    #print(distance(planets[-1].x_pos, planets[-1].y_pos, 0, 0))
    '''while distance(planets[-1].x_pos, planets[-1].y_pos, 0, 0) < abs(1.4335e12) or i<1:
        run_step(planets)
        if i%100:
            print(distance(planets[-1].x_pos, planets[-1].y_pos, 0, 0))'''

    for t_step in steps:
        planets.clear()
        planet_x_positions.clear()
        planet_y_positions.clear()

        read_csv_file('data.csv')

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

        while planets[-3].x_pos>0 or i==0:
        #while i < 10000:
            run_step(planets)

            if i==0:
                beg_energy = float(total_energy[-1])



            if float(total_energy[-1]) > prev_max:
                prev_max = float(total_energy[-1])

            elif float(total_energy[-1]) < prev_min:
                prev_min = float(total_energy[-1])

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

            #testenergy += math.sqrt(abs(float(total_energy[-1]) ** 2 - beg_energy ** 2))
            i = i+1
        dn = time.time() - start_time
        times.append(dn)
        print (dn)


        test_energy.append(abs(prev_max) - abs(prev_min))
        print("ener" , abs(prev_max) - abs(prev_min))
    # increases the dimension so the planet isnt at the border
    dimension = dimension + dimension * 0.1
    plot(planet_x_positions, planet_y_positions, dimension)

    # creates the snaimation
    '''fig = plt.figure("ANIMATION")
    ax = plt.axes(xlim=(-dimension, dimension), ylim=(-dimension, dimension))
    ax.set_xlabel('X Axis', size=12)
    ax.set_ylabel('Y Axis', size=12)

    x = []
    y = []

    scatter = ax.scatter(x, y)

    # updates the animation in real time
    ani = animation.FuncAnimation(fig, update, frames=100000, interval=1)
    plt.show()'''


    string_steps = []

    for step in steps:
        string_steps.append(str(step))

    plt.figure("graph for time")
    plt.title("Time Taken for Each Time Step with Modified Euler's Method", size=12)
    plt.bar(string_steps, times, color=('m', 'g', 'b', 'y', 'r'))
    plt.xlabel('Time Step', size=12)
    plt.ylabel('Time Taken (seconds)', size=12)
    plt.show()

    plt.figure("graph for energy")
    plt.title("Energy change for Each Time Step with Extended Euler's Method", size=12)
    plt.bar(string_steps, test_energy, color=('m', 'g', 'b', 'y', 'r'))
    #plt.ylim(0.1)
    plt.xlabel('Time Step', size=12)
    plt.ylabel('Range in energy', size=12)
    plt.show()
