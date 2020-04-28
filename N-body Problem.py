# import the necessary libraries
import math
import matplotlib.pyplot as plt
import csv
import matplotlib.animation as animation
import numpy as np


# gravitational constant used to find acceleration
gravitational_constant = 6.67e-11

# Allows us to set the number of steps and the amount of each time step skips
n_steps = 50000
time_step = 32000

# check variable to keep keep the Sun central
ch = True

# defines arrays needed to be global
planet_x_positions = []
planet_y_positions = []
planets = []
tot_x = []
tot_y = []
total_energy = []
intensity = []


# defines the class of a planet
class Planet:
    def __init__(self, name,  mass, x_pos, y_pos, x_vel, y_vel):
        self.name = name
        self.mass = mass
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.x_acc = 0
        self.y_acc = 0


# calculates the distance between two points in 2d space
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


# calculates the force on one body
def calc_acc(acc, dis, mass, dist):
    acc += (gravitational_constant * dis * mass) / dist ** 3
    return acc

# calculates the force on one body
def calc_force(dis, mass, mass1, dist):
    force = (gravitational_constant * dis * mass * mass1) / dist ** 3
    return force


# method to just return c and y acceleration used when using the different schemes
def get_acc(planet, temp):
    accx = 0
    accy = 0
    for other in enumerate(planets):

        # ensuring the planet is not using itself to accelerate
        if planet.name == other[1].name:
            pass

        # keeps the Sun central
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


# calculates the derivatives used when using the different schemes
def derive(planet, derivative, time_step_1):
    intermediate = [0, 0, 0, 0]  # intermediate will be overwritten
    intermediate[0] = planet.x_pos + derivative[0] * time_step_1
    intermediate[1] = planet.y_pos + derivative[1] * time_step_1
    intermediate[2] = planet.x_vel + derivative[2] * time_step_1
    intermediate[3] = planet.y_vel + derivative[3] * time_step_1
    x_acc, y_acc = get_acc(planet, intermediate)
    return [intermediate[2], intermediate[3], x_acc, y_acc]


# averages out the rk4 scheme
def calc_rk4(k1,k2,k3,k4, time_step):
    return 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * time_step


# execute a step
def run_step(planets):

    kinetic_energy = 0
    potential_energy = 0

    # gets number of planets to implement the loop that requires half the iterations
    leng = len(planets)

    for planet in planets:
        planet.x_acc = 0
        planet.y_acc = 0

    for planet in enumerate(planets):

        j = planet[0] + 1

        # loops through half of the iterations
        while j < leng:
            other = (planets[j])
            if planet[0] == j:
                pass

            else:
                # calculates the distance between two objects
                dist = distance(planet[1].x_pos, planet[1].y_pos, other.x_pos, other.y_pos)

                # calculates the change in x and y
                x_displacement = other.x_pos - planet[1].x_pos
                y_displacement = other.y_pos - planet[1].y_pos

                # calculates the force in x and y direction
                x_force = calc_force(x_displacement, planet[1].mass, other.mass, dist)
                y_force = calc_force(y_displacement, planet[1].mass, other.mass, dist)

                # calculates the acceleration
                planet[1].x_acc += x_force / planet[1].mass
                planet[1].y_acc += y_force / planet[1].mass

                # calculates the acceleration for the other planet it is compared to
                other.x_acc -= x_force / other.mass
                other.y_acc -= y_force / other.mass

                # keeps the Sun central
                if planet[0] == 0:
                    planet[1].x_acc = 0
                    planet[1].y_acc = 0

            j += 1

        # calculates the total energy at each iteration
        global ch
        if planet[1].name == "Sun" and ch:
            planet[1].x_pos = 1
            ch = False

        else:
            velocity_squared = planet[1].x_vel ** 2 + planet[1].y_vel ** 2
            kinetic_energy = kinetic_energy + 0.5 * planet[1].mass * velocity_squared

            for other in planets:
                if planet[1] == other:
                    pass
                elif planet[1].name == "Sun":
                    pass
                else:
                    dist = distance(planet[1].x_pos, planet[1].y_pos, other.x_pos, other.y_pos)
                    potential_energy = potential_energy + (-gravitational_constant * planet[1].mass * other.mass) / dist

            # calculates the values for each k
            k1 = [planet[1].x_vel, planet[1].y_vel, planet[1].x_acc, planet[1].y_acc]
            k2 = derive(planet[1], k1, time_step * 0.5)
            k3 = derive(planet[1], k2, time_step * 0.5)
            k4 = derive(planet[1], k3, time_step)

            # adds it to the previous values multiplying by the time step
            planet[1].x_pos += calc_rk4(k1[0], k2[0], k3[0], k4[0], time_step)
            planet[1].y_pos += calc_rk4(k1[1], k2[1], k3[1], k4[1], time_step)
            planet[1].x_vel += calc_rk4(k1[2], k2[2], k3[2], k4[2], time_step)
            planet[1].y_vel += calc_rk4(k1[3], k2[3], k3[3], k4[3], time_step)
    total_energy.append((kinetic_energy + potential_energy))


def read_csv_file(file):                                # reads the csv file
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for rows in csv_reader:
            temp = []
            for row in rows:
                temp.append(row)

            # creates a new planet and adds it to the array planets
            planets.append(Planet(x_pos=float(temp[1]), y_pos=float(temp[2]), mass=float(temp[3]), x_vel=float(temp[4]),
                                  y_vel=float(temp[5]), name=temp[0]))


def plot(xpos, ypos, dim):
    # creates figure
    plt.figure("2d Plot")

    # puts limits on the axes
    plt.axes(xlim=(-dim, dim), ylim=(-dim, dim))

    # plots the points stored in two arrays
    for planet_pos in zip(xpos, ypos):
        plt.plot(planet_pos[0]["x"], planet_pos[1]["y"],
                 label=planet_pos[0]["name"])

    plt.legend()


def update(j):
    # gets the latest values
    new_x_values, new_y_values = get_new_values(j)

    # plots the animation
    scatter.set_color(["purple"])
    scatter.set_sizes([5, ])
    scatter.set_offsets(np.c_[new_x_values, new_y_values])


def get_new_values(j):
    new_x = []
    new_y = []
    anim_speed = 10

    for place, planet in enumerate(zip(planet_x_positions, planet_y_positions)):

        # gets the current values at position i * animation speed (otherwise too slow)
        try:
            new_x.append(planet[0]["x"][j*anim_speed])
            new_y.append(planet[1]["y"][j*anim_speed])
        except:
            pass

    return new_x, new_y


if __name__ == "__main__":

    # get data from csv file
    read_csv_file('data3.csv')

    # create the arrays storing the x and y positions
    for planet in planets:
        planet_x_positions.append({"x": [], "name": planet.name})
        planet_y_positions.append({"y": [], "name": planet.name})

    for place, planet in enumerate(planet_x_positions):
        planet["x"].append(planets[place].x_pos)

    for place, planet in enumerate(planet_y_positions):
        planet["y"].append(planets[place].y_pos)

    # set necessary variables
    dimension = 0
    i = 0

    # execute each step one at a time
    while planets[-3].x_pos < 0 or i == 0:
        run_step(planets)

        for place, planet in enumerate(planet_x_positions):
            planet["x"].append(planets[place].x_pos)

            # to find maximum dimension
            if planets[place].x_pos > dimension:
                dimension = planets[place].x_pos

        # store the y position
        for place, planet in enumerate(planet_y_positions):
            planet["y"].append(planets[place].y_pos)

            #  to find maximum dimension
            if planets[place].y_pos > dimension:
                dimension = planets[place].y_pos

        i = i+1

    # print the change in energy
    print("%-42s" % ((max(total_energy) - min(total_energy))/float(-1.9816601e+35)))

    # increases the dimension so the planet isn't at the border
    dimension = dimension + dimension * 0.1
    plot(planet_x_positions, planet_y_positions, dimension)

    # creates the animation
    fig = plt.figure("ANIMATION")
    ax = plt.axes(xlim=(-dimension, dimension), ylim=(-dimension, dimension))
    ax.set_xlabel('X Axis', size=12)
    ax.set_ylabel('Y Axis', size=12)

    x = []
    y = []

    scatter = ax.scatter(x, y)

    # updates the animation in real time
    for planet_pos in zip(planet_x_positions, planet_y_positions):
        ax.plot(planet_pos[0]["x"], planet_pos[1]["y"],
                label=planet_pos[0]["name"], linewidth='0.1')

    ani = animation.FuncAnimation(fig, update, frames=10000, interval=1)
    plt.show()

    # plots the graph to show energy
    plt.figure("graph for energy")
    plt.scatter(np.linspace(0, len(total_energy), len(total_energy)), total_energy)
    plt.show()
