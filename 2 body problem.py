# import the necessary libraries
import math
import matplotlib.pyplot as plt
import csv


# gravitational constant used to calculate acceleration
gravitational_constant = 6.67e-11

# sets the number of steps and the amount of each time step advances
n_steps = 50000
time_step = 10


# defines arrays needed to be global
planet_x_positions = []
planet_y_positions = []
planets = []
tot_x = []
tot_y = []
total_energy = []


# defines the class of a planet
class Planet:
    def __init__(self, name,  mass, x_pos, y_pos, x_vel, y_vel):
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


# calculates the acceleration
def calc_acc(acc, dis, mass, dist):
    acc += (gravitational_constant * dis * mass) / dist ** 3
    return acc


# sets the parameters for the calculation of accleration
def compute_acc(planet1, planet2):
    acc_x = 0
    acc_y = 0

    # calculates the distance between two objects
    dist = distance(planet1.x_pos, planet1.y_pos, planet2.x_pos, planet2.y_pos)

    # calculates the change in x and y
    x_displacement = planet2.x_pos - planet1.x_pos
    y_displacement = planet2.y_pos - planet1.y_pos

    # calculates the acceleration in x and y direction
    acc_x = calc_acc(acc_x, x_displacement, planet2.mass, dist)
    acc_y = calc_acc(acc_y, y_displacement, planet2.mass, dist)

    return acc_x, acc_y


def get_acc(planet, temp):
    accx = 0
    accy = 0

    for other in enumerate(planets):

        # ensuring the planet is not using itself to accelerate
        if planet.name == other[1].name:
            pass

        else:
            dist = distance(temp[0], temp[1], other[1].x_pos, other[1].y_pos)

            # calculates the change in x and y
            x_displacement = other[1].x_pos - temp[0]
            y_displacement = other[1].y_pos - temp[1]

            accx = calc_acc(accx, x_displacement, other[1].mass, dist)
            accy = calc_acc(accy, y_displacement, other[1].mass, dist)
    return accx, accy

def run_step_euler(planet1, planet2):

    planet1.x_pos = 1
    planet1.y_pos = 1

    # computes the x and y accleration
    x_acc, y_acc = compute_acc(planet2, planet1)

    # calculates the velocity by differentiating the acceleration
    planet2.x_pos += planet2.x_vel * time_step
    planet2.y_pos += planet2.y_vel * time_step
    planet2.x_vel += x_acc * time_step
    planet2.y_vel += y_acc * time_step


def run_step_cromer(planet1, planet2):

    planet1.x_pos = 1
    planet1.y_pos = 1

    # computes the x and y accleration
    x_acc, y_acc = compute_acc(planet2, planet1)

    # calculates the velocity by differentiating the acceleration
    planet2.x_vel += x_acc * time_step
    planet2.y_vel += y_acc * time_step
    planet2.x_pos += planet2.x_vel * time_step
    planet2.y_pos += planet2.y_vel * time_step


def run_step_rk4(planet1,planet2):

    planet1.x_pos = 1
    planet1.y_pos = 1

    x_acc, y_acc = compute_acc(planet2, planet1)

    # calculates the values for each k
    k1 = [planet2.x_vel, planet2.y_vel, x_acc, y_acc]
    k2 = derive(planet2, k1, time_step * 0.5)
    k3 = derive(planet2, k2, time_step * 0.5)
    k4 = derive(planet2, k3, time_step)

    # adds it to the previous values multiplying by the time step
    planet2.x_pos += calc_rk4(k1[0], k2[0], k3[0], k4[0], time_step)
    planet2.y_pos += calc_rk4(k1[1], k2[1], k3[1], k4[1], time_step)
    planet2.x_vel += calc_rk4(k1[2], k2[2], k3[2], k4[2], time_step)
    planet2.y_vel += calc_rk4(k1[3], k2[3], k3[3], k4[3], time_step)


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


def plot(xpos, ypos, dim):
    # creates figure
    plt.figure("2d Plot")

    # puts limits on the axes
    plt.axes(xlim=(-dim, dim), ylim=(-dim, dim))

    # plots the points stored in two arrays
    for planet_pos in zip(xpos, ypos):
        plt.plot(planet_pos[0]["x"], planet_pos[1]["y"],
                 label=planet_pos[0]["name"], color='k')


if __name__ == "__main__":
    # get data from csv file
    read_csv_file('EarthMoon.csv')

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
    while i < 250000:
        run_step_rk4(planets[0], planets[1])

        # store the x position
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

    # increases the dimension so the planet isn't at the border
    dimension = dimension + dimension * 0.1
    plot(planet_x_positions, planet_y_positions, dimension)
    plt.show()
