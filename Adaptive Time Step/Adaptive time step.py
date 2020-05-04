# import the necessary libraries
import math
import matplotlib.pyplot as plt
import csv
import matplotlib.animation as animation
import numpy as np
import time

# gravitational constant used to find acceleration
gravitational_constant = 6.67e-11

# Allows us to set the number of steps and the amount of each time step skips
n_steps = 5000
energy = 0
ch = True

# defines arrays needed to be global
planet_x_positions = []
planet_y_positions = []
planets = []
tot_x = []
tot_y = []
total_energy = []
events = []
intensity = []


# defines the class of a planet
class Planet:
    def __init__(self, name,  mass, x_pos, y_pos, x_vel, y_vel ):
        self.name = name
        self.mass = mass
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.vel = 1
        self.prev_vel = 1
        self.k_energy = 0
        self.p_energy = 0
        self.tot_k = 0
        self.tot_p = 0
        self.total_energy = 0
        self.instances = 1
        self.shrink = None
        self.grow = None
        self.explored = False
        self.time_step = 0
        self.grown_larger = False


# calculates the distance between two points in 2d space
def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


# calculates the accleration of the object
def calc_acc(acc, dis, mass, dist):
    acc += (gravitational_constant * dis * mass) / dist ** 3
    return acc


# returns just the x and y acceleration
def get_acc(planet, temp):
    accx = 0
    accy = 0
    for other in enumerate(planets):
        if planet.name == other[1].name:  # ensuring the planet isnt using itself to accelerate
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

            # calculates the acceleration
            accx = calc_acc(accx, x_displacement, other[1].mass, dist)
            accy = calc_acc(accy, y_displacement, other[1].mass, dist)
    return accx, accy


# sets the parameters for the calculation of accleration
def compute_acc(planet):
    # sets relevant variables to 0
    accx = 0
    accy = 0

    potential_energy = 0

    for other in planets:
        # ensuring the planet isn't using itself to accelerate
        if planet.name == other.name:
            pass

        elif planet.name == "Sun":
            accx = 0
            accy = 0

        else:
            # calculates the distance between two objects
            dist = distance(planet.x_pos, planet.y_pos, other.x_pos, other.y_pos)

            # calculates the change in x and y
            x_displacement = other.x_pos - planet.x_pos
            y_displacement = other.y_pos - planet.y_pos

            # calculates the acceleration in x and y direction
            accx = calc_acc(accx, x_displacement, other.mass, dist)
            accy = calc_acc(accy, y_displacement, other.mass, dist)

            # computes the total energy of the system at a given time
            potential_energy = potential_energy + (-gravitational_constant * planet.mass * other.mass) / (dist * planet.instances)

    velocity_squared = planet.x_vel ** 2 + planet.y_vel ** 2

    kinetic_energy = 0.5 * planet.mass * (velocity_squared / planet.instances)

    return accx, accy, kinetic_energy, potential_energy


# calculates the derivatives used when using the different schemes
def derive(planet, derivative, time_step_1):
    # intermediate will be overwritten
    intermediate = [0, 0, 0, 0]

    intermediate[0] = planet.x_pos + derivative[0] * time_step_1
    intermediate[1] = planet.y_pos + derivative[1] * time_step_1
    intermediate[2] = planet.x_vel + derivative[2] * time_step_1
    intermediate[3] = planet.y_vel + derivative[3] * time_step_1

    x_acc, y_acc = get_acc(planet, intermediate)

    return [intermediate[2], intermediate[3], x_acc, y_acc]


def calc_rk4(k1,k2,k3,k4, time_step):
    return 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4) * time_step


def run_step(event):

    planet = event
    time_step = planet.time_step

    # computes the x and y accleration
    x_acc, y_acc, k_energy, p_energy = compute_acc(planet)
    planet.prev_vel = planet.vel

    global ch
    if planet.name == "Sun" and ch:
        planet.x_pos = planet.x_pos + 1
        ch = False
    else:
        # calculates the values for each k
        k1 = [planet.x_vel, planet.y_vel, x_acc, y_acc]
        k2 = derive(planet, k1, time_step * 0.5)
        k3 = derive(planet, k2, time_step * 0.5)
        k4 = derive(planet, k3, time_step)

        # adds it to the previous values multiplying by the time step
        planet.x_pos += calc_rk4(k1[0], k2[0], k3[0], k4[0], time_step)
        planet.y_pos += calc_rk4(k1[1], k2[1], k3[1], k4[1], time_step)
        planet.x_vel += calc_rk4(k1[2], k2[2], k3[2], k4[2], time_step)
        planet.y_vel += calc_rk4(k1[3], k2[3], k3[3], k4[3], time_step)

    planet.vel = planet.x_vel + planet.y_vel
    planet.total_energy = planet.total_energy + (p_energy + k_energy)


# reads the csv file
def read_csv_file(file):
    with open(file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            temp = []
            for y in row:
                temp.append(y)

            # creates a new planet and adds it to the array planets
            planets.append(Planet(x_pos=float(temp[1]), y_pos=float(temp[2]), mass=float(temp[3]), x_vel=float(temp[4]),
                                  y_vel=float(temp[5]), name=temp[0]))


# gives each planet an intitial time step
def initiate_planets(x):
    for planet in planets:
        planet.time_step = x


# plots the graph
def plot(xpos, ypos, dimension):
    # creates figure
    plt.figure("2d Plot")

    # puts limits on the axes
    plt.axes(xlim=(-dimension, dimension), ylim=(-dimension, dimension))

    # plots the points stored in two arrays
    for planet_pos in zip(xpos, ypos):
        plt.plot(planet_pos[0]["x"], planet_pos[1]["y"],
                 label=planet_pos[0]["name"], linewidth = '0.9')

    plt.legend()


# this is used for the animation and gets the points we need to plot in real time
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
    anim_speed = 20

    for place, planet in enumerate(zip(planet_x_positions, planet_y_positions)):

        # gets the current values at position i * animation speed (otherwise too slow)
        try:
            new_x.append(planet[0]["x"][10000+j*anim_speed])
            new_y.append(planet[1]["y"][10000+j*anim_speed])
        except:
            pass

    return new_x, new_y


# calculates the number of events that have a certain planet
def num_Of_planets(name):
    k = 0

    for event in events:
        if event.name == name:
            k += 1
    return k


if __name__ == "__main__":

    # get data from csv file
    read_csv_file('SolarSystemData.csv')
    initiate_planets(25000)

    # create the arrays storing the x and y positions
    for planet in planets:
        planet_x_positions.append({"x": [], "name": planet.name})
        planet_y_positions.append({"y": [], "name": planet.name})

    # set necessary variables
    dimension = 0
    i = 0
    prev_max = 0
    prev_min = 1000000

    for p in planets:
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

    # set the event list and the current maximum time step
    events = planets.copy()
    max_timestep = events[-1].time_step
    seconds = 0

    # starts timer to measure execution time of program
    start_time = time.time()

    # loops through steps
    while (seconds)/31556952 <=12 or i == 0:

        # loops through events
        for event in events:

            # execute each step one at a time
            run_step(event)

        t=0
        # loops through planet to store positions
        while t < len(planets):
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
            t+=1

        # calculates the energy of the system at each iteration
        if i > 0:
            for e in planets:
                n = num_Of_planets(e.name)
                x = e.total_energy
                energy = energy + x
                e.total_energy = 0

            if i == 1:
                pass
            else:
                total_energy.append(energy)

        energy = 0
        grown = False
        seconds = seconds + max_timestep

        #### the adaptive time step ####
        if i % 100 == 0 and i != 0 and i < 1000:
            grown = False
            increase = False

            # iterates through the planets
            iterate_planets = iter(planets)
            next(iterate_planets)

            for num, planet in enumerate(iterate_planets):

                # checks to see if the change in velocity is large enough to warrant reducing the time step
                if planet.grow is not True and abs(planet.vel/planet.prev_vel) < 0.92 or abs(planet.vel/planet.prev_vel)\
                        > 1.08 and planet.explored is not True:
                    planet.shrink = True
                    planet.instances = planet.instances *2

                # checks to see if the change in velocity is small enough to warrant increasing the time step
                if planet.shrink is not True and (planet.vel / planet.prev_vel) > 0.99999 and \
                        abs(planet.vel / planet.prev_vel) < 1.000001:
                    # does the max time step need increasing
                    if planet.time_step == max_timestep:
                        grown = True
                        planet.grown_larger = True

                    else:
                        planet.instances = planet.instances / 2
                    planet.grow = True
                    planet.explored = True

            # updates the amount of instances
            for plan in planets:
                if grown and plan.grown_larger is not True:
                    plan.instances = plan.instances * 2

                elif plan.grown_larger is True:
                    plan.grown_larger = False
                    increase = True
                plan.grow = False

            if increase:
                max_timestep = max_timestep * 2

            events.clear()
            count = 0

            # dynamically updates the list
            for body in planets:
                # ensures Sun has the largest time step
                if body.name == "Sun":
                    body.instances = planets[-1].instances
                # updates the list
                while count < body.instances:
                    body.time_step = max_timestep / body.instances
                    events.append(body)
                    count += 1
                count = 0

        i += 1

    end_time = time.time()

    # prints the time step for each planet
    for e in events:
        print(e.name, e.time_step)

    # prints the relevant information
    print("Time Taken = ", end_time - start_time)
    print("Percentage change in Energy = ", ((max(total_energy) - min(total_energy))*100)/ -1.9816601e+35)

    # increases the dimension so the planet isnt at the border
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

    # prints the graph for energy
    plt.figure("graph for energy")
    plt.scatter(np.linspace(0, len(total_energy),len(total_energy)), total_energy)
    plt.show()

