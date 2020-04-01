# import the necessary libraries
import math
import random
import matplotlib.pyplot as plt
import csv
import matplotlib.animation as animation
import numpy as np
import time


def euler(x, f, h):         # deines how the euler method works
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


# gravitational constant used to find aceleration
gravitational_constant = 6.67e-11

# Allows us to set the number of steps and the amount of each time step skips
n_steps = 5000
energy = 0
nexttimestep = 20000
# defines arrays needed to be global
planet_x_positions = []
planet_y_positions = []
planets = []
tot_x = []
tot_y = []
total_energy = []
events = []
events_orig = []

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
        self.instances = 1
        self.shrink = None
        self.grow = None
        self.explored = None


class Event:
    def __init__(self, planet, time_step):
        self.planet = planet
        self.time_step = time_step

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

    for other in planets:
        if planet.name == other.name:           # ensuring the planet isnt using itself to accelerate
            pass

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
            potential_energy =  potential_energy + (-gravitational_constant * planet.mass * other.mass) / dist

            velocity_squared = planet.x_vel ** 2 + planet.y_vel ** 2

            kinetic_energy = kinetic_energy +  0.5 * planet.mass * velocity_squared

            planet.k_energy = kinetic_energy
            planet.p_energy = potential_energy
    return accx, accy
####################################
def run_step(event):
    planet = event.planet
    act_time_step = event.time_step

    # computes the x and y accleration
    x_acc, y_acc = compute_acc(planet)
    planet.prev_vel = planet.vel

    # calculates the velocity by differentiating the acceleration
    planet.x_vel = RK4(planet.x_vel, lambda x: x_acc, act_time_step)
    planet.y_vel = RK4(planet.y_vel, lambda y: y_acc, act_time_step)

    # calculates the displacment by differentiating the velocity
    planet.x_pos = RK4(planet.x_pos, lambda x: planet.x_vel, act_time_step)
    planet.y_pos = RK4(planet.y_pos, lambda y: planet.y_vel, act_time_step)

    planet.vel = planet.x_vel +planet.y_vel




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


def read_events_file(x):               # reads the events csv file
    with open(x) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            temp = []
            for y in row:
                temp.append(y)

            for planet in planets:
                if planet.name == temp[0]:
                    events_orig.append(Event(planet=planet, time_step=float(temp[1])))

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


def num_Of_planets(name):
    i=0
    for event in events:
        if event.planet.name == name:
            i = i + 1
    return i


if __name__ == "__main__":
    # get data from csv file
    read_csv_file('data.csv')
    read_events_file('events.csv')

    # create the arrays storing the x and y positions
    for planet in planets:
        planet_x_positions.append({"x": [], "name": planet.name})
        planet_y_positions.append({"y": [], "name": planet.name})

    # set necessary variables
    dimension = 0
    i = 0
    prev_max = 0
    prev_min = 1000000
    k_energy = 0
    p_energy = 0

    start_time = time.time()

    # execute each step one at a time
    #while planets[-1].x_pos > 0 or i == 0:
    events = events_orig
    max_timestep = events[0].time_step
    #while i<20000:
    while planets[-3].x_pos > 0 or i == 0:
        for event in events:
            run_step(event)

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

        for e in planets:
            k_energy = k_energy + e.k_energy
            p_energy = p_energy + e.p_energy

        energy = 2*k_energy / abs(p_energy)
        total_energy.append(energy)

        if energy > prev_max:
            prev_max = energy

        elif energy < prev_min:
            prev_min = energy

        energy = 0
        k_energy = 0
        p_energy = 0
        # the adaptive time step
        if i % 100 == 0:
            print(i)
        if i % 100 == 0 and i!=0 and i<1000:
            iterate_planets = iter(planets)
            next(iterate_planets)

            checked = False
            # iterates through the planets
            for num, planet in enumerate(iterate_planets):


                # checks to see if the change in velocity is small enough to warant reducing the time step
                if planet.grow != True and abs(planet.vel/planet.prev_vel) < 0.999 or abs(planet.vel/planet.prev_vel) > 1.001:
                    planet.shrink = True
                    planet.instances = planet.instances *2
                    if planet.name == "Neptune":
                        print ("Oh OH spagetti oos")



                if checked == False and planet.shrink != True and (planet.vel / planet.prev_vel) > 0.99995 and abs(planet.vel / planet.prev_vel) < 1.0005:# and distance(0,0,planet.x_pos,planet.y_pos)>5e11:
                    checked = True
                    max_timestep = max_timestep*2
                    for pos, plan in enumerate(planets):
                        if pos<num:
                            plan.instances = plan.instances * 2
                        elif pos>=num:
                            plan.grow = True
                            print(plan.name)



        events.clear()
        count = 0

        for body in planets:
            while count<body.instances:
                events.append(Event(body,max_timestep / body.instances))
                count+=1
            count = 0



        i = i+1
    for e in events:
        print(e.planet.name, e.time_step)
    end_time = time.time()
    print(end_time - start_time)




    print (prev_max - prev_min)
    # increases the dimension so the planet isnt at the border
    dimension = dimension + dimension * 0.1
    plot(planet_x_positions, planet_y_positions, dimension)

    '''# creates the snaimation
    fig = plt.figure("ANIMATION")
    ax = plt.axes(xlim=(-dimension, dimension), ylim=(-dimension, dimension))
    ax.set_xlabel('X Axis', size=12)
    ax.set_ylabel('Y Axis', size=12)

    x = []
    y = []

    scatter = ax.scatter(x, y)

    # updates the animation in real time
    ani = animation.FuncAnimation(fig, update, frames=100000, interval=1)
    plt.show()'''

    plt.figure("graph for energy")
    plt.scatter(np.linspace(0, len(total_energy[1000:-1]),len(total_energy[1000:-1])), total_energy[1000:-1])
    plt.show()

