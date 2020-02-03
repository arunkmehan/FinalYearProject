
import math
import random
import matplotlib.pyplot as plot
import csv

def Euler(x, f, h):
    return x+f(x)*h


def modified_euler(x,f,h):
    K1 = h*f(x)
    K2 = h*f(x+K1)
    return x+0.5*(K1+K2)


def RK4(x,f,h):
    K1 = h*f(x)
    K2 = h*f(x + 0.5*K1)
    K3 = h*f(x + 0.5*K2)
    K4 = h*f(x + K3)
    return x + (1/6)*(K1 + 2*K2 + 2*K3 + K4)


gravitational_constant = 6.67e-11
n_steps = 1000000
time_step = 100
planet_positions = []
planets = []


class body:
    def __init__(self, name,  mass, x_pos, y_pos, x_vel, y_vel ):
        self.name = name
        self.mass = mass
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_vel = x_vel
        self.y_vel = y_vel


def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


def compute_acc(planet):
    dist = 0
    planet[1].x_acc = 0
    planet[1].x_acc = 0
    accx = 0
    accy=0

    for other in enumerate(planets):
        if planet[0] == other[0]:
            pass

        else:
            dist = distance(planet[1].x_pos, planet[1].y_pos, other[1].x_pos, other[1].y_pos)

            x_displacement = other[1].x_pos - planet[1].x_pos
            y_displacement = other[1].y_pos - planet[1].y_pos

            accx =  accx + (
                        gravitational_constant * x_displacement * other[1].mass) / dist**3

            accy = accy + (
                        gravitational_constant * y_displacement * other[1].mass) / dist ** 3
    return accx, accy


def compute_velocity(planets):
    for planet in enumerate(planets):
        x_acc, y_acc = compute_acc(planet)
        planet[1].x_vel = RK4(planet[1].x_vel, lambda x: x_acc, time_step)
        planet[1].y_vel = RK4(planet[1].y_vel, lambda y: y_acc, time_step)


def update_location(planets):
    for target_body in planets:
        target_body.x_pos = RK4(target_body.x_pos, lambda x: target_body.x_vel, time_step)
        target_body.y_pos = RK4(target_body.y_pos, lambda y: target_body.y_vel, time_step)


def plot_output(planets, outfile=None):
    fig = plot.figure()
    colours = ['r', 'b', 'g', 'y', 'm', 'c']
    ax = fig.add_subplot(1, 1, 1)
    max_range = 0
    for current_body in planets:
        #print(current_body)
        max_dim = max(max(current_body["x"]), max(current_body["y"]))
        if max_dim > max_range:
            max_range = max_dim
        ax.plot(current_body["x"], current_body["y"], c=random.choice(colours),
                label=current_body["name"])

    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.legend()

    plot.show()

def run_step(planets):
    compute_velocity(planets)
    update_location(planets)


if __name__ == "__main__":
    with open('data.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        for row in csv_reader:
            temp = []
            for y in row:
                temp.append(y)

            planets.append(body(x_pos=float(temp[1]), y_pos=float(temp[2]), mass=float(temp[3]), x_vel=float(temp[4]),
                                y_vel=float(temp[5]), name=temp[0]))

    for planet in planets:
        planet_positions.append({"x": [], "y": [], "name": planet.name})

    i=0
    while i < n_steps:
        run_step(planets)

        for planet_number in enumerate(planet_positions):
            planet_number[1]["x"].append(planets[planet_number[0]].x_pos)
            planet_number[1]["y"].append(planets[planet_number[0]].y_pos)
        i = i+1

    plot_output(planet_positions)