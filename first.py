import math
import random
import matplotlib.pyplot as plot


class Planet:
    def __init__(self, name,  mass, x_pos, y_pos, x_vel, y_vel, x_acc=0, y_acc=0 ):
        self.name = name
        self.mass = mass
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.x_vel = x_vel
        self.y_vel = y_vel
        self.x_acc = 0
        self.y_acc = 0


gravitational_constant = 6.67e-11
n_steps = 100000
time_step = 100
planet_positions = []

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


def distance(x1, y1, x2, y2):
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist


def compute_acc(planet_number):
    dist = 0
    #planet_number[1].x_acc = 0
    #planet_number[1].x_acc = 0
    accx = 0
    accy=0


    for other in enumerate(planets):
        if planet_number[0] == other[0]:
            pass

        else:
            dist = distance(planet_number[1].x_pos, planet_number[1].y_pos,
                                other[1].x_pos, other[1].y_pos)
            x_displacement = other[1].x_pos - planet_number[1].x_pos
            y_displacement = other[1].y_pos - planet_number[1].y_pos

            """planet_number[1].x_acc =  planet_number[1].x_acc + (
                        gravitational_constant * x_displacement * other[1].mass) / dist**3

            planet_number[1].y_acc = planet_number[1].y_acc + (
                        gravitational_constant * y_displacement * other[1].mass) / dist ** 3"""

            print(other[1].mass)

            tmp = gravitational_constant * other[1].mass / dist ** 3
            print(tmp)

            accx = accx + tmp * x_displacement
            accy = accy + tmp * y_displacement

            "accy = accy + (gravitational_constant * y_displacement * other[1].mass) / dist ** 3"

    return accx, accy

def calculate_single_body_acceleration(bodies, body_index):
    G_const = 6.67408e-11 #m3 kg-1 s-2
    accx = 0
    accy = 0
    target_body = bodies[body_index]
    for index, external_body in enumerate(bodies):
        if index != body_index:
            r = (target_body.x_pos - external_body.x_pos)**2 + (target_body.y_pos - external_body.y_pos)**2
            r = math.sqrt(r)
            print(r**3)
            tmp = G_const * external_body.mass / r**3
           # print(tmp)
            accx += tmp * (external_body.x_pos - target_body.x_pos)
            accy += tmp * (external_body.y_pos - target_body.y_pos)

    return accx, accx

def compute_velocity(planets):
    for planet_number in enumerate(planets):
        x,y = calculate_single_body_acceleration(planets, planet_number[0])
        """planet_number[1].x_vel = RK4(planet_number[1].x_vel, lambda x:  planet_number[1].x_acc, time_step)
        planet_number[1].y_vel = RK4(planet_number[1].y_vel, lambda y:  planet_number[1].y_acc, time_step)"""

        planet_number[1].x_vel = RK4(planet_number[1].x_vel, lambda x: x, time_step)
        planet_number[1].y_vel = RK4(planet_number[1].y_vel, lambda y: y, time_step)


def update_location(bodies):
    for target_body in bodies:
        target_body.x_pos = RK4(target_body.x_pos, lambda x: target_body.x_vel, time_step)
        target_body.y_pos = RK4(target_body.y_pos, lambda y: target_body.y_vel, time_step)

def run_step(planets):
    compute_velocity(planets)
    update_location(planets)


sun = {"x_pos": 0, "y_pos": 0, "mass": 2e30, "x_vel": 0, "y_vel": 0}
venus = {"x_pos": 0, "y_pos": 1.1e11, "mass": 4.8e24, "x_vel": 35000, "y_vel": 0}
earth = {"x_pos": 0, "y_pos": 1.5e11, "mass": 6e24, "x_vel": 30000, "y_vel": 0}
mars = {"x_pos": 0, "y_pos": 2.2e11, "mass": 2.4e24, "x_vel": 24000, "y_vel": 0}


def plot_output(bodies, outfile=None):
    fig = plot.figure()
    colours = ['r', 'b', 'g', 'y', 'm', 'c']
    ax = fig.add_subplot(1, 1, 1)
    max_range = 0
    for current_body in bodies:
        max_dim = max(max(current_body["x"]), max(current_body["y"]))
        if max_dim > max_range:
            max_range = max_dim
        ax.plot(current_body["x"], current_body["y"], c=random.choice(colours),
                label=current_body["name"])

    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.legend()

    plot.show()


"""if __name__ == "__main__":
    planets = [
        Planet(x_pos=sun["x_pos"], y_pos=sun["y_pos"], mass=sun["mass"], x_vel=sun["x_vel"], y_vel=sun["y_vel"], name="sun"),
        Planet(x_pos=earth["x_pos"], y_pos=earth["y_pos"], mass=earth["mass"], x_vel=earth["x_vel"], y_vel=earth["y_vel"], name="earth"),
        Planet(x_pos=mars["x_pos"], y_pos=mars["y_pos"], mass=mars["mass"], x_vel=mars["x_vel"], y_vel=mars["y_vel"], name="mars"),
        Planet(x_pos=venus["x_pos"], y_pos=venus["y_pos"], mass=venus["mass"], x_vel=venus["x_vel"], y_vel=venus["y_vel"], name="venus"),]

    for planet in planets:
        planet_positions.append({"x": [], "y": [], "name": planet.name})

    i=0
    while i < n_steps:
        run_step(planets)

        for planet_number in enumerate(planet_positions):
            planet_number[1]["x"].append(planets[planet_number[0]].x_pos)
            planet_number[1]["y"].append(planets[planet_number[0]].y_pos)
        i=i+1

    plot_output(planet_positions)"""

def run_simulation(bodies, names=None, time_step=1, number_of_steps=100000, report_freq=10):
    # create output container for each body
    body_locations_hist = []
    for current_body in bodies:
        body_locations_hist.append({"x": [], "y": [], "name": current_body.name})

    for i in range(1, number_of_steps):
        run_step(bodies)

        if i % report_freq == 0:
            for index, body_location in enumerate(body_locations_hist):
                body_location["x"].append(bodies[index].x_pos)
                body_location["y"].append(bodies[index].y_pos)

    return body_locations_hist

if __name__ == "__main__":
    # build list of planets in the simulation, or create your own
    planets = [
        Planet(x_pos=sun["x_pos"], y_pos=sun["y_pos"], mass=sun["mass"], x_vel=sun["x_vel"], y_vel=sun["y_vel"], name="sun"),
        Planet(x_pos=earth["x_pos"], y_pos=earth["y_pos"], mass=earth["mass"], x_vel=earth["x_vel"], y_vel=earth["y_vel"], name="earth"),
        Planet(x_pos=mars["x_pos"], y_pos=mars["y_pos"], mass=mars["mass"], x_vel=mars["x_vel"], y_vel=mars["y_vel"], name="mars"),
        Planet(x_pos=venus["x_pos"], y_pos=venus["y_pos"], mass=venus["mass"], x_vel=venus["x_vel"], y_vel=venus["y_vel"], name="venus"),]

    time_step = 1
    number_of_steps = 100000

    motions = run_simulation(planets, time_step, n_steps, report_freq=10)
    print(motions)
    plot_output(motions, outfile='orbits.png')