
import math
import random
import matplotlib.pyplot as plot
from mpl_toolkits.mplot3d import Axes3D

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

class point:
    def __init__(self, x,y):
        self.x = x
        self.y = y

class body:
    def __init__(self, location, mass, velocity, name = ""):
        self.location = location
        self.mass = mass
        self.velocity = velocity
        self.name = name


def calculate_single_body_acceleration(bodies, body_index):
    G_const = 6.67408e-11 #m3 kg-1 s-2
    acceleration = point(0,0)
    target_body = bodies[body_index]
    for index, external_body in enumerate(bodies):
        if index != body_index:
            r = (target_body.location.x - external_body.location.x)**2 + (target_body.location.y - external_body.location.y)**2
            r = math.sqrt(r)
            print(r**3)
            tmp = G_const * external_body.mass / r**3
            #print(tmp)
            acceleration.x += tmp * (external_body.location.x - target_body.location.x)
            acceleration.y += tmp * (external_body.location.y - target_body.location.y)

    return acceleration



def compute_velocity(bodies, time_step):
    for body_index, target_body in enumerate(bodies):
        acceleration = calculate_single_body_acceleration(bodies, body_index)
        #print(acceleration.x)
        target_body.velocity.x = RK4(target_body.velocity.x, lambda x: acceleration.x, time_step)
        target_body.velocity.y = RK4(target_body.velocity.y, lambda y: acceleration.y, time_step)




def update_location(bodies, time_step):
    for target_body in bodies:
        target_body.location.x = RK4(target_body.location.x, lambda x: target_body.velocity.x, time_step)
        target_body.location.y = RK4(target_body.location.y, lambda y: target_body.velocity.y, time_step)




def compute_gravity_step(bodies, time_step):
    compute_velocity(bodies, time_step = time_step)
    update_location(bodies, time_step = time_step)


def run_simulation(bodies, names=None, time_step=1, number_of_steps=100000, report_freq=10):
    # create output container for each body
    body_locations_hist = []
    for current_body in bodies:
        body_locations_hist.append({"x": [], "y": [], "name": current_body.name})

    for i in range(1, number_of_steps):
        compute_gravity_step(bodies, time_step=1000)

        if i % report_freq == 0:
            for index, body_location in enumerate(body_locations_hist):
                body_location["x"].append(bodies[index].location.x)
                body_location["y"].append(bodies[index].location.y)

    return body_locations_hist


def plot_output(bodies, outfile=None):
    fig = plot.figure()
    colours = ['r', 'b', 'g', 'y', 'm', 'c']
    ax = fig.add_subplot(1, 1, 1)
    max_range = 0
    for current_body in bodies:
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


# planet data (location (m), mass (kg), velocity (m/s)
sun = {"location": point(0, 0), "mass": 2e30, "velocity": point(0, 0)}
mercury = {"location": point(0, 5.7e10), "mass": 3.285e23, "velocity": point(47000, 0)}
venus = {"location": point(0, 1.1e11), "mass": 4.8e24, "velocity": point(35000, 0)}
earth = {"location": point(0, 1.5e11), "mass": 6e24, "velocity": point(30000, 0)}
mars = {"location": point(0, 2.2e11), "mass": 2.4e24, "velocity": point(24000, 0)}
jupiter = {"location": point(0, 7.7e11), "mass": 1e28, "velocity": point(13000, 0)}
saturn = {"location": point(0, 1.4e12), "mass": 5.7e26, "velocity": point(9000, 0)}
uranus = {"location": point(0, 2.8e12), "mass": 8.7e25, "velocity": point(6835, 0)}
neptune = {"location": point(0, 4.5e12), "mass": 1e26, "velocity": point(5477, 0)}
pluto = {"location": point(0, 3.7e12), "mass": 1.3e22, "velocity": point(4748, 0)}

if __name__ == "__main__":
    # build list of planets in the simulation, or create your own
    bodies = [
        body(location=sun["location"], mass=sun["mass"], velocity=sun["velocity"], name="sun"),
        body(location=earth["location"], mass=earth["mass"], velocity=earth["velocity"], name="earth"),
        body(location=mars["location"], mass=mars["mass"], velocity=mars["velocity"], name="mars"),
        body(location=venus["location"], mass=venus["mass"], velocity=venus["velocity"], name="venus"),

    ]
    time_step = 1
    number_of_steps = 100000

    motions = run_simulation(bodies, time_step, number_of_steps, report_freq=10)
    print(motions)
    plot_output(motions, outfile='orbits.png')
