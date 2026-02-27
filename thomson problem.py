import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

num_points = 5
learning_rate = 1/num_points
steps = 100

def initialize(num_points):
    points = []
    for i in range(num_points):
        theta = np.random.uniform(0, 2*np.pi)
        phi = np.random.uniform(-np.pi/2, np.pi/2)
        points.append(convert([theta, phi]))
    points = np.array(points)
    return points

def convert(point):
    theta, phi = point
    x0 = np.cos(theta) * np.cos(phi)
    y0 = np.sin(theta) * np.cos(phi)
    z0 = np.sin(phi)
    return np.array([x0, y0, z0])

def force(p1, p2):
    diff = p1 - p2
    r = np.linalg.norm(diff)
    if r == 0:
        return 0
    return diff/(r**3)

def get_forces(points):
    forces = []
    for point in points:
        netforce = np.zeros(3)
        for p in points:
            netforce += force(point, p)
        forces.append(netforce)
    return np.array(forces)

def get_total_energy(points):
    energy = 0
    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):  # Only unique pairs
            r = np.linalg.norm(points[i] - points[j])
            if r > 0:
                energy += 1.0 / r
    return energy

def update_position(points, forces):
    p = []
    for i in range(num_points):
        point = points[i]
        force = forces[i]
        perpendicular_force = force - np.dot(force, point)*point

        point += perpendicular_force * learning_rate
        p.append(point/np.linalg.norm(point))
    return np.array(p)


points = initialize(num_points)
fig = plt.figure(figsize=(15, 7))
ax = fig.add_subplot(121, projection='3d')
ax1 = fig.add_subplot(122)

energy_history = []

u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x_sphere = np.outer(np.cos(u), np.sin(v))
y_sphere = np.outer(np.sin(u), np.sin(v))
z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

for step in range(steps):
    points = update_position(points, get_forces(points))
    current_energy = get_total_energy(points)
    energy_history.append(current_energy)

    ax.clear()
    ax1.clear()


    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='blue', alpha=0.1,
                    rstride=1, cstride=1, edgecolor='white', linewidth=0.3)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
               color='red', s=150, edgecolor='black')

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"Step {step + 1}/{steps}")

    ax1.plot(energy_history, color='green', linewidth=2)
    ax1.set_title(f"Total Energy: {current_energy:.4f}")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Potential Energy")
    ax1.grid(True)

    plt.pause(0.01)

plt.show()