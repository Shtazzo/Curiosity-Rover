import random as rnd
import numpy as np
from matplotlib import pyplot as plt, animation as animation_mtpl
import math


def generate_rocks(num_rocks):
    coord_x, coord_y = [], []
    water_traces = []

    for i in range(1, num_rocks):
        coord_x.append(rnd.randint(-num_rocks, num_rocks))
        coord_y.append(rnd.randint(-num_rocks, num_rocks))
        water_traces.append(rnd.randint(0, 1))

    return coord_x, coord_y, water_traces


def rover_goals(coord_x=None, coord_y=None, water_traces=None):
    if coord_x is None:
        coord_x = []
    if coord_y is None:
        coord_y = []
    if water_traces is None:
        water_traces = []

    goals, all_goals = [], []

    for i in range(0, len(coord_x)):
        distance = int(math.sqrt(pow(coord_x[i], 2) + pow(coord_y[i], 2)))
        all_goals.append([distance, [coord_x[i], coord_y[i], water_traces[i]]])

    all_goals.sort()
    for i in range(0, len(all_goals)):
        goals.append(all_goals[i][1])

    return goals


def go_ahead(start_x=0, start_y=0, final_y=0, pose=None):
    if pose is None:
        pose = []

    for i in range(0, abs(int(final_y - start_y))):
        start_y = start_y + 1
        pose.append([start_x, start_y])
    return pose


def go_back(start_x=0, start_y=0, final_y=0, pose=None):
    if pose is None:
        pose = []

    for i in range(0, abs(int(final_y - start_y))):
        start_y = start_y - 1
        pose.append([start_x, start_y])
    return pose


def turn_right(start_x=0, start_y=0, final_x=0, pose=None):
    if pose is None:
        pose = []

    for i in range(0, abs(int(final_x - start_x))):
        start_x = start_x + 1
        pose.append([start_x, start_y])
    return pose


def turn_left(start_x=0, start_y=0, final_x=0, pose=None):
    if pose is None:
        pose = []

    for i in range(0, abs(int(final_x - start_x))):
        start_x = start_x - 1
        pose.append([start_x, start_y])
    return pose


def hold_position(start_x=0, start_y=0):
    return [start_x, start_y]


def move_to_goal(obj=None, pose=None):
    if obj is None:
        obj = []
    if pose is None:
        pose = []

    for i in range(0, len(obj)):
        if obj[i][2] == 1:
            if obj[i][0] >= pose[-1][0] and obj[i][1] >= pose[-1][1]:

                if obj[i][0] == pose[-1][0] and obj[i][1] == pose[-1][1]:
                    pose = hold_position(pose[-1][0], pose[-1][1])

                elif obj[i][0] == pose[-1][0] and obj[i][1] > pose[-1][1]:
                    pose = go_ahead(pose[-1][0], pose[-1][1], obj[i][1], pose)

                elif obj[i][0] > pose[-1][0] and obj[i][1] == pose[-1][1]:
                    pose = turn_right(pose[-1][0], pose[-1][1], obj[i][0], pose)

                else:
                    pose = turn_right(pose[-1][0], pose[-1][1], obj[i][0], pose)
                    pose = go_ahead(pose[-1][0], pose[-1][1], obj[i][1], pose)

            elif obj[i][0] >= pose[-1][0] and pose[-1][1] > obj[i][1]:
                if obj[i][0] == pose[-1][0]:
                    pose = go_back(pose[-1][0], pose[-1][1], obj[i][1], pose)

                else:
                    pose = turn_right(pose[-1][0], pose[-1][1], obj[i][0], pose)
                    pose = go_back(pose[-1][0], pose[-1][1], obj[i][1], pose)

            elif obj[i][0] < pose[-1][0] and pose[-1][1] <= obj[i][1]:
                if obj[i][1] == 0:
                    pose = turn_left(pose[-1][0], pose[-1][1], obj[i][0], pose)
                else:
                    pose = turn_left(pose[-1][0], pose[-1][1], obj[i][0], pose)
                    pose = go_ahead(pose[-1][0], pose[-1][1], obj[i][1], pose)

            elif obj[i][0] < pose[-1][0] and obj[i][1] < pose[-1][1]:
                pose = turn_left(pose[-1][0], pose[-1][1], obj[i][0], pose)
                pose = go_back(pose[-1][0], pose[-1][1], obj[i][1], pose)
        else:
            if obj[i][0] >= pose[-1][0] and obj[i][1] >= pose[-1][1]:

                if obj[i][0] == pose[-1][0] and obj[i][1] == pose[-1][1]:
                    pose = hold_position(pose[-1][0], pose[-1][1])

                elif obj[i][0] == pose[-1][0] and obj[i][1] > pose[-1][1]:
                    pose = go_ahead(pose[-1][0], pose[-1][1], obj[i][1], pose)

                elif obj[i][0] > pose[-1][0] and obj[i][1] == pose[-1][1]:
                    pose = turn_right(pose[-1][0], pose[-1][1], obj[i][0], pose)

                else:
                    pose = turn_right(pose[-1][0], pose[-1][1], obj[i][0], pose)
                    pose = go_ahead(pose[-1][0], pose[-1][1], obj[i][1], pose)

            elif obj[i][0] >= pose[-1][0] and pose[-1][1] > obj[i][1]:
                if obj[i][0] == pose[-1][0]:
                    pose = go_back(pose[-1][0], pose[-1][1], obj[i][1], pose)

                else:
                    pose = turn_right(pose[-1][0], pose[-1][1], obj[i][0], pose)
                    pose = go_back(pose[-1][0], pose[-1][1], obj[i][1], pose)

            elif obj[i][0] < pose[-1][0] and pose[-1][1] <= obj[i][1]:
                if obj[i][1] == 0:
                    pose = turn_left(pose[-1][0], pose[-1][1], obj[i][0], pose)
                else:
                    pose = turn_left(pose[-1][0], pose[-1][1], obj[i][0], pose)
                    pose = go_ahead(pose[-1][0], pose[-1][1], obj[i][1], pose)

            elif obj[i][0] < pose[-1][0] and obj[i][1] < pose[-1][1]:
                pose = turn_left(pose[-1][0], pose[-1][1], obj[i][0], pose)
                pose = go_back(pose[-1][0], pose[-1][1], obj[i][1], pose)
    return pose


def back_to_base(pose=None):
    if pose is None:
        pose = []

    if pose[-1][0] >= 0 and pose[-1][1] >= 0:
        if pose[-1][0] == 0 and pose[-1][1] == 0:
            pose = hold_position(pose[-1][0], pose[-1][1])

        elif pose[-1][0] == 0 and pose[-1][1] > 0:
            pose = go_back(pose[-1][0], pose[-1][1], 0, pose)

        elif pose[-1][0] > 0 and pose[-1][1] == 0:
            pose = turn_left(pose[-1][0], pose[-1][1], 0, pose)

        else:
            pose = turn_left(pose[-1][0], pose[-1][1], 0, pose)
            pose = go_back(pose[-1][0], pose[-1][1], 0, pose)

    elif pose[-1][0] >= 0 > pose[-1][1]:
        if pose[-1][0] == 0:
            pose = go_ahead(pose[-1][0], pose[-1][1], 0, pose)

        else:
            pose = turn_left(pose[-1][0], pose[-1][1], 0, pose)
            pose = go_ahead(pose[-1][0], pose[-1][1], 0, pose)

    elif pose[-1][0] < 0 <= pose[-1][1]:
        if pose[-1][1] == 0:
            pose = turn_right(pose[-1][0], pose[-1][1], 0, pose)
        else:
            pose = turn_right(pose[-1][0], pose[-1][1], 0, pose)
            pose = go_back(pose[-1][0], pose[-1][1], 0, pose)

    elif pose[-1][0] < 0 and pose[-1][1] < 0:
        pose = turn_right(pose[-1][0], pose[-1][1], 0, pose)
        pose = go_ahead(pose[-1][0], pose[-1][1], 0, pose)

    return pose


def rover_path(obj=None):
    if obj is None:
        obj = []

    pose = [[0, 0]]
    pose = move_to_goal(obj, pose)
    pose = back_to_base(pose)

    return pose


def rocks_map(goals=None, movements=None, num_rocks=0):
    if goals is None:
        goals = []
    if movements is None:
        movements = []

    r_a_x = [goals[i][0] for i in range(0, len(goals)) if goals[i][2] == 1]
    r_a_y = [goals[i][1] for i in range(0, len(goals)) if goals[i][2] == 1]
    r_v_x = [goals[i][0] for i in range(0, len(goals)) if goals[i][2] == 0]
    r_v_y = [goals[i][1] for i in range(0, len(goals)) if goals[i][2] == 0]
    m_a_x = [movements[i][0] for i in range(0, len(movements))]
    m_a_y = [movements[i][1] for i in range(0, len(movements))]

    def generate_map():

        x = np.arange(-num_rocks - 5, num_rocks + 5, 1)
        y = np.arange(-num_rocks - 5, num_rocks + 5, 1)

        X, Y = np.meshgrid(x, y)
        Z = np.random.normal(0, 1, size=[(num_rocks + 5) * 2, (num_rocks + 5) * 2])

        print(Z)

        return X, Y, Z,

    def init():
        ax.set_title("Rover - {0} rocks on {1}x{2} grid".format(len(goals), num_rocks, num_rocks))
        ax.text(0, 0, 'Base ({0},{1})'.format(0, 0))
        ax.set_xlim(-num_rocks - 5, num_rocks + 5)
        ax.set_ylim(-num_rocks - 5, num_rocks + 5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc='lower right')
        return line,

    def animate(i):
        line.set_data(movements[i][0], movements[i][1])
        for l in range(len(goals)):
            if movements[i] == [goals[l][0], goals[l][1]] and goals[l][2] == 1:
                for k in range(len(r_a_x)):
                    if movements[i] == [r_a_x[k], r_a_y[k]]:
                        del r_a_x[k]
                        del r_a_y[k]
                        break
        r_a_scatter.set_offsets(np.c_[r_a_x, r_a_y])
        return line, r_a_scatter

    X, Y, Z = generate_map()

    fig, ax = plt.subplots(num='My Curiosity Rover - 2D Rover Simulation', figsize=(15, 8), dpi=80, facecolor='w', edgecolor='w')
    #fig.canvas.set_window_title('My Curiosity Rover - 2D Rover Simulation')

    ax.contourf(X, Y, Z, cmap='YlOrRd', alpha=0.8, origin='lower')
    ax.contour(X, Y, Z, colors='k', alpha=0.2, origin='lower')

    r_a_scatter = ax.scatter(r_a_x, r_a_y, s=30, c="blue", marker="^", label='Rocks with Water Trace')
    ax.scatter(r_v_x, r_v_y, s=30, c="black", marker="d", label='Empty Rocks')
    line, = plt.plot([], [], 'ro', animated=True, label='Rover')

    animation = animation_mtpl.FuncAnimation(fig, animate, init_func=init, frames=np.arange(0, len(movements)),
                                             interval=20, blit=True)

    animation.save(f'./my_curiosity_rover_simulation_{num_rocks - 1}_{num_rocks}_X_{num_rocks}.gif', fps=35,
                   writer='pillow')
    plt.show()


def simulation():
    # Generate randomly a certain number of rocks to be explored
    num_rocks = rnd.randint(10, 20)
    coord_x, coord_y, water_trace = generate_rocks(num_rocks)

    # Create list of goals to explore from nearest to farthest based on distance from base (0,0)
    goals = rover_goals(coord_x, coord_y, water_trace)

    # Generate rover movements to achieve all goals
    movements = rover_path(goals)

    # Creo mappa rocce
    rocks_map(goals, movements, num_rocks)


if __name__ == '__main__':
    simulation()
