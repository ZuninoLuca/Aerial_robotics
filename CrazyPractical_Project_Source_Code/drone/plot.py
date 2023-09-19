import numpy as np

def plot_controlled_position_results(ax, desired_position,estimated_position):
    steps = np.arange(1,len(desired_position))

    ax.plot(steps,desired_position[0], label = "x-desired position")
    ax.plot(steps,estimated_position[0], label = "x-estimated position")

    ax.plot(steps,desired_position[1], label = "y-desired position")
    ax.plot(steps,estimated_position[1], label = "y-estimated position")

    ax.plot(steps,desired_position[2], label = "z-desired position")
    ax.plot(steps,estimated_position[2], label = "z-estimated position")

    ax.legend()
    return ax