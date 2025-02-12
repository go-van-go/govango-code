import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import os

def find_spring_params():
    """
    Given noisy measurements of y, the solution of a damped spring system
     m * d^2y/dt^2 + c * dy/dt + k * y = 0, with y(0)=2, dy/dt(0)=0.
     From this, estimate c, k
    """
    
    # Parameters
    n = 15000       # Number of runs 
    m = 2.0         # mass
    k = 4.10        # true spring constant
    c = 0.50        # true damping factor
    k_guess = 4.30     # spring constant guess
    c_guess = 0.45  # damping factor guess
    
    # Setup timestepping 
    t_max = 10
    steps_per_unit_t = 100
    num_t_steps = t_max * steps_per_unit_t + 1
    t = np.linspace(0, t_max, num_t_steps)
    dt = t[1] - t[0]

    # Create folder to store plots
    os.makedirs("mcmc_images", exist_ok=True)
    
    # Create synthetic data
    s = 0.1  # Noise level
    walk_size_k = 4 * s  # Random walk step size
    walk_size_c = 0.2 * s  # Random walk step size
    num_samples = 50
    # sample 50 time points
    time_samples = np.linspace(1, num_t_steps-1, num_samples, dtype=int)
    time_of_samples = (time_samples - 1) / num_t_steps * t_max 
    # find true solution
    y_true = 2 * np.exp(-c * t / (2 * m)) * np.cos(t * np.sqrt(4 * m * k - c * c) / (2 * m))
    # get only sampled points 
    y_true_sampled = y_true[time_samples]
    # add Gaussian noise to create synthetic data
    y_synthetic = y_true_sampled + s * np.random.randn(num_samples)
   
    # initialze arrays to keep track of likelihood
    likelihood = np.zeros(n)
    initial_likelihood = calculate_likelihood(y_synthetic, y_true_sampled, s)
    likelihood[0] = initial_likelihood
    old_likelihood = initial_likelihood

    # array of spring parameter guesses
    inputs = np.zeros((2, n))
    inputs[:, 0] = [k_guess, c_guess]

    # Run Metropolis-Hastings iteration
    for current_run in range(1, n):
        # propose next guesses for k and c
        kp = abs(k_guess + walk_size_k * (2 * np.random.rand() - 1))
        cp = abs(c_guess + walk_size_c * (2 * np.random.rand() - 1))

        # solve simulation with proposed values for k and c
        y_initial = 2.0
        y_dot_initial = 0.0
        y_proposed = damped_oscillator_solver(m, kp, cp,
                                              y_initial,
                                              y_dot_initial,
                                              dt, num_t_steps)
        
        # get values at selected time samples
        y_proposed_sampled = y_proposed[time_samples]

        # calculate the likelihood of the new data
        new_likelihood = calculate_likelihood(y_synthetic, y_proposed_sampled, s)

        ratio = np.exp(-(new_likelihood - old_likelihood) / 2)
        # rejection criteria 
        acceptance_ratio = min(1, ratio)

        # accept new k and c values
        if np.random.rand() < acceptance_ratio:
            k_guess, c_guess = kp, cp
            old_likelihood = new_likelihood

        # update guesses and likelihood
        inputs[:, current_run] = [k_guess, c_guess]
        likelihood[current_run] = old_likelihood

        if current_run < 200:
            plot_data(current_run, inputs, y_true, y_synthetic, y_proposed, t, time_of_samples, m, current_run)
        elif current_run < 2000 and current_run%5==0:
            plot_data(current_run, inputs, y_true, y_synthetic, y_proposed, t, time_of_samples, m, current_run)
        elif current_run > 2000 and current_run%10==0:
            plot_data(current_run, inputs, y_true, y_synthetic, y_proposed, t, time_of_samples, m, current_run)

    plot_k_vs_c(inputs)
    plot_histograms(inputs)
    plot_best(inputs, y_true, k_guess, c_guess, m, t, time_of_samples, y_synthetic)

def calculate_likelihood(y_synthetic, y_proposed, s):
    return np.sum((y_synthetic - y_proposed) ** 2)

def damped_oscillator_solver(m, k, c, y_initial, y_dot_initial, dt, num_t_steps):
    # initialize y and y dot
    y = np.zeros(num_t_steps)
    y_dot = np.zeros(num_t_steps)

    # set initial conditions
    y[0] = y_initial
    y_dot[0] = y_dot_initial

    # compute half time step for midpoint rule 
    ddt = 0.5 * dt
    
    # time loop using midpoint rule
    for t_current in range(1, num_t_steps):
        t_prev = t_current - 1  # previous time index

        # compute midpoint estimate for y and y dot
        y_mid = y[t_prev] + ddt * y_dot[t_prev]
        y_dot_mid = y_dot[t_prev] - (c/m) * ddt * y_dot[t_prev] - (k/m) * ddt * y[t_prev]

        # update y with y_dot at midpoint
        y[t_current] = y[t_prev] + dt * y_dot_mid
        # update y_dot with acceleration at midpoint
        y_dot[t_current] = y_dot[t_prev] - (c/m) * dt * y_dot_mid - (k/m) * dt * y_mid
    
    return y

def plot_k_vs_c(inputs):
    # Plot results
    k_data = inputs[0, :]
    c_data = inputs[1, :]
    
    plt.figure()
    plt.scatter(k_data, c_data, color='b')
    plt.title('Scatterplot of k vs. c')
    plt.show()
    

def plot_histograms(inputs):
    k_data = inputs[0, :]
    c_data = inputs[1, :]
    bins = 50
    edge_color = "#eceff4"
 
    plt.figure()
    plt.hist(k_data, bins=bins, color="#5e81ac", edgecolor=edge_color)
    plt.title(f'Histogram of k values')
    plt.show()
    
    plt.figure()
    plt.hist(c_data, bins=bins, color="#bf616a", edgecolor=edge_color)
    plt.title(f'Histogram of c values')
    plt.show()

       
def plot_best(inputs, y_true, k_guess, c_guess, m, t, time_of_samples, y_synthetic):
    plt.figure(figsize=(10, 8))
   
    # best guess
    k_data = inputs[0, :]
    c_data = inputs[1, :]
    k = mode(np.round(k_data, decimals=2))[0]
    c = mode(np.round(c_data, decimals=2))[0]
    y_best_guess = 2 * np.exp(-c * t / (2 * m)) * np.cos(t * np.sqrt(4 * m * k - c * c) / (2 * m))

    # first guess
    k = k_guess
    c = c_guess
    y_first= 2 * np.exp(-c * t / (2 * m)) * np.cos(t * np.sqrt(4 * m * k - c * c) / (2 * m))

    plt.plot(t, y_true, 'r', label='True Solution')
    plt.plot(t, y_best_guess, 'k', label='Best Guess')
    plt.plot(t, y_first, 'g', label='Initial Estimate')
    plt.scatter(time_of_samples, y_synthetic, color='b', marker='o', label='Noisy Measurements')
    plt.legend()
    plt.title('Current y_proposed vs yTrue')
   
    # Save the figure
    plt.tight_layout()
    plt.show()
    plt.savefig(f'best_guess.png')
    plt.close()


def plot_data(current_run, inputs, y_true, y_synthetic, y_proposed, t, time_of_samples, m, n):
    # --- Plotting settings ---
    edge_color = "#eceff4"
    font_color = "#eceff4"
    bg_color = "#4c566a"
    axis_color = "#eceff4"
    border_color = "#eceff4"
    bins = 50
    
    # Create figure and subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), facecolor=bg_color)
    fig.patch.set_facecolor(bg_color)

    # Add a supertitle for the figure
    fig.suptitle(f"Run {n}", x=0.511, color=font_color, fontsize=16)
    
    # Apply common settings to every subplot
    for ax in axs.flat:
        ax.tick_params(axis='both', colors=axis_color)
        for spine in ax.spines.values():
            spine.set_color(border_color)
        ax.set_facecolor(bg_color)
        ax.xaxis.label.set_color(font_color)
        ax.yaxis.label.set_color(font_color)
        ax.title.set_color(font_color)

    # Scatter plot of k vs. c
    axs[0, 1].scatter(inputs[0, :current_run], inputs[1, :current_run], color=font_color, alpha=0.2)
    axs[0, 1].set_xlabel('k', color=font_color)
    axs[0, 1].set_ylabel('c', color=font_color)
    axs[0, 1].set_title('Scatter plot of k vs. c', color=font_color)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    
    # y_proposed vs yTrue
    axs[1, 0].plot(t, y_true, label='True Solution', color="#b48ead")
    axs[1, 0].plot(t, y_proposed, label='Current Guess', color="#ebcb8b")
    axs[1, 0].scatter(time_of_samples, y_synthetic, color="#b48ead", marker='.', label='Measurements')
    axs[1, 0].legend(facecolor=bg_color, labelcolor=font_color, loc="upper right")
    axs[1, 0].set_title('Current y_proposed vs y_true', color=font_color)
    axs[1, 0].set_xlabel('time', color=font_color)
    axs[1, 0].set_ylim(-2.1, 2.5)
    axs[1, 0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axs[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
   
    # Histogram of k values
    axs[1, 1].hist(inputs[0, :current_run], bins=bins, color="#5e81ac", edgecolor=edge_color)
    axs[1, 1].set_xlabel('k values', color=font_color)
    axs[1, 1].set_title('Histogram of k', color=font_color)
    axs[1, 1].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    axs[1, 1].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    # Histogram of c values
    axs[0, 0].hist(inputs[1, :current_run], bins=bins, orientation="horizontal", color="#bf616a", edgecolor=edge_color)
    axs[0, 0].set_ylabel('c values', color=font_color)
    axs[0, 0].set_title('Histogram of c', color=font_color)
    axs[0, 0].xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    axs[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
  
    # Save the figure
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(f'mcmc_images/frame_{current_run:06d}.png')
    plt.close()

    
# Run the function
find_spring_params()
