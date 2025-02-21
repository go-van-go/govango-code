import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.special import kv, gamma
import pdb

class GaussianProcess:
    def __init__(self, X_train, y_train, length_scale=1.0, period=1.0, decay=1.0, sigma_f=1.0, noise=1e-6, nu=1):
        """Initialize the Gaussian Process with kernel parameters."""
        self.length_scale = length_scale
        self.period = period
        self.decay = decay
        self.sigma_f = sigma_f
        self.nu = nu
        self.noise = noise
        self.X_train = X_train
        self.y_train = y_train
        self.K = self.compute_covariance_matrix(self.X_train)
        self.compute_posterior(X_train)
    
    def damped_periodic_kernel(self, x1, x2):
        """Damped harmonic oscillator kernel with variance scaling."""
        diff = np.abs(x1 - x2)
        periodic_component = np.exp(-2 * (np.sin(np.pi * diff / self.period) ** 2) / self.length_scale**2)
        #periodic_component = 1
        decay_component = np.exp(-self.decay * diff)
        #decay_component = 1
        return self.sigma_f**2 * periodic_component * decay_component

    def kernel(self, x1, x2):
        """
        Compute the Matern kernel between two points x1 and x2.
        """
        # Compute the distance between x1 and x2
        diff = np.abs(x1 - x2)
        
        # Handle the case where diff is zero to avoid division by zero
        if np.isscalar(diff) and diff == 0:
            return self.sigma_f**2
        elif isinstance(diff, np.ndarray):
            diff[diff == 0] = 1e-10  # Small value to avoid division by zero
            
        # Compute the Matern kernel
        d_scaled = np.sqrt(2 * self.nu) * diff / self.length_scale
        bessel_term = kv(self.nu, d_scaled)  # Modified Bessel function of the second kind
        gamma_term = gamma(self.nu)  # Gamma function
        prefactor = (2**(1 - self.nu)) / gamma_term
            
        return self.sigma_f**2 * prefactor * (d_scaled**self.nu) * bessel_term


    
    def compute_covariance_matrix(self, X):
        """Compute the covariance matrix using the kernel function with added noise."""
        n = len(X)
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(X[i], X[j])
        
        # Add a small noise term to the diagonal for numerical stability
        #K += self.noise * np.eye(n)*0.001
        
        return K


    def compute_posterior(self, X_test):
        """
        Computes the posterior mean and covariance for a Gaussian Process conditioned on training data.

        Args:
            X_test (numpy array): Test points where predictions are to be made.
        """
        X_train, y_train, K_train = self.X_train, self.y_train, self.K

        # Compute the cross-covariance between training and test points
        K_s = np.array([[self.kernel(x_train, x_test) for x_test in X_test] for x_train in X_train])

        # Compute the covariance matrix for test points
        K_ss = np.array([[self.kernel(x_test1, x_test2) for x_test2 in X_test] for x_test1 in X_test])

        # Compute the inverse of the training covariance matrix
        K_inv = np.linalg.inv(K_train)

        # Compute the posterior mean and store it as self.mu
        self.mu = K_s.T @ K_inv @ y_train

        # Compute the posterior covariance and store it as self.posterior_covariance
        self.posterior_covariance = K_ss - K_s.T @ K_inv @ K_s

    def predict(self, X_test):
        """
        Make predictions using the Gaussian Process.

        Args:
        X_test (numpy array): Test points where predictions are to be made.

        Returns:
        mu (numpy array): Predicted mean at the test points.
        std (numpy array): Standard deviation of the predictions at the test points.
        """
        # Compute the posterior mean and covariance for the test points
        self.compute_posterior(X_test)

        # Extract the mean (mu) and standard deviation (std) from the posterior
        mu = self.mu
        std = np.sqrt(np.diag(self.posterior_covariance))

        return mu, std

def exact_solution(t, m, k, c):
    """
    Computes the exact solution of the damped harmonic oscillator.

    Args:
        t (np.array): Time values at which to compute the solution.
        m (float): Mass of the oscillator.
        k (float): Spring constant.
        c (float): Damping coefficient.

    Returns:
        np.array: Exact solution values at the given times.
    """
    omega = np.sqrt(4 * m * k - c**2) / (2 * m)
    return 2 * np.exp(-c * t / (2 * m)) * np.cos(omega * t)

def generate_damped_oscillator_data(m, k, c, t_min, t_max, num_samples, noise_std=0.0, random_seed=None):
    """
    Generates synthetic data from the analytic solution of a damped harmonic oscillator.

    Args:
        m (float): Mass of the oscillator.
        k (float): Spring constant.
        c (float): Damping coefficient.
        t_min (float): Start time.
        t_max (float): End time.
        num_samples (int): Number of sampled points.
        noise_std (float): Standard deviation of Gaussian noise (0 for noiseless data).
        random_seed (int, optional): Seed for reproducibility.

    Returns:
        X (numpy array): Sampled time points.
        y (numpy array): Function values (with noise if specified).
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Evenly spaced time samples
    X = np.linspace(t_min, t_max, num_samples) 
    # add noise to time samples
    X += noise_std * np.random.randn(num_samples) 
    # Compute the exact solution at these points
    y = exact_solution(X, m, k, c)
    
    # Add Gaussian noise if noise_std > 0
    if noise_std > 0:
        y += noise_std * np.random.randn(num_samples)
    
    return X, y

# Define Nord color scheme
nord_colors = {
    'polar_night': {
        'nord0': '#2E3440',
        'nord1': '#3B4252',
        'nord2': '#434C5E',
        'nord3': '#4C566A',
    },
    'snow_storm': {
        'nord4': '#D8DEE9',
        'nord5': '#E5E9F0',
        'nord6': '#ECEFF4',
    },
    'frost': {
        'nord7': '#8FBCBB',
        'nord8': '#88C0D0',
        'nord9': '#81A1C1',
        'nord10': '#5E81AC',
    },
    'aurora': {
        'nord11': '#BF616A',
        'nord12': '#D08770',
        'nord13': '#EBCB8B',
        'nord14': '#A3BE8C',
        'nord15': '#B48EAD',
    },
}

# Define a Nord-based colormap
from matplotlib.colors import LinearSegmentedColormap
nord_cmap = LinearSegmentedColormap.from_list('nord', [nord_colors['polar_night']['nord0'], nord_colors['aurora']['nord13']])
nord_cmap = LinearSegmentedColormap.from_list('nord', [nord_colors['polar_night']['nord3'], nord_colors['frost']['nord10'], nord_colors['aurora']['nord13']]
)


def plot_covariance_matrix(ax, gp):
    """
    Plots the covariance matrix for a given set of input points.

    Args:
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        gp (GaussianProcess): The Gaussian Process instance.
    """
    im = ax.imshow(gp.K, cmap=nord_cmap, interpolation='nearest')
    plt.colorbar(im, ax=ax, label="Covariance")
    ax.set_title("Covariance Matrix")
    ax.set_xlabel("Index")
    ax.set_ylabel("Index")

def plot_posterior_covariance(ax, gp):
    """
    Visualizes the posterior covariance matrix of a GaussianProcess instance as a heatmap.

    Args:
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        gp (GaussianProcess): The Gaussian Process instance.
    """
    im = ax.imshow(gp.posterior_covariance, cmap=nord_cmap, interpolation='nearest')
    plt.colorbar(im, ax=ax, label="Covariance")
    ax.set_title("Posterior Covariance Matrix")
    ax.set_xlabel("Test Points Index")
    ax.set_ylabel("Test Points Index")

def plot_predictions_with_uncertainty(ax, X_test, gp, m, k, c):
    """
    Plots the GP predictions with uncertainty and overlays the exact solution.

    Args:
        ax (matplotlib.axes.Axes): The subplot axis to plot on.
        X_test (np.array): Test points (time values) for prediction.
        gp (GaussianProcess): Trained Gaussian Process.
        m (float): Mass of the oscillator.
        k (float): Spring constant.
        c (float): Damping coefficient.
    """
    # Get GP predictions
    mu, std = gp.predict(X_test)

    # Compute the exact solution
    y_exact = exact_solution(X_test, m, k, c)

    # Plotting
    ax.plot(X_test, mu, label='Predicted Mean', color=nord_colors['frost']['nord10'])
    ax.fill_between(X_test, mu - 2*std, mu + 2*std, alpha=0.2, label='95% Confidence Interval', color=nord_colors['polar_night']['nord2'])
    ax.plot(X_test, y_exact, label='Exact Solution', color=nord_colors['aurora']['nord11'], linestyle='--', linewidth=2)
    ax.scatter(gp.X_train, gp.y_train, c=nord_colors['polar_night']['nord0'], label='Training Data', marker='o', alpha=0.7)

    # Labels and legend
    ax.set_xlabel("Time")
    ax.set_ylabel("Displacement")
    ax.set_title("Predictions vs. Exact Solution")
    ax.legend()
    ax.grid(True)

def plot_all_in_one(X_test, gp, m, k, c):
    """
    Combines the covariance matrix, posterior covariance, and predictions with uncertainty into one figure.

    Args:
        X_test (np.array): Test points (time values) for prediction.
        gp (GaussianProcess): Trained Gaussian Process.
        m (float): Mass of the oscillator.
        k (float): Spring constant.
        c (float): Damping coefficient.
    """
    # Create a custom grid layout
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

    # Top row: Covariance matrices
    ax1 = fig.add_subplot(gs[0, 0])  # Top-left
    ax2 = fig.add_subplot(gs[0, 1])  # Top-right

    # Bottom row: Predictions with uncertainty (spanning both columns)
    ax3 = fig.add_subplot(gs[1, :])  # Bottom row, full width

    # Plot covariance matrix
    plot_covariance_matrix(ax1, gp)

    # Plot posterior covariance
    plot_posterior_covariance(ax2, gp)

    # Plot predictions with uncertainty
    plot_predictions_with_uncertainty(ax3, X_test, gp, m, k, c)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()   

    
if __name__ == "__main__":
    m = 2.0
    k = 4.10
    c = 0.50
    t_min = 0
    t_max = 10
    num_samples = 15
    noise_std = 0.1
    random_seed = 42
    X_train, y_train = generate_damped_oscillator_data(m, k, c, t_min, t_max, num_samples, noise_std, random_seed)

    length_scale =1.9
    period = 4.1
    decay = 0.50
    sigma_f = 0.5
    noise = 0.1
    
    Gp = GaussianProcess(X_train, y_train, length_scale, period, decay, sigma_f, noise)
    #plot_exact_solution_with_data(Gp, m, k, c, t_min, t_max)
    #plot_covariance_matrix(Gp)
    #plot_posterior_covariance(Gp)
    X_test = np.linspace(0,10,300)

    mu, std = Gp.predict(X_test)

    #plot_predictions_with_uncertainty(X_test, Gp, m, k, c)
    plot_all_in_one(X_test, Gp, m, k, c)
