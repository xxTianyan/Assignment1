import numpy as np
import matplotlib.pyplot as plt

def generate_strain_path(max_strain: float = 0.08, 
                        n_steps: int = 25000, 
                        plot: bool = False) -> np.ndarray:
    """
    Generate a strain path vector with specified amplitude and steps.
    
    Parameters:
        max_strain (float): Maximum absolute strain value for scaling (default: 0.08)
        n_steps (int): Total number of steps in the generated path (default: 25000)
        plot (bool): Flag to enable plotting (default: False)
        save_path (str, optional): Path to save the plot image (default: None)
    
    Returns:
        np.ndarray: Generated strain path vector
    
    Raises:
        ValueError: If input parameters are invalid
    """
    # Validate input parameters
    if max_strain <= 0:
        raise ValueError("max_strain must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be greater than 0")
    
    # Define normalized pattern points (relative to max_strain)
    pattern_points = np.array([0, 0.6, -0.8, 0.95, -0.7])  # Key points for strain pattern
    scaled_strain = max_strain * pattern_points  # Scale pattern to target strain range
    
    # Create linear interpolation between pattern points
    x_new = np.linspace(0, len(pattern_points)-1, n_steps)  # New interpolation positions
    x_original = np.arange(len(pattern_points))  # Original point indices
    strain_vector = np.interp(x_new, x_original, scaled_strain)
    
    if plot:
        plt.figure(figsize=(10, 5))
        plt.plot(np.arange(n_steps), strain_vector, 
                color='navy', 
                linewidth=1.5,
                label='Strain Path')
        plt.title('Strain Loading Path Generation', fontsize=14)
        plt.xlabel('Step Number', fontsize=12)
        plt.ylabel('Normalized Strain', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # Set axis limits to show full pattern
        plt.xlim(0, n_steps)
        plt.ylim(-max_strain*1.1, max_strain*1.1)
        
        if plt:
            plt.savefig('strain_path.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    return strain_vector

class ElastoPlastic:
    """
    Simulates elasto-plastic material behavior with isotropic or kinematic hardening.

    Parameters:
        E (float): Young's modulus [MPa]
        H (float): Hardening modulus [MPa] (must be ≥0 for isotropic)
        Yi (float): Initial yield stress [MPa]
        mode (str): Hardening type ('isotropic' or 'kinematic')
    """
    def __init__(self, E: float, H: float, Yi: float, mode: str = 'isotropic'):
        # : Add input validation 
        if mode not in ["isotropic", "kinematic"]:
            raise ValueError("Invalid mode. Choose 'isotropic' or 'kinematic'")
        if H < 0 and mode == "isotropic":
            raise ValueError("H must be ≥0 for isotropic hardening")
        
        self.E = E
        self.H = H
        self.Yi = Yi
        self.mode = mode
        
        # Initialize state variables
        self.sigma_n = 0.0           # Current stress
        self.epsilon_p = 0.0         # Accumulated plastic strain
        self.Yn = Yi                 # Current yield stress (for isotropic hardening)
        self.alpha_n = 0.0           # Back stress (for kinematic hardening)
        
        # History recording structure
        self.stress_history = []      # Stores all stress values (including the initial value)
        self.strain_history = []      # Stores all strain values (including the initial value)

    def clear_history(self):
        """Clear history records"""
        self.stress_history = []
        self.strain_history = []

    def apply_loading(self, strain_path: np.ndarray):
        """Apply strain path and compute the stress response"""
        self.clear_history()
        
        if len(strain_path) == 0:
            return
        
        # Record initial state
        self.stress_history.append(self.sigma_n)
        self.strain_history.append(strain_path[0])
        
        # Compute strain increments
        delta_strain = np.diff(strain_path)
        current_strain = strain_path[0]
        
        for d_epsilon in delta_strain:
            current_strain += d_epsilon
            if self.mode == "isotropic":
                self._update_isotropic(d_epsilon)
            else:
                self._update_kinematic(d_epsilon)
            self.strain_history.append(current_strain)

    def _update_isotropic(self, delta_epsilon: float):
        """Isotropic hardening update logic"""
        # Elastic trial stress
        sigma_trial = self.sigma_n + self.E * delta_epsilon
        f_trial = abs(sigma_trial) - self.Yn
        
        if f_trial <= 0:
            # Elastic phase
            self.sigma_n = sigma_trial
        else:
            # Add denominator check
            denominator = self.E + self.H
            if denominator <= 1e-10:
                raise ZeroDivisionError("E + H cannot be zero")
            
            # Plastic correction
            d_lambda = f_trial / denominator
            self.sigma_n = sigma_trial - np.sign(sigma_trial) * self.E * d_lambda
            self.epsilon_p += d_lambda
            self.Yn = self.Yi + self.H * self.epsilon_p  # Update yield stress
        
        # Record stress
        self.stress_history.append(self.sigma_n)

    def _update_kinematic(self, delta_epsilon: float):
        """Kinematic hardening update logic"""
        # Elastic trial stress
        sigma_trial = self.sigma_n + self.E * delta_epsilon
        eta_trial = sigma_trial - self.alpha_n
        f_trial = abs(eta_trial) - self.Yi
        
        if f_trial <= 0:
            # Elastic phase
            self.sigma_n = sigma_trial
        else:
            # Add denominator check
            denominator = self.E + self.H
            if denominator <= 1e-10:
                raise ZeroDivisionError("E + H cannot be zero")
            
            # Plastic correction
            d_lambda = f_trial / denominator
            self.sigma_n = sigma_trial - np.sign(eta_trial) * self.E * d_lambda
            self.alpha_n += np.sign(eta_trial) * self.H * d_lambda
            self.epsilon_p += d_lambda
        
        # Record stress
        self.stress_history.append(self.sigma_n)

    def plot_curve(self, save_path: str = 'loading_plot.png'):
        """Plot the stress-strain curve"""
        if len(self.strain_history) != len(self.stress_history):
            raise ValueError("Strain and stress data length mismatch")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.strain_history, self.stress_history, "b-", lw=1.5)
        plt.xlabel("Strain")
        plt.ylabel("Stress (MPa)")
        plt.title(f"Stress-Strain Curve ({self.mode.capitalize()} Hardening)")
        plt.grid(ls="--", alpha=0.5)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")



if __name__ == '__main__':
    strain_vector = generate_strain_path()
    model = ElastoPlastic(E=1000, H=111.1111, Yi=10, mode='kinematic')
    model.apply_loading(strain_vector)
    model.plot_curve()
