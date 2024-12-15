# Importing packages:
import math
import statistics
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from itertools import product  # For machinelearning/factorial design
import pandas as pd

"""
rx_entropy scales with c^2, while diffusion scales with c.

idea: try to balance rate constants such that the rx scaling equalus diff scaling
"""

steps_per_frame         = 5                     # Number of frames produced per iteration step in the output GIF
gif_duration            = 15                    # Specifies how long (in milliseconds) each frame should be displayed. 
                                                # Increase/decrease to slowdown/speedup video. 
                                                # 60fr/s = 16.67 ms/fr
k                       = 500                  # Number of iterations (= number of frames produced)

no0_matrix              = 1e-8                  # Scalar used to remove 0 for mathemetical operations like division
R                       = 1                     # Gass constant                        (currently not in use) 
laplacian_scalar        = 1                     # Scaling factor for the Laplacians    (currently not in use) 
RxScalar                = 1                     #                                      (currently not in use) 

Lx, Ly                  = 200, 200              # Length of the box
dx                      = 1                     # Grid spacing (dx = dy) & number of grid points: Nx=Lx/dx
dt                      = 0.001

c_A_init, c_B_init, c_C_init = 1, 1, 1          # Initital concentration for setting up the system
sigma_A, sigma_B = 50, 50                       # Standard deveation for the Gaussian initialization function

rate_scalar1, rate_scalar2, rate_scalar3 = (
    1,
    1,
    1,
)                                               # Controls how many times each reaction proceeds per unit time

"""
###############################################################################
Paramters used to run a single simulation with function: simulation()
###############################################################################
"""
Keq1, Keq2, Keq3        = 40, 20, 0.050               # Defines dimensionless rate constants
delta, tau              = 100, 20             # Dimensionless diffusion constants
index                   = "test3"                # Name for outputs when running single simulation
"""
###############################################################################
Parameters used to run a factorial simulation with function: factorial_simulation()
###############################################################################
"""
# Levels factorial design
Keq1_levels  = [100,200,300]
Keq2_levels  = [100,200,300]
Keq3_levels  = [100,200,300]
delta_levels = [10,100,200]          # Smaller diffusion coefficients for species C
tau_levels   = [10,100,200]
factorial_design = list(
    product(Keq1_levels, Keq2_levels, Keq3_levels, delta_levels, tau_levels)
)

Keq1_max, Keq2_max, Keq3_max = max(Keq1_levels), max(Keq2_levels), max(Keq3_levels)
Keq1_min, Keq2_min, Keq3_min = min(Keq1_levels), min(Keq2_levels), min(Keq3_levels)
tau_max, delta_max = max(tau_levels), max(delta_levels)
# Variables for dissociation: 
mu, sigma, omega = rate_scalar1 / Keq1_min, rate_scalar2 / Keq2_min, rate_scalar3 / Keq3_min


# Variables for association:   
epsilon, kappa, gamma = (     
    rate_scalar1 * Keq1_max,  
    rate_scalar2 * Keq2_max,  
    rate_scalar3 * Keq3_max,  
)


# Ensure numerical stability (optional: adjust dt if necessary)
D_max = laplacian_scalar * max(1, tau_max, delta_max)
K_max = max(mu, sigma, omega, epsilon, kappa, gamma)
dt_max = 1/((4 * D_max) / dx**2 + K_max)

if dt > dt_max:
    print("timestep to big. Max value =", dt_max)

#Main program - Choose setupy my commenting/uncommenting functions
def run_program():

    """Choose initialization function for box concentrations"""

    A, B, C = init_concentration_uniform(Lx, Ly, dx, c_A_init, c_B_init, c_C_init)
    # A, B, C = init_concentrations_gaussian(Lx, Ly, dx, c_A_init, c_B_init, c_C_init, sigma_A, sigma_B)

    
    """Run singe simulation or factorial design simulation"""

    #simulation(A,B,C,Keq1,Keq2,Keq3,delta,tau,k,index)
    factorial_simulation(factorial_design, k, A, B, C)



"""
INITIALIZATION OF CONCENTRATIONS
"""


def init_concentrations_gaussian(
    Lx, Ly, dx, c_A_init, c_B_init, c_C_init, sigma_A, sigma_B
):
    """
    Initialize the concentration arrays for species A, B, and C using Gaussian distributions.

    Parameters:
    Lx, Ly: float
        Size of the domain in x and y directions.
    dx : float
        Grid spacing (assumed equal in x and y directions).
    c_A_init, c_B_init, c_C_init : float, optional
        Initial concentrations for species A, B, and C. Defaults are 1.0 for A and B, 0.15 for C.
    sigma_A, sigma_B: float
        Standard deviations for the Gaussian distributions of species A and B.

    Returns:
      [np.ndarray]: Arrays containing the initial concentration distribution of A, B and C for each grid point.
    """
    np.random.seed(42)  # for reuseability between each simulation

    # Calcualte the number of grid points for the box
    Nx = int(Lx / dx)
    Ny = int(Ly / dx)

    # Create meshgrid for calculating Gaussian distributions
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Center points for the Gaussians - Left and right side of box
    x0_A, y0_A = Lx / 3, Ly / 2  # Center of Gaussian for species A
    x0_B, y0_B = 2 * Lx / 3, Ly / 2  # Center of Gaussian for species B

    # Initialize arrays with ghost cells for periodic BC: shape = [402,202]
    A = np.zeros((Ny + 2, Nx + 2))
    B = np.zeros((Ny + 2, Nx + 2))
    C = np.zeros((Ny + 2, Nx + 2))

    # Calculate Gaussian distributions
    A[1:-1, 1:-1] = c_A_init * np.exp(
        -((X - x0_A) ** 2 + (Y - y0_A) ** 2) / (2 * sigma_A**2)
    )
    B[1:-1, 1:-1] = c_B_init * np.exp(
        -((X - x0_B) ** 2 + (Y - y0_B) ** 2) / (2 * sigma_B**2)
    )
    C[1:-1, 1:-1] = c_C_init * np.random.random(
        (Ny, Nx)
    )  # Add small random noise to start reactions

    return A, B, C


def init_concentration_uniform(
    Nx, Ny, dx, c_A_init, c_B_init, c_C_init, noise_level=0.5
):

    # Calcualte the number of grid points for the box
    Nx = int(Lx / dx)
    Ny = int(Ly / dx)

    #    np.random.seed(42)
    # Initialize arrays with ghost cells for periodic BC: shape = [402,202]
    A = np.zeros((Ny + 2, Nx + 2))
    B = np.zeros((Ny + 2, Nx + 2))
    C = np.zeros((Ny + 2, Nx + 2))

    np.random.seed(42)
    A = c_A_init * noise_level * (np.random.random((Nx, Ny)))
    np.random.seed(30)
    B = c_B_init * noise_level * (np.random.random((Nx, Ny)))
    np.random.seed(50)
    C = c_C_init * noise_level * (np.random.random((Nx, Ny)))

    return A, B, C


"""
UTILITY FUNCTIONS
"""


def periodic_bc(u):
    """
    Impose periodic boundary conditions on a two dimensional array.

    Parameters:
    u: np.ndarray
      A two dimensional numpy array representing the simulation grid.

    Returns:
      [None]
    """
    u[0, :] = u[-2, :]
    u[-1, :] = u[1, :]
    u[:, 0] = u[:, -2]
    u[:, -1] = u[:, 1]


def laplacian(u, dx):
    """
    Compute the Laplacian of a two dimensional array.

    Parameters:
    u: np.ndarray
      A two dimensional numpy array representing the simulation grid.

    Returns:
      [np.ndarray]: A two dimensional numpy array holding the Laplacian of u.
    """
    return (
        u[:-2, 1:-1] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1] + u[1:-1, 2:] + u[2:, 1:-1]
    ) / dx**2


def gradient(u, dx):
    """
    Compute the gradient of a 2D array using central finite differences.

    Parameters:
    u : numpy.ndarray
        A two-dimensional numpy array representing the simulation grid.
    dx : float
        The grid spacing (assumed equal in x and y directions).

    Returns:
        tuple of numpy.ndarray: (du_dx, du_dy), the gradient components along x and y.
    """
    # np.roll assumes periodic boundry conditions

    # Compute partial derivative with respect to x
    du_dx = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) / (2 * dx)

    # Compute partial derivative with respect to y
    du_dy = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)

    return du_dx, du_dy


"""
ENTROPY CALCULATION
"""


def rxSum(
    a,
    b,
    c,
    eta,
    mu,
    kappa,
    nu,
    epsilon,
    rho,
    sigma,
    delta,
    gamma,
    pi,
    phi,
    tau,
    lamb,
    omega,
):
    """
    Compute the total reaction entropy production.

    Parameters:
    a, b, c : numpy.ndarray
        Concentration profiles of species A, B, and C.

    Returns:
        numpy.ndarray
            The reaction entropy production at each point.
     + no0_matrix
     + no0_matrix
     + no0_matrix

    """

    # Adjust concentrations
    ano0 = a + no0_matrix
    bno0 = b + no0_matrix
    cno0 = c + no0_matrix

    # Dobbelt sjekk stabliserings kriterie for ano0... multiplikasjon
    # Compute forward and reverse rates
    r1f = epsilon * ano0 * bno0
    r1r = mu * ano0**2
    r2f = kappa * bno0 * cno0
    r2r = sigma * bno0**2
    r3f = gamma * ano0 * cno0
    r3r = omega * cno0**2

    # Compute net rates and affinities
    net_rate1 = r1f - r1r
    affinity1 = np.log(r1f / r1r)
    net_rate2 = r2f - r2r
    affinity2 = np.log(r2f / r2r)
    net_rate3 = r3f - r3r
    affinity3 = np.log(r3f / r3r)

    
    # Handle potential division by zero or log of zero
    affinity1 = np.nan_to_num(affinity1)
    affinity2 = np.nan_to_num(affinity2)
    affinity3 = np.nan_to_num(affinity3)
    
    # Compute entropy production terms
    term1 = net_rate1 * affinity1
    term2 = net_rate2 * affinity2
    term3 = net_rate3 * affinity3

    # Total reaction entropy production
    rx_entropy = R * (term1 + term2 + term3)

    return rx_entropy, term1, term2, term3


def entropy_calc(
    a,
    b,
    c,
    eta,
    mu,
    kappa,
    nu,
    epsilon,
    rho,
    sigma,
    delta,
    gamma,
    pi,
    phi,
    tau,
    lamb,
    omega,
):
    """
    Compute the total entropy production due to reactions and diffusion.
    First extract rx_diffusion from rxSum
    Then calculates entropy contribution
    Lastly sums each contribution to get total entropy production

    Parameters:
    a, b, c : numpy.ndarray
        Concentration profiles of species A, B, and C.
    params : dict
        Dictionary containing all necessary parameters:
        - Reaction parameters (as in rxSum)
        - Diffusion parameters: 'laplacian_scalar', 'tau', 'delta', 'dx'
        - 'R', 'no0_matrix'

    Returns:
        numpy.ndarray
            The total entropy production at each point.
     + no0_matrix
     + no0_matrix
     + no0_matrix

    """
    # Compute reaction entropy production
    reaction_entropy, term1, term2, term3 = rxSum(
        a,
        b,
        c,
        eta,
        mu,
        kappa,
        nu,
        epsilon,
        rho,
        sigma,
        delta,
        gamma,
        pi,
        phi,
        tau,
        lamb,
        omega,
    )

    # Adjust concentrations
    ano0 = a
    bno0 = b
    cno0 = c

    # Compute gradients
    da_dx, da_dy = gradient(ano0, dx)
    db_dx, db_dy = gradient(bno0, dx)
    dc_dx, dc_dy = gradient(cno0, dx)

    # Compute magnitude squared of gradients - Output scalar field
    grad_a_squared = da_dx**2 + da_dy**2
    grad_b_squared = db_dx**2 + db_dy**2
    grad_c_squared = dc_dx**2 + dc_dy**2
    
    # Avoid division by zero in diffusion entropy production
    ano0_safe = np.maximum(ano0, no0_matrix)
    bno0_safe = np.maximum(bno0, no0_matrix)
    cno0_safe = np.maximum(cno0, no0_matrix)
    
    a_entropy_diff = laplacian_scalar * grad_a_squared / ano0_safe
    b_entropy_diff = laplacian_scalar * grad_b_squared / bno0_safe
    c_entropy_diff = laplacian_scalar * grad_c_squared / cno0_safe

    # Compute diffusion entropy production
    diffusion_entropy = R * (a_entropy_diff + b_entropy_diff + c_entropy_diff)
    

    # Total entropy production
    total_entropy = reaction_entropy + diffusion_entropy

    return (
        total_entropy,
        reaction_entropy,
        diffusion_entropy,
        a_entropy_diff,
        b_entropy_diff,
        c_entropy_diff,
        term1,
        term2,
        term3,
        grad_a_squared,
        grad_b_squared,
        grad_c_squared,
    )


def RxDiffusion(
    A,
    B,
    C,
    eta,
    mu,
    kappa,
    nu,
    epsilon,
    rho,
    sigma,
    delta,
    gamma,
    pi,
    phi,
    tau,
    lamb,
    omega,
    i,
):
    """
    Simulate the reaction-diffusion system, compute the total entropy production,
    and track concentration profiles over time.

    Parameters:
    ----------
    A, B, C : numpy.ndarray
        Concentration arrays for species A, B, and C, including ghost cells.
    """

    # Extract the computational domain (excluding ghost cells)
    a = A[1:-1, 1:-1]
    b = B[1:-1, 1:-1]
    c = C[1:-1, 1:-1]
    # Compute laplacian terms
    La = laplacian_scalar * laplacian(A, dx)
    Lb = laplacian_scalar * laplacian(B, dx) * tau
    Lc = laplacian_scalar * laplacian(C, dx) * delta

    # Precompute products for reaction terms
    aa = a * a  # a^2
    bb = b * b  # b^2
    cc = c * c  # c^2
    ab = a * b  # a * b
    ac = a * c  # a * c
    bc = b * c  # b * c

    # Compute net reaction rates
    Ra = epsilon * ab + eta * cc - lamb * ac - mu * aa
    Rb = kappa * bc + nu * aa - rho * ab - sigma * bb
    Rc = gamma * ac + pi * bb - phi * bc - omega * cc

    # Update concentrations using the explicit Euler method
    a_new = a + (La + Ra) * dt
    b_new = b + (Lb + Rb) * dt
    c_new = c + (Lc + Rc) * dt

    
    # Prevent negative concentrations
    a_new = np.maximum(a_new, 0, out=a_new)
    b_new = np.maximum(b_new, 0, out=b_new)
    c_new = np.maximum(c_new, 0, out=c_new)
    

    # Update the arrays in the computational domain
    a[:, :] = a_new
    b[:, :] = b_new
    c[:, :] = c_new

    # Apply periodic boundary conditions
    periodic_bc(A)
    periodic_bc(B)
    periodic_bc(C)

    # Track total concentrations
    total_a = np.sum(a)
    total_b = np.sum(b)
    total_c = np.sum(c)
    total_concentration = total_a + total_b + total_c
    max_a = np.max(a)
    min_a = np.min(a)
    max_b = np.max(b)
    min_b = np.min(b)
    max_c = np.max(c)
    min_c = np.min(c)
    var_a = np.var(a)
    var_b = np.var(b)
    var_c = np.var(c)


    # Compute entropy production
    entropy_results = entropy_calc(
        a,
        b,
        c,
        eta,
        mu,
        kappa,
        nu,
        epsilon,
        rho,
        sigma,
        delta,
        gamma,
        pi,
        phi,
        tau,
        lamb,
        omega,
    )

    entropy_value = np.sum(entropy_results[0])
    reaction_entropy = np.sum(entropy_results[1])
    diffusion_entropy = np.sum(entropy_results[2])
    a_entropy_diff = np.sum(entropy_results[3])
    b_entropy_diff = np.sum(entropy_results[4])
    c_entropy_diff = np.sum(entropy_results[5])
    term1 = np.sum(entropy_results[6])
    term2 = np.sum(entropy_results[7])
    term3 = np.sum(entropy_results[8])
    grad_a_squared = np.sum(entropy_results[9])
    grad_b_squared = np.sum(entropy_results[10])
    grad_c_squared = np.sum(entropy_results[11])
    #    ano0_safe           = (entropy_results[12])
    #    bno0_safe           = (entropy_results[13])
    #    cno0_safe           = (entropy_results[14])

    # Validate physical behavior (checking for negative concentrations)
    if np.any(a < 0):
        print(f"A negative concentration at iteration {i}")
    if np.any(b < 0):
        print(f"B negative concentration at iteration {i}")
    if np.any(c < 0):
        print(f"C negative concentration at iteration {i}")


    return (
        a,
        b,
        c,
        (total_a, total_b, total_c, total_concentration),
        entropy_value,
        reaction_entropy,
        diffusion_entropy,
        a_entropy_diff,
        b_entropy_diff,
        c_entropy_diff,
        term1,
        term2,
        term3,
        grad_a_squared,
        grad_b_squared,
        grad_c_squared,
        max_a,
        min_a,
        max_b,
        min_b,
        max_c,
        min_c,
        var_a,
        var_b,
        var_c,
    )


def create_image(a, b, c):
    global_min = min(a.min(), b.min(), c.min())
    global_max = max(a.max(), b.max(), c.max())
    a_scaled = np.uint8(255 * (a) / (global_max - global_min + 1e-8))
    b_scaled = np.uint8(255 * (b) / (global_max - global_min + 1e-8))
    c_scaled = np.uint8(255 * (c) / (global_max - global_min + 1e-8))

    """
    # Scale concentrations
    a_scaled = np.uint8(255 * (a / (a.max() - a.min() + 1e-8)))
    b_scaled = np.uint8(255 * (b / (b.max() - b.min() + 1e-8)))
    c_scaled = np.uint8(255 * (c / (c.max() - c.min() + 1e-8)))

        # Create RGB image
        rgb_image = np.stack((a_scaled, b_scaled, c_scaled), axis=-1)

    """   
    # RGB:
    rgb_image = np.zeros((a_scaled.shape[0], a_scaled.shape[1], 3), dtype=np.uint8)
    rgb_image[..., 0] = a_scaled 
    rgb_image[..., 1] = b_scaled
    rgb_image[..., 2] = c_scaled
    return rgb_image


def plot_entopy_totalC(
    totalCA, totalCB, totalCC, totalC, iter, entropy, rx_entropy, diff_entropy, index
):

    totalC = np.array(totalC)
    totalCA = np.array(totalCA)
    totalCB = np.array(totalCB)
    totalCC = np.array(totalCC)
    entropy = np.array(entropy)
    rx_entropy = np.array(rx_entropy)
    diff_entropy = np.array(diff_entropy)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(iter, totalCA / totalC, color="r", label=r"$x_{A}$")
    ax1.plot(iter, totalCB / totalC, color="g", label=r"$x_{B}$")
    ax1.plot(iter, totalCC / totalC, color="b", label=r"$x_{C}$")
    ax2.plot(iter, entropy/np.max(entropy), label=r"$\sigma_{tot}$", color = 'purple')
    ax2.plot(iter, rx_entropy/np.max(entropy), label=r"$\sigma_{rx}$", color = 'yellow', linestyle = ':')
    ax2.plot(iter, diff_entropy/np.max(entropy), label=r"$\sigma_{diffusion}$", color = 'green', linestyle = ':')
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Entropy production")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Mole fraction")
    ax1.grid()
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax2.legend(fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # no idea why this needs to be here but if its not the plot looks like crap. 
    # plt.show()
    #print("Maximum concentration of A", totalCA.max())
    # Save the concentration plot with a numeric filename
    filename_conc = f"plot_{index}_concentration.png"
    fig.suptitle('Mole Fractions and Entropy Production', fontsize=14)
    plt.savefig(filename_conc, dpi=300)  # Save the concentration plot
    plt.close(fig)  # Close the figure to free memory


def plot_dconc(totalCA, totalCB, totalCC, totalC, index):
    dconcA = np.diff(totalCA)
    dconcB = np.diff(totalCB)
    dconcC = np.diff(totalCC)

    totalC = np.array(totalC[:-1])
    totalCA = np.array(totalCA[:-1])
    totalCB = np.array(totalCB[:-1])
    totalCC = np.array(totalCC[:-1])

    fig, ax = plt.subplots()
    ax.plot(totalCA / totalC, dconcA, color="red", label=r"$\frac{dx_A}{dt}$")
    ax.plot(totalCB / totalC, dconcB, color="green", label=r"$\frac{dx_B}{dt}$")
    ax.plot(totalCC / totalC, dconcC, color="blue", label=r"$\frac{dx_C}{dt}$")
    ax.set_xlabel(r"$x_A$, $x_B$, $x_C$")
    ax.set_ylabel(r"$\frac{dx_A}{dt}$, $\frac{dx_B}{dt}$, $\frac{dx_C}{dt}$")
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    # plt.show()
    # Save the rate of change plot with a numeric filename
    filename_rate = f"plot_{index}_rate.png"
    plt.savefig(filename_rate, dpi=300)  # Save the rate of change plot
    plt.close(fig)  # Close the figure to free memory


def plot_entropy_loadings(
    a_entropy_diff,
    b_entropy_diff,
    c_entropy_diff,
    a_entropy_rx,
    b_entropy_rx,
    c_entropy_rx,
    iter,
    index,
):

    a_entropy_diff = np.array(a_entropy_diff)
    b_entropy_diff = np.array(b_entropy_diff)
    c_entropy_diff = np.array(c_entropy_diff)
    a_entropy_rx   = np.array(a_entropy_rx)
    b_entropy_rx   = np.array(b_entropy_rx)
    c_entropy_rx   = np.array(c_entropy_rx)
    tot_entropy = a_entropy_diff + b_entropy_diff + c_entropy_diff + a_entropy_rx + b_entropy_rx + c_entropy_rx

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(iter, a_entropy_diff/np.max(tot_entropy), color="r", label=r"$\sigma_A^{diff}$")  # Normalized w.r.t. to the total entropy production. 
    ax1.plot(iter, b_entropy_diff/np.max(tot_entropy), color="g", label=r"$\sigma_B^{diff}$")
    ax1.plot(iter, c_entropy_diff/np.max(tot_entropy), color="b", label=r"$\sigma_C^{diff}$")
    ax2.plot(iter, a_entropy_rx/np.max(tot_entropy), color="r", label=r"$\sigma_A^{rx}$")
    ax2.plot(iter, b_entropy_rx/np.max(tot_entropy), color="g", label=r"$\sigma_B^{rx}$")
    ax2.plot(iter, c_entropy_rx/np.max(tot_entropy), color="b", label=r"$\sigma_C^{rx}$")
    ax1.set_xlabel('Iteration')
    ax2.set_xlabel('Iteration')
    ax1.set_ylabel(r"$\sigma^{diff}$")
    ax2.set_ylabel(r"$\sigma^{rx}$")
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.suptitle('Contributions to the Entropy Production', fontsize=14)
    plt.savefig(f"plot_{index}entropy_loadings", dpi=300)
    plt.close(fig)


def plot_diffusive_terms(
    entropy_grad_a_squared, entropy_grad_b_squared, entropy_grad_c_squared, iter, index
):

    entropy_grad_a_squared = np.array(entropy_grad_a_squared)
    entropy_grad_b_squared = np.array(entropy_grad_b_squared)
    entropy_grad_c_squared = np.array(entropy_grad_c_squared)

    fig, (ax1) = plt.subplots()
    ax1.plot(iter, entropy_grad_a_squared/np.max(entropy_grad_a_squared), color="r", label=r"$|\nabla c_A|^2$")  # Normalized w.t.r. to their max so its c and not x here. 
    ax1.plot(iter, entropy_grad_b_squared/np.max(entropy_grad_b_squared), color="g", label=r"$|\nabla c_B|^2$")
    ax1.plot(iter, entropy_grad_c_squared/np.max(entropy_grad_c_squared), color="b", label=r"$|\nabla c_C|^2$")
    ax1.legend(fontsize=10)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Square magnitude of concentration gradient')
    ax1.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    ax1.set_title('Diffusive Terms', fontsize=14)
    plt.savefig(f"plot_{index}diffusive_terms", dpi=300)
    plt.close(fig)
    """
    plt.imshow(np.array(entropy_cno0_safe[175]), cmap='viridis')
    plt.colorbar(label='ano0_safe')
    plt.title("ano0_safe at final time step")
    plt.show()
    """

def plot_additional_data(iter, variance_A, variance_B, variance_C, max_molfrac_A_list, min_molfrac_A_list, max_molfrac_B_list, min_molfrac_B_list, max_molfrac_C_list, min_molfrac_C_list, A, B, C, index):
  """
  Plot additional relevant data with professional styling.
  """
  # Plotting variances of concentrations
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(iter, variance_A, label='Var[$x_A$]', color='red')
  ax.plot(iter, variance_B, label='Var[$x_B$]', color='green')
  ax.plot(iter, variance_C, label='Var[$x_C$]', color='blue')
  ax.set_xlabel('Iteration', fontsize=12)
  ax.set_ylabel('Variance', fontsize=12)
  ax.set_title('Spatial Variance of Mole Fraction Over Time', fontsize=14)
  ax.grid(True, linestyle='--', alpha=0.7)
  ax.legend(fontsize=10)
  plt.tight_layout()
  filename_variance = f"plot_{index}_variance.png"
  plt.savefig(filename_variance, dpi=300)
  plt.close(fig)
  

  # Plotting max and min concentrations over time (Runs after the time loop so I'm pretty sure that the A, B and C passed into here are the final profiles:P 
  fig, ax = plt.subplots(figsize=(10, 6))
  ax.plot(iter, max_molfrac_A_list, label='$x_{A, max}$', color='red')
  ax.plot(iter, min_molfrac_A_list, label='$x_{A, min}$', color='red', linestyle='--')
  ax.plot(iter, max_molfrac_B_list, label='$x_{B, max}$', color='green')
  ax.plot(iter, min_molfrac_B_list, label='$x_{B, min}$', color='green', linestyle='--')
  ax.plot(iter, max_molfrac_C_list, label='$x_{C, max}$', color='blue')
  ax.plot(iter, min_molfrac_C_list, label='$x_{C, min}$', color='blue', linestyle='--')
  ax.set_xlabel('Iteration', fontsize=12)
  ax.set_ylabel('Molefraction', fontsize=12)
  ax.set_title('Maximum and Minimum Mole Fractions Over Time', fontsize=14)
  ax.grid(True, linestyle='--', alpha=0.7)
  ax.legend(fontsize=10, ncol=2)
  plt.tight_layout()
  filename_max_min_conc = f"plot_{index}_max_min_concentration.png"
  plt.savefig(filename_max_min_conc, dpi=300)
  plt.close(fig)
  
 # Creating separate heatmaps and quiver plots for species A, B, and C at the final time step
  species_dict = {'A': np.nan_to_num(A[1:-1, 1:-1]), 'B': np.nan_to_num(B[1:-1, 1:-1]), 'C': np.nan_to_num(C[1:-1, 1:-1])}  # Remove the ghost cells
  for species_label, concentration in species_dict.items():
    # Quiver and Contour plot
    du_dy, du_dx = np.gradient(concentration)  # Should point in the direction where the concentration is increasing most rappidly. c = f(x,y) gradx = g(x,y) vector field in ecaht point 
    du_dx = du_dx / dx
    du_dy = du_dy / dx

    magnitude = np.sqrt(du_dx**2 + du_dy**2)

    X = np.arange(0, concentration.shape[1])
    Y = np.arange(0, concentration.shape[0])
    X, Y = np.meshgrid(X, Y)
    fig, ax = plt.subplots(figsize=(8,6))
    
    # Plot contour of the concentration
    contour = ax.contourf(X, Y, concentration, levels=20, cmap='grey')
    
    # Vector-visualization + contour plot.
    skip = (slice(None, None, 10), slice(None, None, 10))  # Put the gradient-vector in every tenth gridpoint 

    ax.quiver(X[skip], Y[skip], du_dx[skip], du_dy[skip], color='red', scale=10)
    ax.set_title(f'Gradient Field and Contour Map for Species {species_label}', fontsize=14)
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    cbar = fig.colorbar(contour, ax=ax)
    cbar.set_label('Concentration', fontsize=12)  # Note both histogram and contour plot based ont he concentration not the mole fraction. 
    plt.tight_layout()
    filename = f"plot_{index}_quiver_contour_{species_label}.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    
    # Histogram of final concetration distribution 
    fig, ax = plt.subplots(figsize=(8,6))
    ax.hist(concentration.flatten(), bins=50, color='blue', alpha=0.7)
    ax.set_title(f'Concentration Distribution for Species {species_label}', fontsize=14)
    ax.set_xlabel('Concentration', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    filename = f"plot_{index}_histogram_{species_label}.png"
    plt.savefig(filename, dpi=300)
    plt.close(fig)

def find_entropy_stabilization_step(entropy_values, threshold=0.01, window_size=10):
    """
    Find the iteration step at which entropy stabilizes.
    """
    entropy_values = np.array(entropy_values)
    n = len(entropy_values)
    for i in range(n - window_size):
        window = entropy_values[i : i + window_size]
        mean_value = np.mean(window)
        max_deviation = np.max(np.abs((window - mean_value) / mean_value))
        if max_deviation < threshold:
            # Check that subsequent values remain within the threshold
            remaining_values = entropy_values[i + window_size :]
            if len(remaining_values) == 0:
                return i, mean_value
            relative_diffs = np.abs((remaining_values - mean_value) / mean_value)
            if np.all(relative_diffs < threshold):
                return i, mean_value
    return None, None


def extract_results(
    entropy,
    rx_entropy,
    diff_entropy,
    stabilization_step,
    a,
    b,
    c,
#    totalCA,
#    totalCB,
#    totalCC,
#    totalC,
    term1,
    term2,
    term3,
    index,
    Keq1,
    Keq2,
    Keq3,
    tau,
    delta,
    c_A_init,
    rate_scalar1,
    rate_scalar2,
    rate_scalar3,
    eta,
    mu,
    kappa,
    nu,
    epsilon,
    rho,
    sigma,
    gamma,
    pi,
    phi,
    lamb,
    omega,
    K,
    D,
):
    """
    FEATURES
    """
    # Calculate cumulative entropy production after the simulation ends
    cumulative_entropy_total = sum(entropy)
    cumulative_rx_entropy = sum(rx_entropy)
    cumulative_diff_entropy = sum(diff_entropy)

    # Calculate additional features
    max_entropy = max(entropy)
    #   time_to_max_entropy = iter[entropy.index(max_entropy)] * dt

    if stabilization_step is not None:
        entropy_value_at_stab = entropy[stabilization_step]
        stabilization_time = stabilization_step * dt

    else:
        entropy_value_at_stab = float("nan")
        stabilization_time = stabilization_step * dt


    entropy_start_end = entropy[41] / entropy[-1]

#    avg_mole_frac_a = np.mean(totalCA) / totalC
#    avg_mole_frac_b = np.mean(totalCB) / totalC
#    avg_mole_frac_c = np.mean(totalCC) / totalC

    variance_A_final = np.var(a)
    variance_B_final = np.var(b)
    variance_C_final = np.var(c)

    var_A_norm = variance_A_final / c_A_init
    var_B_norm = variance_B_final / c_A_init
    var_C_norm = variance_C_final / c_A_init
    
    ratio_var_AB = variance_A_final / variance_B_final
    ratio_var_BC = variance_B_final / variance_C_final
    ratio_var_AC = variance_A_final / variance_C_final

    total_variance = variance_A_final + variance_B_final + variance_C_final
    var_span = max(variance_A_final, variance_B_final, variance_C_final) - min(variance_A_final, variance_B_final, variance_C_final)
    frac_var_A = variance_A_final / total_variance
    frac_var_B = variance_B_final / total_variance
    frac_var_C = variance_C_final / total_variance


    frac_rx_entropy = cumulative_rx_entropy / cumulative_entropy_total
    frac_diff_entropy = cumulative_diff_entropy / cumulative_entropy_total
    ratio_rx_diff_entropy = cumulative_rx_entropy / cumulative_diff_entropy
    
    frac_entropy_at_stab = entropy_value_at_stab / max_entropy

    

    #    time_to_max_conc_A = iter[totalCA.index(max_conc_A)] * dt
    #    time_to_max_conc_B = iter[totalCB.index(max_conc_B)] * dt
    #    time_to_max_conc_C = iter[totalCC.index(max_conc_C)] * dt

    #    average_rate_A = np.mean(rx_rate_A)
    #    average_rate_B = np.mean(rx_rate_B)
    #    average_rate_C = np.mean(rx_rate_C)

    #    max_rate_A = np.max(rx_rate_A)
    #    max_rate_B = np.max(rx_rate_B)
    #    max_rate_C = np.max(rx_rate_C)

    ratio_sumK_sumD = sum(K) / sum(D)
    ratio_maxK_minD = max(K) / min(D)
    ratio_minK_maxD = min(K) / max(D)
    ratio_geom_K_D = ((Keq1*Keq2*Keq3)**(1/3)) / ((tau*delta)**(1/2))
    
    mean_K = statistics.mean(K)
    std_K = statistics.pstdev(K)
    cv_K = std_K / mean_K
    
    mean_D = statistics.mean(D)
    std_D = statistics.pstdev(D)
    cv_D = std_D / mean_D
    
    ratio_cvK_cvD = cv_K / cv_D
    diff_cvK_cvD = cv_K - cv_D
    
    log_product_K = math.log(Keq1 * Keq2 * Keq3)
    log_product_D = math.log(1*tau * delta)
    log_diff = log_product_K - log_product_D
    
    ratio_median_K_D = statistics.median(K) / statistics.median(D)

    """No Diffusion for A"""
    Da_A = (term1 * np.max(a) * Lx * Ly)
    Da_B = (term2 * np.max(b) * Lx * Ly) / tau  # 2D-rectangle with y=length, x=2*length
    Da_C = (term3 * np.max(c) * Lx * Ly) / delta

    # Append the results for the current parameter set
    result = {
        'index': index,
        'delta': delta,
        'tau': tau,
        'Keq1': Keq1,
        'Keq2': Keq2,
        'Keq3': Keq3,
        'dt': dt,
        'dx': dx,
        'culumative_entropy': cumulative_entropy_total,
        'culumative_rx_entropy': cumulative_rx_entropy,
        'culumative_diff_entropy': cumulative_diff_entropy,
        'max_entropy': max_entropy,
        'entropy_value_at_stab': entropy_value_at_stab,
        'entropy_start_end': entropy_start_end,
        'stabilization_step': stabilization_step,
        'variance_A_final': variance_A_final,
        'variance_B_final': variance_B_final,
        'variance_C_final': variance_C_final,
#        'avg_mole_frac_a': avg_mole_frac_a,
#        'avg_mole_frac_b': avg_mole_frac_b,
#        'avg_mole_frac_c': avg_mole_frac_c,
        "max_Da_A": Da_A,
        'max_Da_B': Da_B,
        'max_Da_C': Da_C,
        'c_init': c_A_init,
        "rate_scalar1": rate_scalar1,
        "rate_scalar2": rate_scalar2,
        "rate_scalar3": rate_scalar3,
        "eta": eta,
        "mu": mu,
        "kappa": kappa,
        "nu": nu,
        "epsilon": epsilon,
        "rho": rho,
        "sigma": sigma,
        "gamma": gamma,
        "pi": pi,
        "phi": phi,
        "lamb": lamb,
        "omega": omega,
        "ratio_sumK_sumD": ratio_sumK_sumD,
        "ratio_maxK_minD": ratio_maxK_minD,
        "ratio_minK_maxD": ratio_minK_maxD,
        "ratio_geom_K_D": ratio_geom_K_D,
        "ratio_cvK_cvD": ratio_cvK_cvD,
        "diff_cvK_cvD": diff_cvK_cvD,
        "log_diff": log_diff,
        "ratio_median_K_D": ratio_median_K_D,
        'var_A_norm': var_A_norm,
        'var_B_norm': var_B_norm, 
        'var_C_norm': var_C_norm, 
        'ratio_var_AB': ratio_var_AB,
        'ratio_var_BC': ratio_var_BC,
        'ratio_var_AC': ratio_var_AC,
        'total_variance': total_variance,
        'var_span': var_span,
        'frac_var_A': frac_var_A,
        'frac_var_B': frac_var_B, 
        'frac_var_C': frac_var_C,
        'frac_rx_entropy': frac_rx_entropy,
        'frac_diff_entropy': frac_diff_entropy,
        'ratio_rx_diff_entropy': ratio_rx_diff_entropy,
        'frac_entropy_at_stab': frac_entropy_at_stab,
        'stabilization_time': stabilization_time,
    }



    columns = [
        "index",
        "delta",
        "tau",
        "Keq1",
        "Keq2",
        "Keq3",
        "dt",
        "dx",
        "ratio_sumK_sumD", 
        "ratio_maxK_minD",
        "ratio_minK_maxD",
        "ratio_geom_K_D", 
        "ratio_cvK_cvD",  
        "diff_cvK_cvD",
        "log_diff",
        "ratio_median_K_D",
        "eta",
        "mu",
        "kappa",
        "nu",
        "epsilon",
        "rho",
        "sigma",
        "gamma",
        "pi",
        "phi",
        "lamb",
        "omega",
        "culumative_entropy",
        "culumative_rx_entropy",
        "culumative_diff_entropy",
        'frac_rx_entropy',
        'frac_diff_entropy',
        'ratio_rx_diff_entropy',
        "max_entropy",
        "entropy_value_at_stab",
        'frac_entropy_at_stab', 
        "entropy_start_end_ratio",
        "stabilization_step",
        'stabilization_time',
        "variance_A_final",
        "variance_B_final",
        "variance_C_final",
#        "avg_mole_frac_a",
#        "avg_mole_frac_b",
#        "avg_mole_frac_c",
        "max_Da_A",
        "max_Da_B",
        "max_Da_C",
        'var_A_norm',
        'var_B_norm',
        'var_C_norm',
        'total_variance',
        'var_span',
        'frac_var_A',
        'frac_var_B',
        'frac_var_C',
        'ratio_var_AB',
        'ratio_var_BC',
        'ratio_var_AC',
        "c_init",
        "rate_scalar1",
        "rate_scalar2",
        "rate_scalar3",
    ]


    # Create a DataFrame for the current result
    results_df = pd.DataFrame([result], columns=columns)

    # Check if the file exists to handle the header
    file_exists = os.path.exists('simulation_results.csv')

    # Write the current result to the CSV file
    results_df.to_csv('simulation_results.csv', mode='a', header=not file_exists, index=False)

def factorial_simulation(factorial_design, k, A, B, C):
    #    results = []  # To store results for each parameter set
    index = 0
    # Loop over each parameter set in the factorial design
    for Keq1, Keq2, Keq3, delta, tau in factorial_design:


        # Print or log the current parameter set
        print(
            f"Running simulation [{index}] with delta={delta}, tau={tau}, Keq1={Keq1}, Keq2={Keq2}, Keq3={Keq3}"
        )
        simulation(
            A,
            B,
            C,
            Keq1,
            Keq2,
            Keq3,
            delta,
            tau,
            k,
            index,
        )
        index += 1


def simulation(
    A,
    B,
    C,
    Keq1,
    Keq2,
    Keq3,
    delta,
    tau,
    k,
    index,
):
    """
    Simulate the reaction-diffusion system and generate frames for animation.
    """

    frames = []
    totalCA, totalCB, totalCC, totalC = (
        [],
        [],
        [],
        [],
    )  # List holding the total (dimensionless) concentration of A for each iteration
    maxcAlist, mincAlist, maxcBlist, mincBlist, maxcClist, mincClist = (
    	[],
    	[],
    	[],
    	[],
    	[],
    	[],
    )	
    
    varAlist, varBlist, varClist = (
    	[],
    	[],
    	[],
    )	
    
    iter = []
    entropy = []
    rx_entropy, diff_entropy = [], []
    a_entropy_diff, b_entropy_diff, c_entropy_diff = [], [], []
    a_entropy_rx, b_entropy_rx, c_entropy_rx = [], [], []
    (
        entropy_grad_a_squared,
        entropy_grad_b_squared,
        entropy_grad_c_squared,
    ) = ([], [], [])
    stabilization_step = None
    stabilization_time = None

    # Variables for dissociation:
    mu, sigma, omega = rate_scalar1 / Keq1, rate_scalar2 / Keq2, rate_scalar3 / Keq3
    nu, pi, eta = mu, sigma, omega

    # Variables for association:
    epsilon, kappa, gamma = (
        rate_scalar1 * Keq1,
        rate_scalar2 * Keq2,
        rate_scalar3 * Keq3,
    )
    rho, lamb, phi = epsilon, gamma, kappa
    
    if dt > dt_max:
        print(
            f"Warning: dt ({dt}) exceeds stability limit ({dt_max}). Consider reducing dt."
        )

    for i in tqdm(range(k), desc="Generating frames"):
        (
            a,
            b,
            c,
            total_conc,
            entropy_value,
            reaction_entropy,
            diffusion_entropy,
            a_entropy_difff,
            b_entropy_difff,
            c_entropy_difff,
            term1,
            term2,
            term3,
            grad_a_squared,
            grad_b_squared,
            grad_c_squared,
            max_a,
            min_a,
            max_b,
            min_b,
            max_c,
            min_c,
            var_a,
            var_b,
            var_c,
        ) = RxDiffusion(
            A,
            B,
            C,
            eta,
            mu,
            kappa,
            nu,
            epsilon,
            rho,
            sigma,
            delta,
            gamma,
            pi,
            phi,
            tau,
            lamb,
            omega,
            i,
            )

        totalCA.append(total_conc[0])
        totalCB.append(total_conc[1])
        totalCC.append(total_conc[2])
        totalC.append(total_conc[3])
        iter.append(i)
        maxcAlist.append(max_a/total_conc[3])
        mincAlist.append(min_a/total_conc[3])
        maxcBlist.append(max_b/total_conc[3])
        mincBlist.append(min_b/total_conc[3])
        maxcClist.append(max_c/total_conc[3])
        mincClist.append(min_c/total_conc[3])
        varAlist.append(var_a)
        varBlist.append(var_b)
        varClist.append(var_c)
        

        if len(iter) > 40:
            entropy.append(entropy_value)
            rx_entropy.append(reaction_entropy)
            diff_entropy.append(diffusion_entropy)
            a_entropy_diff.append(a_entropy_difff)
            b_entropy_diff.append(b_entropy_difff)
            c_entropy_diff.append(c_entropy_difff)
            a_entropy_rx.append(term1)
            b_entropy_rx.append(term2)
            c_entropy_rx.append(term3)
            entropy_grad_a_squared.append(grad_a_squared)
            entropy_grad_b_squared.append(grad_b_squared)
            entropy_grad_c_squared.append(grad_c_squared)
            
       

        else:
            entropy.append(0)
            rx_entropy.append(0)
            diff_entropy.append(0)
            a_entropy_diff.append(0)
            b_entropy_diff.append(0)
            c_entropy_diff.append(0)
            a_entropy_rx.append(0)
            b_entropy_rx.append(0)
            c_entropy_rx.append(0)
            entropy_grad_a_squared.append(0)
            entropy_grad_b_squared.append(0)
            entropy_grad_c_squared.append(0)
        

        frames.append(create_image(a, b, c))

    """Parameter relations"""
    K = [Keq1, Keq2, Keq3]
    D = [1, tau, delta]

    

    if len(entropy) > 40 and stabilization_step is None:
        stab_step, stab_value = find_entropy_stabilization_step(
            entropy[41:], threshold=0.01, window_size=10
        )
        if stab_step is not None:
            stabilization_step = stab_step
    #                stabilization_time = stabilization_step * dt
    #                print(f"Entropy stabilizes at iteration {stabilization_step}")

    plot_entopy_totalC(
        totalCA,
        totalCB,
        totalCC,
        totalC,
        iter,
        entropy,
        rx_entropy,
        diff_entropy,
        index,
    )
    plot_dconc(totalCA, totalCB, totalCC, totalC, index)
    plot_entropy_loadings(
        a_entropy_diff,
        b_entropy_diff,
        c_entropy_diff,
        a_entropy_rx,
        b_entropy_rx,
        c_entropy_rx,
        iter,
        index,
    )
    plot_diffusive_terms(
        entropy_grad_a_squared,
        entropy_grad_b_squared,
        entropy_grad_c_squared,
        iter,
        index,
    )
    
    plot_additional_data(
    	iter,
    	varAlist,
    	varBlist,
    	varClist,
    	maxcAlist,
    	mincAlist,
    	maxcBlist,
    	mincBlist,
    	maxcClist,
    	mincClist,
    	A,
    	B,
    	C,
    	index
    )
    selected_frames = frames[::steps_per_frame]  # Every third frame
    gif_frames = [
        Image.fromarray(frame) for frame in selected_frames
    ]  # Convert to PIL Images
    gif_filename = f"animation_{index}.gif"  # Unique filename for each iterationi
    gif_frames[0].save(
        gif_filename, save_all=True, append_images=gif_frames[1:], duration=gif_duration, loop=0
    )

    extract_results(
        entropy,
        rx_entropy,
        diff_entropy,
        stabilization_step,
        a,
        b,
        c,
#        totalCA,
#        totalCB,
#        totalCC,
#        totalC,
        term1,
        term2,
        term3,
        index,
        Keq1,
        Keq2,
        Keq3,
        tau,
        delta,
        c_A_init,
        rate_scalar1,
        rate_scalar2,
        rate_scalar3,
        eta,
        mu,
        kappa,
        nu,
        epsilon,
        rho,
        sigma,
        gamma,
        pi,
        phi,
        lamb,
        omega,
        K,
        D,
    )



run_program()
