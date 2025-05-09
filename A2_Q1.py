# Import Packages
import numpy as np
import matplotlib.pyplot as plt

def Numerical_sol_Transient(dt, f, total_time):
    # Problem Paramters
    L = 1.0         
    dx = 0.05      
    
    # Number of nodes
    nx = int(L/dx) + 1
    
    alpha = 1.0     # Thermal diffusivity (m^2/hr)
    
    # Time parameters
    nt = int(total_time / dt)   # Time steps
    record_times = np.arange(0, total_time + dt, 0.1)  # Record at every 0.1 hr
    
    # Initial and boundary conditions
    phi = np.ones(nx) * 100   
    phi_old = np.ones(nx) * 100
    phi[0] = 300              
    phi[-1] = 300             
    
    x = np.linspace(0, L, nx)
    
    # Set up coefficients
    Ae = np.zeros(nx)
    Aw = np.zeros(nx)
    Ap = np.zeros(nx)
    d  = np.zeros(nx)
    
    Ap0 = dx / (dt * alpha) 
    
    for i in range(1, nx-1):
         Ae[i] = alpha / dx
         Aw[i] = alpha / dx
    
    Ae[nx-1] = 2*alpha/dx
    Aw[0] = 2*alpha/dx
    
    for i in range(nx):
         d[i]  = Ap0 - ((1 - f) * (Ae[i] + Aw[i]))
         Ap[i] = (f * (Ae[i] + Aw[i])) + Ap0
        
        
    # Convergence criterion
    tol = 1e-3
    max_iter = 1000000
    
    # Dictionary to record solutions at the desired times.
    solutions = {}
    solutions[0.0] = phi.copy()  # store initial condition
    
    # Time Marching
    for t in range(1, nt + 1):
         iteration = 0
         # Use phi_new to store the iterated solution
         phi_new = phi.copy()
         while iteration < max_iter:
              phi_temp = phi_new.copy()  # store current iteration results
              # Update all interior nodes in one full sweep
              for i in range(1, nx-1):
                   phi_new[i] = (Ae[i] * (f * phi_temp[i+1] + (1 - f) * phi_old[i+1]) + Aw[i] * (f * phi_temp[i-1] + (1 - f) * phi_old[i-1]) + d[i] * phi_old[i]) / Ap[i]
              # Check convergence
              if np.max(np.abs(phi_new - phi_temp)) < tol:
                   print(f"\n Number of iterations for {t}th timestep: {iteration}")
                   break
              iteration += 1
         # Update phi with the converged solution
         phi = phi_new.copy()
         phi_old = phi.copy()
         
         current_time = t * dt
         # Record the solution if the current time matches one of the record times
         if np.any(np.isclose(current_time, record_times, atol=dt/10)):
              solutions[current_time] = phi.copy()
              
    return solutions, x

# Define a function for analytical solution
def analytical_solution(t, n_terms=20):
     
    # Problem Parameters
    alpha = 1
    L = 1.0
    dx = 0.05
    
    # Number of nodes
    nx = int(L/dx) + 1
    
    x = np.linspace(0, L, nx)
    
    # Given 
    T_i = 100
    T_s = 300
    
    theta = np.zeros_like(x, dtype=float)
    for k in range(0, n_terms):
        m = (2*k) + 1  # odd 
        # Each term in the series
        term = (4 * (T_i - T_s) * np.sin(m * np.pi * x / L) * np.exp(-alpha * t * ((m * np.pi / L)**2))) / (m * np.pi)
        theta += term # Add each term
    return T_s + theta # Final Analytical solution

if __name__ == "__main__":
    dt = 0.001 # time step in hr
    schemes = [0,0.5,1]  # f values: explicit, Crank–Nicolson, and implicit
    # To see solution for one scheme just remove other 2 f values from the list
    
    # Run for each scheme and store the recorded solutions
    scheme_results = {}
    for scheme in schemes:
         sol, x = Numerical_sol_Transient(dt, scheme, 0.5)
         scheme_results[scheme] = sol

    record_times = np.arange(0, 0.5 + dt, 0.1)
    
    # Plot 
    for t in record_times:
         plt.figure(figsize=(8, 6))
         for scheme in schemes:
              times = np.array(list(scheme_results[scheme].keys()))
              idx = (np.abs(times - t)).argmin()
              closest_time = times[idx]
              phi = scheme_results[scheme][closest_time]
              plt.plot(x, phi, label=f"f = {scheme}")
              plt.ticklabel_format(style='plain', useOffset=False, axis='y')
         phi_analytical = analytical_solution(t, n_terms=50)
         plt.plot(x, phi_analytical, 'k--', linewidth=2, label="Analytical")
         plt.xlabel("x (m)")
         plt.ylabel("Temperature (°C)")
         plt.title(f"Temperature distribution at t = {t:.1f} hr")
         plt.legend()
         plt.grid(True)
         plt.show()
