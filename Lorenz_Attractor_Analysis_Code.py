import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# Note: Figures will save to the folder that holds the code, but will be overwritten everytime you run the code

############################################################################################################################

# Systems

def lorenz(x, y, z, t, params):
    dx_dt = params[0]*(y-x)
    dy_dt = params[1]*x - y - x*z
    dz_dt = x*y - params[2]*z
    
    L_lorenz = [dx_dt, dy_dt, dz_dt]
    return L_lorenz

def rabinovich_fabrikant(x, y, z, t, params):
    dx_dt = y * (z - 1 + x**2) + params[1]*x
    dy_dt = x * (3*z + 1 - x**2) + params[1]*y
    dz_dt = -2*z * (params[0] + x*y)

    L_rf = [dx_dt, dy_dt, dz_dt]
    return L_rf

def chen(x, y, z, t, params):

    dx_dt = params[0]*(y - x)
    dy_dt = (params[2] - params[0])*x - x*z + params[2]*y
    dz_dt = x*y - params[1]*z

    L_ds = [dx_dt, dy_dt, dz_dt]
    return L_ds

def henon(init, params, num_steps_to_stop):
    
    points = np.empty((2, num_steps_to_stop))
    x = init[0]
    y = init[1]    
    points[0][0] = x # IC's
    points[1][0] = y

    for i in range(1, num_steps_to_stop-2):
        x_next = 1 - params[0]*x**2 + y
        y_next = params[1]*x

        points[0, i] = x_next
        points[1, i] = y_next

        x = x_next
        y = y_next

    plt.figure()
    plt.scatter(points[0], points[1], s=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Hénon System")

    return

def modified_chua(x, y, z, t, params):

    if x >= 2 * params[2] * params[4]:
        f = params[3] * np.pi / (2*params[2]) * (x - 2 * params[2] * params[4])
    elif -2 * params[2] * params[4] < x < 2 * params[2] * params[4]:
        f = -params[3] * np.sin(np.pi * x / (2*params[2]) + params[5])
    else:
        f = params[3] * np.pi / (2*params[2]) * (x + 2 * params[2] * params[4])

    dx_dt = params[0]*(y - f)
    dy_dt = x - y + z
    dz_dt = -params[1]*y

    L_m_chua = [dx_dt, dy_dt, dz_dt]
    return L_m_chua

def duffing(x, y, z, t, params):

    dx_dt = y
    dy_dt = -params[3]*y - params[0]*x - params[1]*x**3 + params[2]*np.cos(params[4]*t)
    dz_dt = 0

    L_duffing = [dx_dt, dy_dt, dz_dt]

    return L_duffing

############################################################################################################################

# eulers method

def eulers_method(init, system, sub_method, dt, params, num_steps_to_stop):

    # x, y, z
    values = np.zeros((3, num_steps_to_stop+1))
    values[0][0] = init[0]
    values[1][0] = init[1]
    values[2][0] = init[2]

    # get each value at every
    for i in range(num_steps_to_stop):
        dx_dt, dy_dt, dz_dt = system(values[0][i], values[1][i], values[2][i], (i+1)*dt, params)

        # y_n+1 = y_n + h*(dy_dx|x)
        values[0][i+1] = values[0][i] + dt*dx_dt
        values[1][i+1] = values[1][i] + dt*dy_dt
        values[2][i+1] = values[2][i] + dt*dz_dt

    return values[0], values[1], values[2]

############################################################################################################################

# improved EM

def improved_eulers_method(init, system, sub_method, dt, params, num_steps_to_stop):

    values = np.zeros((3, num_steps_to_stop+1))
    values[0][0] = init[0]
    values[1][0] = init[1]
    values[2][0] = init[2]

    for i in range(num_steps_to_stop):

        dx1, dy1, dz1 = system(values[0][i], values[1][i], values[2][i], (i+1)*dt, params)


        x_pred = values[0][i] + dt * dx1
        y_pred = values[1][i] + dt * dy1
        z_pred = values[2][i] + dt * dz1


        dx2, dy2, dz2 = system(x_pred, y_pred, z_pred, (i+1)*dt, params)


        x_next = values[0][i] + (dt / 2) * (dx1 + dx2)
        y_next = values[1][i] + (dt / 2) * (dy1 + dy2)
        z_next = values[2][i] + (dt / 2) * (dz1 + dz2)

        values[0][i+1] = x_next
        values[1][i+1] = y_next
        values[2][i+1] = z_next

    return values[0], values[1], values[2]

############################################################################################################################

# RK4

def runge_kutta_4(f, t, params, y, h):
    k1 = f(y[0], y[1], y[2], t, params)
    k2 = f(y[0] + h * k1[0] / 2.0, y[1] + h * k1[1] / 2.0, y[2] + h * k1[2] / 2.0, t, params)
    k3 = f(y[0] + h * k2[0] / 2.0, y[1] + h * k2[1] / 2.0, y[2] + h * k2[2] / 2.0, t, params)
    k4 = f(y[0] + h * k3[0], y[1] + h * k3[1], y[2] + h * k3[2], t, params)
    slopes = [y[0] + (h/6.0) * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]), 
              y[1] + (h/6.0) * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]), 
              y[2] + (h/6.0) * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])]
    return slopes

############################################################################################################################

# RK8

def runge_kutta_8(system, t, params, points, dt):

    y_vec = np.array(points, dtype=float)

    def f(t, Y): 
        dx, dy, dz = system(Y[0], Y[1], Y[2], 0, params)
        return np.array([dx, dy, dz], dtype=float)

    y_next = rk8_dop853_step(f, 0.0, y_vec, dt)
    return float(y_next[0]), float(y_next[1]), float(y_next[2])

def rk8_dop853_step(f, t, y, h):
    c = np.array([
        0.0,
        1/18,
        1/12,
        1/8,
        5/16,
        3/8,
        59/400,
        93/200,
        5490023248/9719169821,
        13/20,
        1201146811/1299019798,
        1.0,
        1.0,
    ], dtype=float)

    a = np.zeros((13, 12), dtype=float)

    a[1, 0] = 1/18

    a[2, 0] = 1/48
    a[2, 1] = 1/16

    a[3, 0] = 1/32
    a[3, 2] = 3/32

    a[4, 0] = 5/16
    a[4, 2] = -75/64
    a[4, 3] = 75/64

    a[5, 0] = 3/80
    a[5, 3] = 3/16
    a[5, 4] = 3/20

    a[6, 0] = 29443841/614563906
    a[6, 3] = 77736538/692538347
    a[6, 4] = -28693883/1125000000
    a[6, 5] = 23124283/1800000000

    a[7, 0] = 16016141/946692911
    a[7, 3] = 61564180/158732637
    a[7, 4] = 22789713/633445777
    a[7, 5] = 545815736/2771057229
    a[7, 6] = -180193667/1043307555

    a[8, 0] = 39632708/573591083
    a[8, 3] = -433636366/683701615
    a[8, 4] = -421739975/2616292301
    a[8, 5] = 100302831/723423059
    a[8, 6] = 790204164/839813087
    a[8, 7] = 800635310/3783071287

    a[9, 0] = 246121993/1340847787
    a[9, 3] = -37695042795/15268766246
    a[9, 4] = -309121744/1061227803
    a[9, 5] = -12992083/490766935
    a[9, 6] = 6005943493/2108947869
    a[9, 7] = 393006217/1396673457
    a[9, 8] = 123872331/1001029789

    a[10, 0] = -1028468189/846180014
    a[10, 3] = 8478235783/508512852
    a[10, 4] = 1311729495/1432422823
    a[10, 5] = -10304129995/1701304382
    a[10, 6] = -48777925059/3047939560
    a[10, 7] = 15336726248/1032824649
    a[10, 8] = -45442868181/3398467696
    a[10, 9] = 3065993473/597172653

    a[11, 0]  = 185892177/718116043
    a[11, 3]  = -3185094517/667107341
    a[11, 4]  = -477755414/1098053517
    a[11, 5]  = -703635378/230739211
    a[11, 6]  = 5731566787/1027545527
    a[11, 7]  = 5232866602/850066563
    a[11, 8]  = -4093664535/808688257
    a[11, 9]  = 3962137247/1805957418
    a[11,10] = 65686358/487910083

    a[12, 0]  = 403863854/491063109
    a[12, 3]  = -5068492393/434740067
    a[12, 4]  = -411421997/543043805
    a[12, 5]  = 652783627/914296604
    a[12, 6]  = 11173962825/925320556
    a[12, 7]  = -13158990841/6184727034
    a[12, 8]  = 3936647629/1978049680
    a[12, 9]  = -160528059/685178525
    a[12,10] = 248638103/1413531060

    b = np.array([
        14005451/335480064,
        0.0,
        0.0,
        0.0,
        0.0,
        -59238493/1068277825,
        181606767/758867731,
        561292985/797845732,
        -1041891430/1371343529,
        760417239/1151165299,
        118820643/751138087,
        -528747749/2220607170,
        1/4
    ], dtype=float)

    k = []
    for i in range(13):
        yi = y
        if i > 0:
            s = np.zeros_like(y)
            for j in range(i):
                s += a[i, j] * k[j]
            yi = y + h * s

        k.append(f(t + c[i] * h, yi))

    incr = np.zeros_like(y)
    for i in range(13):
        incr += b[i] * k[i]

    return y + h * incr

############################################################################################################################

# RK_integrate

def runge_kutta(init, system, sub_method, dt, params, num_steps_to_stop):
    x = np.zeros(num_steps_to_stop+1)
    y = np.zeros(num_steps_to_stop+1)
    z = np.zeros(num_steps_to_stop+1)
    x[0], y[0], z[0] = init

    for i in range(num_steps_to_stop):
        t = dt*(i+1)
        x[i+1], y[i+1], z[i+1] = sub_method(system, t, params, [x[i], y[i], z[i]], dt)

    return x, y, z

############################################################################################################################

# model

def model(method, init, system, sub_method, dt, params, num_steps_to_stop):
    x, y, z = method(init, system, sub_method, dt, params, num_steps_to_stop)
    return x, y, z

############################################################################################################################

# plotting functions:

def plot_system(x, y, z, system_name, pc, p_axis, plane):

    fig = plt.figure(dpi=200)
    ax = fig.add_subplot(projection='3d')
    ax.plot(x, y, z, linewidth=0.1)
    ax.set_title(f"{system_name}", y=0.9)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('w')
    ax.yaxis.pane.set_edgecolor('w')
    ax.zaxis.pane.set_edgecolor('w')
    ax.grid(False)
    sys_name_nospace = nospace(system_name)
    plt.savefig(f"figures/3D/{sys_name_nospace}")
    plt.savefig(f"figures/3D/{sys_name_nospace}.eps")
    print(f"The plot of {system_name} has been saved to figures/3D/")

    if pc: # plot poincare intersection plane with the lorenz system
        x_temp = np.arange(start=np.min(x)-10, stop=np.max(x)+10, step=0.1)
        y_temp = np.arange(start=np.min(y)-10, stop=np.max(y)+10, step=0.1)
        z_temp = np.arange(start=np.min(z)-10, stop=np.max(z)+10, step=0.1)

        if p_axis == 'z':
            X, Y = np.meshgrid(x_temp, y_temp)
            Z = plane + 0*X + 0*Y
            ax.plot_surface(X, Y, Z, color='silver', alpha=0.7)
        elif p_axis == 'y':
            X, Z = np.meshgrid(x_temp, z_temp)
            Y = plane + 0*X + 0*Z
            ax.plot_surface(X, Y, Z, color='silver', alpha=0.7)        
        elif p_axis == 'x':
            Y, Z = np.meshgrid(y_temp, z_temp)
            X = plane + 0*Y + 0*Z
            ax.plot_surface(X, Y, Z, color='silver', alpha=0.7)

        ax.set_title(f"Poincare Plane Intersection for the Plane {p_axis} = {plane} of the {system_name}")
        sys_name_nospace = nospace(system_name)
        plt.savefig(f"figures/analysis/Poincare_Maps/Poincare_Plane_Intersection_{sys_name_nospace}")
        plt.savefig(f"figures/analysis/Poincare_Maps/Poincare_Plane_Intersection_{sys_name_nospace}.eps")
        print(f"The Poincare plane intersection plot for the Plane {p_axis} = {plane} of the {system_name} has been saved to figures/analysis/Poincare_Maps/")

    return

def plot_xy_proj(x, y, system_name):

    fig, ax = plt.subplots(dpi=200) # necessary to plot on seperate windows/graphs
    plt.plot(x, y, linewidth = 0.05)
    plt.title(f"{system_name} - xy Projection")
    plt.xlabel("x")
    plt.ylabel("y")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sys_name_nospace = nospace(system_name)
    plt.savefig(f"figures/projections/xy/xy_proj_{sys_name_nospace}")
    plt.savefig(f"figures/projections/xy/xy_proj_{sys_name_nospace}.eps")
    print(f"The xy projection of the {system_name} has been saved to figures/projections/xy/")

    return

def plot_xz_proj(x, z, system_name):

    fig, ax = plt.subplots(dpi=200) # necessary to plot on seperate windows/graphs
    plt.plot(x, z, linewidth = 0.05)
    plt.title(f"{system_name} - xz Projection")
    plt.xlabel("x")
    plt.ylabel("z")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sys_name_nospace = nospace(system_name)
    plt.savefig(f"figures/projections/xz/xz_proj_{sys_name_nospace}")
    plt.savefig(f"figures/projections/xz/xz_proj_{sys_name_nospace}.eps")
    print(f"The xz projection of the {system_name} has been saved to figures/projections/xz/")

    return

def plot_yz_proj(y, z, system_name):

    fig, ax = plt.subplots(dpi=200) # necessary to plot on seperate windows/graphs
    plt.plot(y, z, linewidth = 0.05)
    plt.title(f"{system_name} - yz Projection")
    plt.xlabel("y")
    plt.ylabel("z")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sys_name_nospace = nospace(system_name)
    plt.savefig(f"figures/projections/yz/yz_proj_{sys_name_nospace}")
    plt.savefig(f"figures/projections/yz/yz_proj_{sys_name_nospace}.eps")
    print(f"The yz projection of the {system_name} has been saved to figures/projections/yz/")

    return

############################################################################################################################

#Gramm-Schmidt Orthonormalization for Lyapunov Exponents

def GS_ortho(x, y, z, params, dt, num_steps_to_stop, discard):

    # Computation values

    basis = np.eye(3) # standard basis for R3
    
    I = np.array([x[0]*basis[0], 
                  y[0]*basis[1], 
                  z[0]*basis[2]]) # initial value matrix under the standard basis for R3
    
    l = np.array([0., 0., 0.]) # vector of lyapunov spectrum

    # J is initialized in the loop
    Y = I # Y = I for the first step
    T = 0

    for i in range(num_steps_to_stop):
        # integrate over (t, t + dt) ########
        # x' = f(x) integrating over this just gets our next point 

        '''
        J = [-s,    s,  0
             r-z, -1,  x
             y,    x, -B]
        '''
        J = np.array([[    -params[0],     params[0],         0.],
                      [params[1]-z[i],            -1,      -x[i]],
                      [          y[i],          x[i], -params[2]]])
        
        # Yprime = J @ Y = f(Y)
        # Integrate: s1 = Y[i] @ J
        #            s2 = J @ (Y[i] + dt*(Y[i] @ J)) = J @ (Y[i] + dt*(s1))
        #            Y[i+1] = Y[i] + dt * (s1 + s2) / 2

        s1 = J @ Y
        s2 = J @ (Y + dt * (s1))
        Y = Y + dt * (s1 + s2) / 2

        Q, R = np.linalg.qr(Y)

        if i >= discard: # discard some initial amount to avoid ln(0) errors
            l += np.log(np.abs(np.diag(R)))
            T += dt

        Y = Q
        

    return l/T

############################################################################################################################

# Orbit Separation

def orbsep(method, sub_method, system, init, d0, plot_running, dt, num_steps_to_stop, params, discard, system_name):

    x_discard, y_discard, z_discard = method(init, system, sub_method, dt, params, 300)

    x_a = np.array([x_discard[-1], y_discard[-1], z_discard[-1]])
    
    d = np.random.normal(size=3)
    d = d / np.linalg.norm(d) * d0
    x_b = x_a + d

    log_sum = 0.0
    running_avg = []

    for i in range(num_steps_to_stop):

        xa1, ya1, za1 = method(x_a, system, sub_method, dt, params, 1)
        xb1, yb1, zb1 = method(x_b, system, sub_method, dt, params, 1)

        xa1, ya1, za1 = xa1[1], ya1[1], za1[1]  
        xb1, yb1, zb1 = xb1[1], yb1[1], zb1[1]

        diff = np.array([xb1 - xa1, yb1 - ya1, zb1 - za1])
        d1 = np.linalg.norm(diff)
        d1 = max(d1, 1e-20)

        
        logd = np.log(d1 / d0)
        if i >= discard:
            log_sum += logd
            running_avg.append(log_sum / (i - discard + 1) / dt)  

        
        x_b = np.array([xa1, ya1, za1]) + d0 * diff / d1
        x_a = np.array([xa1, ya1, za1])
    
    if plot_running == 'y':
        x_vals = np.empty_like(running_avg)
        for i in range(len(x_vals)):
            x_vals[i] = dt*i + discard*dt
        plt.figure()
        plt.ylabel("Running Lyaponuv Values")
        plt.xlabel("Time")
        plt.title(f"Running Lyapunov Values vs. Time for the {system_name}")
        plt.plot(x_vals, running_avg)
        sys_name_nospace = nospace(system_name)
        plt.savefig(f"figures/analysis/Lyapunov_Exponents/Orbit_Separation_Running_Lyaponuv_Values_{sys_name_nospace}")
        plt.savefig(f"figures/analysis/Lyapunov_Exponents/Orbit_Separation_Running_Lyaponuv_Values_{sys_name_nospace}.eps")
        print(f"The running Lyapunov values plot for the {system_name} has been saved to figures/analysis/Lyapunov_Exponents/")

    return running_avg[-1]

############################################################################################################################

# Average Lyapunov Exponent Calculation

def avg_lyapunov(lyapunov_method, params, system, method, sub_method, dt, num_steps_to_stop, d0, discard, num_iterations):

    if lyapunov_method == GS_ortho:

        exponents = np.zeros((num_iterations, 3)) # rows, columns

        for i in range(num_iterations):
            init = np.random.uniform(0.0, 100.0, 3) # generate random initial conditions each time on the interval [0.0, 100.0)
            x, y, z = model(method, init, system, sub_method, dt, params, num_steps_to_stop)
            exponents[i] = lyapunov_method(x, y, z, params, dt, num_steps_to_stop, discard) # at i-th row, put in the i-th caluclation values for lyapunov exponents
        
        average = np.mean(exponents, 0) # produces the mean of each column and stores in a a 1x3 average array
        uncertainty = np.std(exponents, 0) # produces the standard deviation of each column and stores in a a 1x3 average array

        return f"Average Lyapunov Spectrum = {average} \u00B1 {uncertainty}"


    elif lyapunov_method == orbsep:

        exponents = np.empty((num_iterations, 1))

        for i in range(num_iterations):
            init = np.random.uniform(0.0, 100.0, 3) # generate random initial conditions on the interval [0.0, 100.0) each iteration 
            exponents[i] = lyapunov_method(method, sub_method, system, init, d0, 'n', dt, num_steps_to_stop, params, discard, None)
        
        average = np.mean(exponents, 0) # produces the mean of each column and stores in a a 1x3 average array
        uncertainty = np.std(exponents, 0) # produces the standard deviation of each column and stores in a a 1x3 average array

        return f"Average Maximal Lyapunov Value = {average[0]} \u00B1 {uncertainty[0]}"

    return "Invalid Lyapunov method"

############################################################################################################################

# Poincare Maps

def poincare(x, y, z, system_name):

    axis = (input("Input axis of intersecting plane (x, y or z): ")).lower()
    plane = int(input("Input equation for constant plane of intersection: "))
    crossings_intsec1, crossings_intsec2 = np.array([]), np.array([])

    if axis == 'x':
        axis_vals = x
        intsec1, intsec_axis1 = y, 'y'
        intsec2, intsec_axis2 = z, 'z'
    elif axis == 'y':
        axis_vals = y
        intsec1, intsec_axis1 = x, 'x'
        intsec2, intsec_axis2 = z, 'z'
    elif axis == 'z':
        axis_vals = z
        intsec1, intsec_axis1 = x, 'x'
        intsec2, intsec_axis2 = y, 'y'

    for i in range(len(axis_vals) - 1):
        if (axis_vals[i] - plane) * (axis_vals[i + 1] - plane) < 0 and axis_vals[i] < axis_vals[i + 1]:
            ratio = (plane - axis_vals[i]) / (axis_vals[i + 1] - axis_vals[i])
            intsec1_cross = intsec1[i] + ratio * (intsec1[i + 1] - intsec1[i])
            intsec2_cross = intsec2[i] + ratio * (intsec2[i + 1] - intsec2[i])
            crossings_intsec1 = np.append(crossings_intsec1, intsec1_cross)
            crossings_intsec2 = np.append(crossings_intsec2, intsec2_cross)

    sys_name_nospace = nospace(system_name)

    plt.figure(figsize=(6, 6))
    plt.scatter(crossings_intsec1, crossings_intsec2, s=10, color='blue')
    plt.title(f"Poincaré Section of the {system_name} at {axis} = {plane}")
    plt.xlabel(intsec_axis1)
    plt.ylabel(intsec_axis2)
    plt.grid(True)
    plt.savefig(f"figures/analysis/Poincare_Maps/Poincare_Map_{sys_name_nospace}_{axis}={plane}")
    plt.savefig(f"figures/analysis/Poincare_Maps/Poincare_Map_{sys_name_nospace}_{axis}={plane}.eps")
    print(f"The Poincare Map for the {system_name} at {axis}={plane} has been saved to figures/analysis/Poincare_Maps/")

    plot_system(x, y, z, system_name, True, axis, plane)

    intsec_points = [[intsec_axis1, intsec_axis2]]

    with open(f"figures/analysis/Poincare_Maps/Intersection_Points/poincare_intersections_{sys_name_nospace}_{axis}={plane}.txt", "w") as write_file:
        write_file.write(str(intsec_points[0])+'\n')
        for j in range(1, len(crossings_intsec1)+1):
            intsec_points.append([float(crossings_intsec1[j-1]), float(crossings_intsec2[j-1])])
            write_file.write(str(intsec_points[j])+'\n')
        print(f"The intersection points for the {system_name} at {axis}={plane} have been saved to figures/analysis/Poincare_Maps/Intersection_Points/")

    return

############################################################################################################################

# Richardson Extrapolation for error analysis

def richardson_extrapolation(sol_h, init, plot, model, method, system, sub_method, dt, params, num_steps_to_stop, log_scale, system_name):
    
    x2, y2, z2 = model(method, init, system, sub_method, dt/2, params, num_steps_to_stop*2)
    sol_h_2 = np.array([x2, y2, z2])

    local_errors = np.empty((num_steps_to_stop))

    if method == eulers_method:
        model_str = "EM"
        order = 1
    elif method == improved_eulers_method:
        model_str = "IEM"
        order = 2
    elif sub_method == runge_kutta_4:
        model_str = "RK4"
        order = 4
    else:
        model_str = "RK8"
        order = 8

    for i in range(int(num_steps_to_stop)):
        diff_x = sol_h[0][i] - sol_h_2[0][2*i]
        diff_y = sol_h[1][i] - sol_h_2[1][2*i]
        diff_z = sol_h[2][i] - sol_h_2[2][2*i]

        diff = (diff_x**2 + diff_y**2 + diff_z**2)**0.5 # pythagoreas

        local_error = np.abs(diff / (2**order - 1))

        local_errors[i] = local_error

    # plot the error as a function of time
    
    if plot: 
        time = np.linspace(0, num_steps_to_stop*dt, num_steps_to_stop)

        plt.figure()
        plt.plot(time, local_errors)
        plt.title(f"Approximate Error of {model_str} Modelling Method, {system_name}")
        plt.xlabel("Time (s)")

        sys_name_nospace = nospace(system_name)

        if log_scale:
            plt.yscale("log")
            plt.ylabel("Error, Logarithmic Scale")
            plt.legend(loc="lower right")
            plt.savefig(f"figures/analysis/Modelling_Error/Richardson_Extrapolation_Local_Trucation_Error_log_{sys_name_nospace}")
            plt.savefig(f"figures/analysis/Modelling_Error/Richardson_Extrapolation_Local_Trucation_Error_log_{sys_name_nospace}.eps")
            print(f"The error plot for the {system_name} on a log scale has been saved to figures/analysis/Modelling_Error/")
        else:
            plt.ylabel("Error")
            plt.legend(loc="upper left")
            plt.savefig(f"figures/analysis/Modelling_error/Richardson_Extrapolation_Local_Trucation_Error_{sys_name_nospace}") 
            plt.savefig(f"figures/analysis/Modelling_error/Richardson_Extrapolation_Local_Trucation_Error_{sys_name_nospace}.eps")
            print(f"The error plot for the {system_name} has been saved to figures/analysis/Modelling_Error/")      

    return local_errors

############################################################################################################################

# Richardson Extrapolation Error Comparison

def RE_error_comp(init, dt, params, num_steps_to_stop, system, log_scale, system_name):

    methods = [eulers_method, improved_eulers_method, runge_kutta, runge_kutta]
    str_methods = ["EM", "IEM", "RK4", "RK8"]
    sub_methods = [None, None, runge_kutta_4, runge_kutta_8]

    time = np.linspace(0, num_steps_to_stop*dt, num_steps_to_stop)
    plt.figure()

    for i in range(len(methods)):

        x, y, z = model(methods[i], init, system, sub_methods[i], dt, params, num_steps_to_stop)
        sol_h = np.array([x, y, z])

        local_errors = richardson_extrapolation(sol_h, init, 0, model, methods[i], system, sub_methods[i], dt, params, num_steps_to_stop, log_scale, system_name)

        plt.plot(time, local_errors, label=str_methods[i])

    plt.xlabel("Time (s)")
    plt.title(f"Error Comparison for all Modelling Methods, {system_name}")
    
    sys_name_nospace = nospace(system_name)

    if log_scale:
        plt.yscale("log")
        plt.ylabel("Error, Logarithmic Scale")
        plt.legend(loc="lower right")
        plt.savefig(f"figures/analysis/Modelling_Error/Error_comparison_log_{sys_name_nospace}")
        plt.savefig(f"figures/analysis/Modelling_Error/Error_comparison_log_{sys_name_nospace}.eps")
        print(f"The error comparison plot for the {system_name} on a log scale has been saved to figures/analysis/Modelling_Error/")
    else:
        plt.ylabel("Error")
        plt.legend(loc="upper left")
        plt.savefig(f"figures/analysis/Modelling_Error/Error_comparison_{sys_name_nospace}")
        plt.savefig(f"figures/analysis/Modelling_Error/Error_comparison_{sys_name_nospace}.eps")
        print(f"The error comparison plot for the {system_name} has been saved to figures/analysis/Modelling_Error/")

    return

############################################################################################################################

# Sensitive dependence to initial conditions

def senstive_dep(init, x, y, z, params, dt, num_steps_to_stop, system, method, sub_method, d0, system_name):

    x_prime_0 = init[0] + d0 # disturb the x initial point by some small disturbance d0
    y_prime_0 = init[1]
    z_prime_0 = init[2]
    init_prime = np.array([x_prime_0, y_prime_0, z_prime_0])

    x_prime, y_prime, z_prime = model(method, init_prime, system, sub_method, dt, params, num_steps_to_stop)

    x_diff = x - x_prime
    y_diff = y - y_prime
    z_diff = z - z_prime
    time = np.linspace(0, dt*num_steps_to_stop, num_steps_to_stop+1)

    fig=plt.figure()
    fig.suptitle(f"Differences in Each Dimension for a Disturbance of x = {d0} \n for the {system_name}")

    plt.subplot(3, 1, 1)
    plt.plot(time, x_diff)
    plt.ylabel("Difference in x")
    plt.xlabel("Time")

    plt.subplot(3, 1, 2)
    plt.plot(time, y_diff)
    plt.ylabel("Difference in y")
    plt.xlabel("Time")

    plt.subplot(3, 1, 3)
    plt.plot(time, z_diff)
    plt.ylabel("Difference in z")
    plt.xlabel("Time")

    fig.tight_layout()
    sys_name_nospace = nospace(system_name)
    fig.savefig(f"figures/analysis/Sensitive_Dependance/Sensitive_Dependence_{sys_name_nospace}")
    fig.savefig(f"figures/analysis/Sensitive_Dependance/Sensitive_Dependence_{sys_name_nospace}.eps")
    print(f"The sensitive dependance plot for the {system_name} has been saved to figures/analysis/Sensitive_Dependance/")
    return

############################################################################################################################

# get system name

def get_system_name(system):

    if system == lorenz:
        return "Lorenz System"
    elif system == rabinovich_fabrikant:
        return "Rabinovich-Fabrikant System"
    elif system == chen:
        return "Chen System"
    elif system == henon:
        return "Hénon System"
    elif system == duffing:
        return "Duffing System"
    else:
        return "Modified Chua System"

def nospace(system_name):

    if system_name == "Lorenz System":
        return "Lorenz_System"
    elif system_name == "Rabinovich-Fabrikant System":
        return "Rabinovich-Fabrikant_System"
    elif system_name == "Chen System":
        return "Chen_System"
    elif system_name == "Hénon System":
        return "Hénon_System"
    elif system_name == "Duffing System":
        return "Duffing_System"
    else:
        return "Modified_Chua_System"

############################################################################################################################

# run the code

def run(init, dt, num_steps_to_stop, params, runtime, system, EM, improved_EM, RK4, RK8, model_henon, plot_all, plot, plot_xy, 
        plot_xz, plot_yz, sensitive_dependence, disturbance, orbit_sep, method, sub_method, d0, plot_running, GS, average_lyapunov, 
        lyapunov_method, num_iterations, Poincare, discard, modelling_error, error_comparison, log_scale):

    print("Running")

    if runtime:
        start_time = time.time()

    # get system name as a string for plot titles

    system_name = get_system_name(system)

    # modelling methods
    if EM:
        x, y, z = model(eulers_method, init, system, None, dt, params, num_steps_to_stop)
        method = eulers_method
        sub_method = None

    if improved_EM:
        x, y, z = model(improved_eulers_method, init, system, None, dt, params, num_steps_to_stop)
        method = improved_eulers_method
        sub_method = None

    if RK4:
        x, y, z = model(runge_kutta, init, system, runge_kutta_4, dt, params, num_steps_to_stop)
        method = runge_kutta
        sub_method = runge_kutta_4

    if RK8:
        x, y, z = model(runge_kutta, init, system, runge_kutta_8, dt, params, num_steps_to_stop)
        method = runge_kutta
        sub_method = runge_kutta_8

    if model_henon:
        henon(init, params, num_steps_to_stop)

    # visualization of sensitive dependence to initial conditions
    if sensitive_dependence:
        senstive_dep(init, x, y, z, params, dt, num_steps_to_stop, system, method, sub_method, disturbance, system_name)

    # error analysis
    if modelling_error:
        sol_h = np.array([x, y, z])
        richardson_extrapolation(sol_h, init, 1, model, method, system, sub_method, dt, params, num_steps_to_stop, log_scale, system_name)

    if error_comparison:
        RE_error_comp(init, dt, params, num_steps_to_stop, system, log_scale, system_name)

    # plot the whole system - Note: plots will save to the working directory (the folder that the code is in)
    if plot or plot_all:
        plot_system(x, y, z, system_name, False, None, None)

    # projections
    if plot_xy or plot_all:
        plot_xy_proj(x, y, system_name)

    if plot_xz or plot_all:
        plot_xz_proj(x, z, system_name)

    if plot_yz or plot_all:
        plot_yz_proj(y, z, system_name)

    if orbit_sep:
        print(orbsep(method, sub_method, system, init, d0, plot_running, dt, num_steps_to_stop, params, discard, system_name))

    if GS:
        print(GS_ortho(x, y, z, params, dt, num_steps_to_stop, discard))

    if average_lyapunov:
        print(avg_lyapunov(lyapunov_method, params, system, method, sub_method, dt, num_steps_to_stop, d0, discard, num_iterations))

    if Poincare:
        poincare(x, y, z, system_name)

    if runtime:
        end_time = time.time()
        print(f"Simulation runtime = {end_time - start_time}") # calculate runtime

    print("Done")

    if plot or plot_xy or plot_xz or plot_yz or plot_all or orbit_sep or modelling_error or error_comparison or model_henon or sensitive_dependence or Poincare:
        plt.show()

    return