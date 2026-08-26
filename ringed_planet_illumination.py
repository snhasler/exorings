'''
author: S. Hasler
Script to simulate and visualize the illumination of ringed planet across orbital epochs.
'''
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import matplotlib.tri as mtri
import datetime

import ringed_planet as rings
import ray_tracing as ray_trace

# Parameters
# ---------------------
plot_closeup = False                                  # whether to plot close-up views of planet and ring at each epoch
plot = True                                         # whether to plot or not        
save_files = False                                   # whether to save output files or not (set to True to get output for illu)
save_figs = False
scaling_factor = 1/1#0**5                            # scaling factor for plotting
planet_radius = 69911 * scaling_factor               # km -- planet radius (Jupiter-sized)
ring_inner = 1.43 * planet_radius #100e3 * scaling_factor                  # km -- inner ring radius
ring_outer = 2.47 * planet_radius #830e3 * scaling_factor                  # km -- outer ring radius

print("Ring inner radius (km): ", ring_inner, "\t Ring outer radius (km): ", ring_outer)
# Orbital parameters
a = (0.05*u.AU).to(u.km).value * scaling_factor      # km -- semi-major axis
a_PLANET = 2.513                                      # AU -- true planet separation to be used later for rescaling
e = 0.0                                              # eccentricity
inc = np.radians(90.0)                                  # inclination
Omega = np.radians(0)                                # longitude of ascending node
omega = np.radians(170.0)                              # argument of periastron

# Ring orientation relative to planet's spin axis
# If inc = 90, obliquity = 90 is equivalent to obl = 0 (rings edge-on w.r.t observer)
obliquity = np.radians(30)                           # Ring obliquity (tilt from spin axis)
spin_longitude = np.radians(0) #-90                    # planet spin longitude
# obliquity = obliquity - np.radians(90) # adjust obliquity so that 0 means rings are edge-on to observer at 90 deg inclination

# Number of orbital positions to sample
n_epochs = 36

now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") 

# Files to save output
f_planet_illum = f"/Users/shasler/Documents/Projects/Roman/fraction_lit_output/for_paper/1xSat_incl30_obl90_rerun_Aug2026/planet_illumination_e{e}_inc{np.degrees(inc):.1f}deg_obliq{np.degrees(obliquity):.1f}deg_{ring_outer - ring_inner:.2f}Rring_{n_epochs}epochs_{now}.txt"
f_ring_illum = f"/Users/shasler/Documents/Projects/Roman/fraction_lit_output/for_paper/1xSat_incl30_obl90_rerun_Aug2026/ring_illumination_e{e}_inc{np.degrees(inc):.1f}deg_obliq{np.degrees(obliquity):.1f}deg_{ring_outer - ring_inner:.2f}Rring_{n_epochs}epochs_{now}.txt"

# At top of each file, save the run parameters
if save_files:
    with open(f_planet_illum, 'w') as f_planet, open(f_ring_illum, 'w') as f_ring:
        header = (
            f"# Ringed planet illumination simulation\n"
            f"# Semi-major axis (km): {a}\n"
            f"# Eccentricity: {e}\n"
            f"# Inclination (deg): {np.degrees(inc):.2f}\n"
            f"# Longitude of ascending node (deg): {np.degrees(Omega):.2f}\n"
            f"# Argument of periastron (deg): {np.degrees(omega):.2f}\n"
            f"# Planet radius (km): {planet_radius}\n"
            f"# Ring inner radius (km): {ring_inner}\n"
            f"# Ring outer radius (km): {ring_outer}\n"
            f"# Ring obliquity (deg): {np.degrees(obliquity):.2f}\n"
            f"# Number of epochs saved (i.e., number of samples around the orbit): {n_epochs}\n\n"
        )
        f_planet.write(header)
        f_ring.write(header)

# Lists for output to file
phases = []
planet_observable_lit_forFile = []
ring_observable_lit_forFile = []

# Star and observer geometry
d = (13.48*u.pc).to(u.km).value                         # km - observer distance

star_position = np.array([0.0, 0.0, 0.0])            # star at origin
observer_dir = np.array([0.0, 0.0, -d])              # km -- observer in -z direction
observer_dir /= np.linalg.norm(observer_dir)

# Get full orbit track
# ts = 2*np.pi*np.sqrt(a**3/(4*np.pi**2))*np.arange(0,1.1,0.01)
orbit_coords = rings.full_orbit(a, e, inc, Omega, omega) # Get coordinates of fulll orbit

# Convert to intertial coordinates and get angular momentum vector
R_pCi = rings.R_pCi(inc, Omega, omega)
h = rings.ang_momentum_vector(R_pCi)

if plot:    
    # Set up plot
    fig1 = plt.figure(figsize=(7,7))
    ax = fig1.add_subplot(111, projection='3d')

    # Plot orbit path
    ax.plot(orbit_coords[:,0], orbit_coords[:,1], orbit_coords[:,2], color='k', linestyle='dashed', alpha=0.6, linewidth=0.5) 

# ---------------------
# Get planet positions at each epoch and perform illumination calculations
# ---------------------
# For ups And sims
# phase_start = np.radians(275.3082882)
# true_anomaly_arr = (phase_start + np.arange(0, 2*np.pi, 2*np.pi/n_epochs)) % (2*np.pi)

for epoch in range(n_epochs):
    # Get true anomaly
    true_anomaly = np.arange(0, 2*np.pi, 2*np.pi/n_epochs)[epoch]
    # true_anomaly = true_anomaly_arr[epoch] # for Ups And sims
    # true_anomaly = np.radians(([36.0]))[epoch]
    print('true_anomaly (deg): ', np.degrees(true_anomaly))
    theta = true_anomaly + omega

    # get vector to plot obliquity/"rotation axis" at each point in the orbit
    v1, rotation = rings.orthogonal_to_h(h)
    v2 = rotation[:, 1]
    h_hat = rotation[:, 2]

    # compute spin axis and ring tilt direction relative to orbital plane
    tilt_dir = np.cos(spin_longitude) * v1 + np.sin(spin_longitude) * v2 # unit vector in orbit plane
    spin_axis = np.cos(obliquity) * h_hat + np.sin(obliquity) * tilt_dir
    spin_axis /= np.linalg.norm(spin_axis)
    ring_normal = spin_axis

    # obl_rotation_matrix = rings.obliquity_rot_matrix(obliquity, v1)
    # obl_vector = np.dot(obl_rotation_matrix, h)

    # Get planet position at epoch
    planet_center, s = rings.kepler_orbit(a, e, inc, Omega, omega, true_anomaly, theta)
    # print(f"Projected separation (arcsec): {((s*u.km).to(u.AU)/((d*u.km).to(u.AU))*u.rad).to(u.arcsec):.4f}") 
    # Print real planet position with real separation
    planet_center_real, s_real = rings.kepler_orbit((a_PLANET*u.AU).to(u.km).value, e, inc, Omega, omega, true_anomaly, theta)
    print(f"\nEpoch {epoch}: Planet position (AU): {(s_real*u.km).to(u.AU):.4f} (~{(((s_real*u.km).to(u.AU)/(d*u.km).to(u.pc))).value*1000:.4f} mas)")

    # Vector from planet to star
    star_dir = star_position - planet_center
    star_dir /= np.linalg.norm(star_dir)

    # Calculate phase angle with planet-star and planet-observer vectors
    to_star = star_position - planet_center
    phase_angle = np.degrees(np.arccos(np.dot(to_star, observer_dir) / (np.linalg.norm(to_star) * np.linalg.norm(observer_dir))))

    # If planet is on right side of star w.r.t. observer, phase angle is positive, else negative
    if planet_center[0] < 0:
        phase_angle = phase_angle - 2*phase_angle

    phases.append(phase_angle)
    print(f'phase angle (from vectors): {phase_angle:.2f}')

    # -------------------------------
    # Planet sphere and ring sampling
    # -------------------------------
    # Generate surface points on unit sphere with corresponding normal vectors
    planet_unit_pts, planet_normals = rings.sample_sphere(n_theta=260, n_phi=260) 
    planet_pts = planet_center + planet_radius * planet_unit_pts # scale to actual planet position/size

    # Get reference for rotating ring points
    ref = v1
    if abs(np.dot(v1, ring_normal)) > 0.99:
        # If ring normal is nearly parallel to v1, use v2 as reference
        ref = v2
    e1 = np.cross(ring_normal, ref)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(ring_normal, e1)
    B = np.column_stack((e1, e2, ring_normal)) # rotation from local to inertial

    # Generate points in ring plane 
    ring_pts0 = rings.sample_ring(ring_inner, ring_outer, n_r=260, n_phi=360) # n_r=180, n_phi=260
    x_ring, y_ring, z_ring = ring_pts0.T
    ring_pts = np.vstack((x_ring, y_ring, z_ring))

    # Put ring points in planet's coordinate system and scale by planet radius
    ring_pts = planet_center + ring_pts0 @ B.T #* planet_radius  

    # Find normal to ring plane using 3 random ring points
    ring_normal = rings.get_ring_normal(ring_pts)

    # -------------------
    # Illumination tests
    # -------------------
    planet_lit = []
    ring_lit = []
    planet_observable = []
    ring_observable = []

    # Check illumination for each planet surface point
    for p, n in zip(planet_pts, planet_normals):
        ray_origin = star_position
        ray_dir = p - star_position
        dist = np.linalg.norm(ray_dir)

        # Is this point on the planet visible to the observer?
        # i.e., is it on the visible hemisphere w.r.t observer direction
        visible = np.dot(n, observer_dir) > 0
        planet_observable.append(visible)

        if not visible:
            planet_lit.append(False)
            continue

        # Check if ray intersects planet before reaching surface point (self-shadowing)
        # This catches points on the "night" side of the planet
        lit = True # Assume point is illuminated until we find an obstruction

        t_planet = ray_trace.intersect_sphere_distance(ray_origin, ray_dir, 
                                                       planet_center, planet_radius)
        if t_planet is not None and t_planet < dist - 1e-6:
            lit = False  # Another part of planet blocks this point

        # Check if ring casts shadow on this planet surface point
        t_ring = ray_trace.intersect_ring_distance(
            ray_origin, ray_dir,
            planet_center, ring_normal, 
            ring_inner, ring_outer
        )
        if t_ring is not None and t_ring < dist - 1e-6:
            lit = False  # Ring blocks light 

        planet_lit.append(lit)

    # Check illumination for each ring point
    for p in ring_pts:#.T:
        ray_origin = star_position
        ray_dir = p - star_position
        dist = np.linalg.norm(ray_dir)

        lit = True

        # Check if planet casts shadow on this ring point
        t_planet = ray_trace.intersect_sphere_distance(
            ray_origin, ray_dir, planet_center, planet_radius
        )
        if t_planet is not None and t_planet < dist - 1e-6:
            lit = False 

        ring_lit.append(lit)

        # Does this ring point face the observer?
        to_star = star_position - p
        to_star /= np.linalg.norm(to_star)

        if np.sign(np.dot(ring_normal, to_star)) != np.sign(np.dot(ring_normal, observer_dir)):
            lit = False

        # Block ring points that are hidden behind the planet w.r.t observer
        t_planet = ray_trace.intersect_sphere_distance(p, observer_dir,
                                                       planet_center, planet_radius)

        if t_planet is not None:
            lit = False

        ring_observable.append(lit)

    # Convert to arrays 
    planet_lit = np.array(planet_lit)
    ring_lit = np.array(ring_lit)
    ring_observable = np.array(ring_observable) 
    planet_observable = np.array(planet_observable) 

    # Calculate fraction of illumination for both planet & ring
    planet_frac = planet_lit.mean()
    ring_frac = ring_lit.mean()
    ring_observable_frac = ring_observable.mean() 
    planet_observable_frac = planet_lit[planet_observable].mean()

    print(f"Illuminated planet fraction: {planet_frac:.3f}")
    print(f"Illuminated ring fraction (from star --> ring):   {ring_frac:.3f}")
    print(f"Observable ring fraction:    {ring_observable_frac:.3f}") 
    print(f"illuminated fraction of observable planet: ", planet_lit[planet_observable].mean())

    # Save for output files
    planet_observable_lit_forFile.append(planet_observable_frac) # save observable fraction to list
    ring_observable_lit_forFile.append(ring_observable_frac)

    # -----------------
    # Plot
    # -----------------
    if plot: # Plot all epochs together with illuminated fractions
        # planet points
        ax.scatter(*planet_pts[planet_lit].T, s=1, alpha=0.6, c='gold')#, label=f'phase={phase_angle:.1f}°') # illuminated planet points
        ax.scatter(*planet_pts[~planet_lit].T, s=1, alpha=0.5, c='black') # shadowed planet points
        
        # Plot true anomalies as text labels
        if n_epochs < 20: # only label if not too many points
            ax.text(*planet_center+1e6, f"{np.degrees(true_anomaly):.1f}°", color='k', fontsize=8, ha='center', va='center')

        # Plot ring
        ax.scatter(*ring_pts[ring_observable].T, s=1, alpha=0.6, c='goldenrod') # illuminated
        ax.scatter(*ring_pts[~ring_observable].T, s=1, alpha=0.5, c='darkgray') # shadowed

        # Plot star 
        ax.scatter(*star_position, color='gold', marker='*', s=100, edgecolors='k', linewidths=0.5)
        # arrow showing observer direction 
        ax.quiver(*planet_center, *observer_dir, length=3*ring_inner, color='k', linewidth=1, 
                label='Observer dir' if epoch == 0 else "")
        # arrow showing star direction 
        ax.quiver(*planet_center, *star_dir, length=3*ring_inner, color='gold', linewidth=1,
                linestyle='dashed', label='Star dir' if epoch == 0 else "")

    if plot_closeup:
        fig2, ax2 = plt.subplots(figsize=(7, 7))

        # Depth split: compare each ring point's z to the planet centre's z.
        # To prevent ring/planet overlapping each other on plot
        planet_z  = planet_center[2]
        ring_z    = ring_pts[:, 2]

        back_mask  = ring_z >  planet_z   # further from observer → draw first
        front_mask = ring_z <= planet_z   # closer to observer   → draw last

        # --- Layer 1: back ring (behind planet disk) ----------------------
        ax2.scatter(ring_pts[back_mask & ~ring_observable, 0], # changed ring_lit to ring_observable to show only the visible ring points
                    ring_pts[back_mask & ~ring_observable, 1],
                    s=1, c='dimgray',  alpha=0.8, zorder=1)
        ax2.scatter(ring_pts[back_mask &  ring_observable, 0],
                    ring_pts[back_mask &  ring_observable, 1],
                    s=1, c='goldenrod', alpha=0.7, zorder=1)

        # --- Layer 2: planet disk -----------------------------------------
        ax2.scatter(planet_pts[~planet_lit, 0], planet_pts[~planet_lit, 1],
                    s=1, c='black', alpha=0.2, zorder=2)
        ax2.scatter(planet_pts[ planet_lit, 0], planet_pts[ planet_lit, 1],
                    s=1, c='gold',  alpha=0.9, zorder=2,
                    label=f'phase={phase_angle:.1f}°')

        # --- Layer 3: front ring (on top of planet disk) ------------------
        ax2.scatter(ring_pts[front_mask &  ~ring_observable, 0],
                    ring_pts[front_mask &  ~ring_observable, 1],
                    s=1, c='dimgray', alpha=0.8, zorder=3)
        ax2.scatter(ring_pts[ring_lit & ring_observable, 0],
                    ring_pts[ring_lit & ring_observable, 1],
                    s=1, c='goldenrod', alpha=0.7, zorder=3) # visible ring


        # Direction vectors projected onto the xy plane.
        # observer_dir = [0, 0, -1] has zero xy-component, so omit that arrow.
        vec_len = 3 * ring_inner
        # ax2.quiver(planet_center[0], planet_center[1],
        #            star_dir[0] * vec_len, star_dir[1] * vec_len,
        #            color='gold', linewidth=1, linestyle='dashed', zorder=4,
        #            angles='xy', scale_units='xy', scale=1)
        # ax2.quiver(planet_center[0], planet_center[1],          # ring normal
        #            ring_normal[0]*3*ring_outer, ring_normal[1]*3*ring_outer,
        #            color='magenta', linewidth=2, zorder=4,
        #            angles='xy', scale_units='xy', scale=1)

        ax2.set_xlim([planet_center[0] - ring_outer, planet_center[0] + ring_outer])
        ax2.set_ylim([planet_center[1] - ring_outer, planet_center[1] + ring_outer])
        ax2.set_aspect('equal')
        ax2.axis('off')          # FIX 2: removes all ticks, labels and frame

        fig2.patch.set_alpha(0)
        ax2.patch.set_alpha(0)

        if save_figs:
            fig2.savefig(
                f"/Users/shasler/Documents/Projects/Roman/fraction_lit_output/plots/"
                f"inc{np.degrees(inc):.1f}_obl{np.degrees(obliquity):.1f}_"
                f"v{np.degrees(true_anomaly):.1f}_phase{phase_angle:.1f}.png",
                dpi=250, transparent=True, format='png', bbox_inches='tight')
        plt.tight_layout()
        fig2.show()


if plot: # Plot all epochs together with illuminated fractions
    planet_sep = np.linalg.norm(planet_center)
    max_radius = ring_outer + planet_sep + 1e6
    ax.set_xlim([-max_radius, max_radius])
    ax.set_ylim([-max_radius, max_radius])
    ax.set_zlim([-max_radius, max_radius])

    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')

    ax.set_box_aspect([1,1,1])
    ax.set_aspect(aspect='equal')

    ax.set_title(
        f"Incl. = {np.degrees(inc):.1f}°\n"
        f"Ring obliquity = {np.degrees(obliquity):.1f}°"
    )

    # plt.legend()
    ax.grid(False)
    ax.view_init(elev=-90, azim=90) # to view system from "below" (down on -XY-plane) -- "observer's perspective"
    # ax.view_init(elev=0, azim=-90)    # to view "top-down" (XZ-plane)
    if save_figs:
        fig1.savefig(f"/Users/shasler/Documents/Projects/Roman/fraction_lit_output/plots/all_epochs_inc{np.degrees(inc):.1f}deg_obliq{np.degrees(obliquity):.1f}deg_{ring_outer - ring_inner:.2f}Rring.png", 
                    dpi=400, bbox_inches='tight')
    fig1.show()


# Output phase angles and illumination fractions to files
true_anomaly = np.degrees(np.arange(0, 2*np.pi, 2*np.pi/n_epochs)).tolist()

# Zip together the phases and planet_observable_lit_forFile and sort the pairs by phase angle
phases_sorted, planet_observable_lit_forFile = zip(*sorted(zip(phases, planet_observable_lit_forFile)))
phases_sorted, ring_observable_lit_forFile = zip(*sorted(zip(phases, ring_observable_lit_forFile)))
phases_sorted, true_anomaly_sorted = zip(*sorted(zip(phases, true_anomaly)))

if save_files:
    with open((f_planet_illum), 'a') as f_planet, open(f_ring_illum, 'a') as f_ring:
        # Planet
        f_planet.write("Phase_angle(deg),true_anomaly(deg),Fraction_illuminated\n") 
        for phase, nu, frac in zip(phases_sorted, true_anomaly_sorted, planet_observable_lit_forFile):
            f_planet.write(f"{phase},{nu},{frac:.4f}\n")
        # Ring
        f_ring.write("Phase_angle(deg),true_anomaly(deg),Fraction_illuminated\n") 
        for phase, nu, frac in zip(phases_sorted, true_anomaly_sorted, ring_observable_lit_forFile):
            f_ring.write(f"{phase},{nu},{frac:.4f}\n")