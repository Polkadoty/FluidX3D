#include "setup.hpp"
#include "opencl.hpp" // for print_device_info
#include "info.hpp"   // for info object


// ************************************************************************** //
// Fuselage Interior Flow with Bottom Bay Opening Setup                     //
// ************************************************************************** //
void main_setup() {

    // --- Simulation Parameters ---
    const uint Nx = 480, Ny = 135, Nz = 96;
    const float nu = 0.0024f;
    // External flow velocity imposed at the bay opening
    const float3 u_external = float3(0.05f, 0.0f, 0.0f);

    // --- Geometry Definitions ---
    // Defines the bay opening area on the y=0 plane
    const uint door_x_min = 100, door_x_max = 160; // Opening start/end in x
    const uint door_z_min = 24,  door_z_max = 72;  // Opening start/end in z
    // Defines the height of the vertical side walls (doors) above the opening
    const uint fuselage_floor_y = 30; // Y-level corresponding to the main fuselage floor

    // --- Output Control ---
    const uint total_time_steps = 50000u;
    const uint vtk_output_freq = 500u;
    const uint console_output_freq = 100u;

    // --- Initialize LBM Simulation ---
    LBM lbm(Nx, Ny, Nz, nu);
    print_device_info(lbm.lbm_domain[0]->get_device().info);

    // --- Define Geometry and Boundary Conditions ---
    parallel_for(lbm.get_N(), [&](ulong n) {
        uint x = 0u, y = 0u, z = 0u;
        lbm.coordinates(n, x, y, z);

        // 1. Default: All nodes are fluid with zero initial velocity
        lbm.flags[n] = TYPE_F;
        lbm.rho[n] = 1.0f;
        lbm.u.x[n] = 0.0f;
        lbm.u.y[n] = 0.0f;
        lbm.u.z[n] = 0.0f;

        // 2. Define the Outer Solid Enclosure (except for bay opening)
        bool is_outer_wall = (x == 0 || x == Nx - 1 || y == Ny - 1 || z == 0 || z == Nz - 1 || y == 0);
        if (is_outer_wall) {
            lbm.flags[n] = TYPE_S;
            // Velocity already zero
        }

        // 3. Define Bay Opening on Bottom Face (y=0)
        bool is_in_opening_xz = (x >= door_x_min && x < door_x_max &&
                                 z >= door_z_min && z < door_z_max);

        if (y == 0 && is_in_opening_xz) { // If on bottom face AND within opening footprint
            lbm.flags[n] = TYPE_E; // Override the solid bottom wall flag
            lbm.rho[n] = 1.0f;
            lbm.u.x[n] = u_external.x; // Impose external flow velocity
            lbm.u.y[n] = u_external.y;
            lbm.u.z[n] = u_external.z;
        }

        // 4. Define Vertical Solid Side Walls (Doors) above the opening
        bool is_door_side_wall   = (z == door_z_min || z == door_z_max - 1);
        bool is_in_door_x_range = (x >= door_x_min && x < door_x_max);
        bool below_fuselage_floor = (y > 0 && y < fuselage_floor_y); // y>0 because y=0 is opening BC

        if (is_in_door_x_range && below_fuselage_floor && is_door_side_wall) {
            lbm.flags[n] = TYPE_S; // Override fluid flag
            // Velocity already zero
        }
        // All other interior nodes remain TYPE_F with zero initial velocity

    }); // End parallel_for

    // --- Run Simulation ---
    printf("Starting fuselage interior simulation with bottom bay opening...\n");
    printf("Grid: %u x %u x %u = %.2f M Cells\n", Nx, Ny, Nz, (float)(Nx*Ny*Nz)/1e6f);
    printf("Viscosity (nu_LB): %f\n", nu);
    printf("External Flow Velocity at Opening (Vx): %f\n", u_external.x);
    printf("Target Time Steps: %u\n", total_time_steps);
    printf("VTK Output Freq: %u steps\n", vtk_output_freq);
    printf("--------------------------------------------------\n");

    lbm.run(0u); // Initialize fields based on flags/rho/u

    ulong t_start = lbm.get_t();
    while (lbm.get_t() < t_start + total_time_steps) {
        if (lbm.get_t() % vtk_output_freq == 0) {
            printf("Writing VTK files at step %lu...\n", lbm.get_t());
            lbm.u.write_device_to_vtk(get_exe_path() + "export/");
            lbm.rho.write_device_to_vtk(get_exe_path() + "export/");
        }
        if (lbm.get_t() % console_output_freq == 0 && lbm.get_t() > t_start) {
            printf("Time Step: %9lu / %u (%5.1f%%) | Sim Time: %10.6f | Perf: %8.2f MCells/s\n",
                lbm.get_t(), total_time_steps, (float)(lbm.get_t() - t_start) * 100.0f / (float)total_time_steps,
                info.runtime_lbm, (float)lbm.get_N() * 1e-6f / (float)info.runtime_lbm_timestep_smooth);
        }
        lbm.run(1u);
    }

    printf("--------------------------------------------------\n");
    printf("Simulation finished at step %lu.\n", lbm.get_t());

    printf("Writing final VTK files...\n");
    lbm.u.write_device_to_vtk(get_exe_path() + "export/");
    lbm.rho.write_device_to_vtk(get_exe_path() + "export/");

    printf("Output saved in: %s\n", (get_exe_path() + "export/").c_str());

} // End of main_setup()