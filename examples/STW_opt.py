import csdl_alpha as csdl
import numpy as np

class STW_fuelburn():

    def __init__(self, q_inf, V_cruise=252.474):
        # nonlinear iteration parameters
        self.tol = 1e-10  # nonlinear solver convergence tolerance
        self.max_iter = 1000  # max number of nonlinear iterations
        
        # constants
        self.q_inf = q_inf
        self.tsfc = 18e-6  # thrust-specific fuel consumption [kg/(N*s)]
        self.gamma = np.deg2rad(2.054)  # climb angle [rad]
        self.R_cruise = 3815e3  # cruise range [m]
        self.V_cruise = V_cruise  # cruise speed [m/s]; pass M * a of the cruise condition
        self.R_climb = 290e3  # climb range [m]
        self.V_climb = 156.464  # climb speed [m/s]
        self.N_z = 3.75  # ultimate load factor

        self.M_payload = 14.5e3  # Payload mass [kg]
        self.M_frame = 25e3  # Operating empty mass minus wings [kg]
        self.M_fuel_res =  2e3  # Fuel reserve mass [kg]

        self.C_D_frame = 0.01508  # frame drag coefficient [-]
        self.S_csw_fraction = 0.3  # control surface area as a fraction of wing surface area [-]

        self.g = 9.81  # gravitational acceleration [m/s^2]
        self.kg_to_lbs = 2.20462  # conversion from kg's to lbs
        self.m_to_ft = (1./0.3048)  # conversion from meters to feet

    def compute_cruise_drag(self, S_wings, D_wings):
        # We add twice the drag of the single wing, plus the frame drag
        D_total = D_wings + self.q_inf * S_wings * self.C_D_frame
        return D_total

    def compute_wing_mass(self, design_mass, S_wings, AR, tc_root, taper, sweep):
        # NOTE: The inputs to this equation are first scaled to imperial units where necessary.
        #       Moreover, we assume here that the control surface area is a constant 
        #       fraction of the wing planform area
        design_weight = design_mass * self.kg_to_lbs  # design weight in lbf
        S_wings_ft = S_wings * (self.m_to_ft**2)

        wing_weight = 0.0051 * ((design_weight * self.N_z) ** 0.557) * (S_wings_ft ** 0.649) * (AR ** 0.5) * (tc_root ** -0.4) * ((1 + taper) ** 0.1) * (csdl.cos(sweep) ** -1) * ((self.S_csw_fraction * S_wings_ft) ** 0.1)
        wing_mass = wing_weight / self.kg_to_lbs  # convert wing_weight in lbf to mass in kg
        return wing_mass

    def compute_landing_mass(self, design_mass, S_wings, AR, tc_root, taper, sweep):
        wing_mass = self.compute_wing_mass(design_mass, S_wings, AR, tc_root, taper, sweep)
        landing_mass = wing_mass + self.M_payload + self.M_frame + self.M_fuel_res
        return landing_mass

    def compute_cruise_start_mass(self, LGM, L_wings, S_wings, D_wings):
        D_cruise = self.compute_cruise_drag(S_wings, D_wings)
        cruise_start_mass = LGM * csdl.exp( (self.R_cruise * self.tsfc * self.g / self.V_cruise) * (D_cruise / L_wings))
        
        return cruise_start_mass

    def compute_take_off_mass(self, LGM, L_wings, S_wings, D_wings):
        D_cruise = self.compute_cruise_drag(S_wings, D_wings)
        
        cruise_start_mass = self.compute_cruise_start_mass(LGM, L_wings, S_wings, D_wings)
        take_off_mass = cruise_start_mass * csdl.exp( (self.R_climb * self.tsfc * self.g / self.V_climb) * ( (np.cos(self.gamma) / (L_wings / D_cruise)) + np.sin(self.gamma) ) )

        return take_off_mass

    def evaluate(self, S_wings, AR, tc_root, taper, sweep, L_wings, D_wings):     
        # NOTE: Wing mass depends on take-off mass, so we have to use a nonlinear solver here to solve this cyclical dependency
        # first define the implicit variables
        landing_gross_mass = csdl.ImplicitVariable(value=5e4)
        take_off_gross_mass = csdl.ImplicitVariable(value=6e4)

        # then define the masses as computed by the methods here
        landing_gross_mass_computed = self.compute_landing_mass(take_off_gross_mass, S_wings, AR, tc_root, taper, sweep)
        take_off_gross_mass_computed = self.compute_take_off_mass(landing_gross_mass, L_wings, S_wings, D_wings)

        # followed by the nonlinear residuals (the differences between the variable values and the computed values)
        landing_mass_residual = landing_gross_mass_computed - landing_gross_mass
        take_off_mass_residual = take_off_gross_mass_computed - take_off_gross_mass

        # define and set up the nonlinear solution loop
        solver = csdl.nonlinear_solvers.Newton(tolerance=self.tol, max_iter=self.max_iter)

        solver.add_state(landing_gross_mass, landing_mass_residual)
        solver.add_state(take_off_gross_mass, take_off_mass_residual)
        solver.run()

        cruise_start_mass = self.compute_cruise_start_mass(landing_gross_mass, L_wings, S_wings, D_wings)

        fuelburn = take_off_gross_mass - landing_gross_mass
        return take_off_gross_mass, landing_gross_mass, cruise_start_mass, fuelburn

if __name__ == '__main__':
    # We test the STW_fuelburn class above with some assumed values
    L = 5e4 * 9.81
    D_wings = 2e3 * 9.81
    # cruise at 10,400 m, M 0.77
    rho, p, M = 0.3926, 24857., 0.77
    V_cruise = M * np.sqrt(1.4 * p / rho)  # ~229.2 m/s
    q_inf = 0.5 * rho * V_cruise ** 2  # dynamic pressure at cruise

    # we also use the standard STW parameters
    S = 45.5 * 2  # surface area of both wings
    AR = 8.615
    tc_root = 0.121  # t/c of the RAE2822 airfoil
    taper = 1.5 / 5  # tip / root chord
    sweep = np.deg2rad(25.32)  # sweep angle at quarter-chord [rad]


    recorder = csdl.Recorder(inline = True)
    recorder.start()
    
    STW = STW_fuelburn(q_inf, V_cruise=V_cruise)

    takeoff_mass, landing_mass, cruise_start_mass, fuel_burn_mass = STW.evaluate(S, AR, tc_root, taper, sweep, L, D_wings)

    print("Take-off mass: {} [kg]".format(takeoff_mass.value))
    print("Cruise start mass: {} [kg]".format(cruise_start_mass.value))
    print("Landing mass: {} [kg]".format(landing_mass.value))
    print("Fuel burn mass: {} [kg]".format(fuel_burn_mass.value))

    recorder.stop()