import numpy as np

class Constants:
    def EPSILON0():
        """Dielectric in void [C ^ 2 / (N * m ^ 2)]"""
        return 8.85e-12
    def ATM_PRESSURE():
        """Atmosferic pressure [N / m ^ 2]"""
        return 1.013e5
    def WATER_DENSITY():
        """Water density at 25 °C [kg / m ^ 3]"""
        return 997
    def GRAVITY():
        """Gravity acceleration [m / s ^ 2]"""
        return 9.806
    def SURFACE_TENSION():
        """Water surface tension [N / m]"""
        return 72e-3
    def GLASS_REL_DIEL():
        """Relative dieletric of glass"""
        return 4
    def AIR_REL_DIEL():
        """Relative dieletric of air"""
        return 1
    def CONTACT_ANGLE():
        """Contact angle of water"""
        return np.radians(1)
    def CRITICAL_RADIUS():
        """Critical radius of water [m]"""
        return np.sqrt(Constants.SURFACE_TENSION() 
                       / (Constants.WATER_DENSITY() * 
                          Constants.GRAVITY()))
    def MAX_HEIGHT():
        """Asyntotic height of a water drop [m]"""
        return Constants.CRITICAL_RADIUS() * np.sqrt(2 * 
            (1 - np.cos(Constants.CONTACT_ANGLE())))
