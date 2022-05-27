from mumax_helper import * 
import numpy as np 
from adaptive import Runner, Learner1D


# NUMERICAL PARAMETERS RELEVANT FOR THE SPECTRUM ANALYSIS
T    = 15e-9        # simulation time (longer -> better frequency resolution)


# Note that this is a format string, this means that the statements inside the
# curly brackets get evaluated by python. In this way, we insert the values of
# the variables above in the script.

def simulation(freq,amp):

    dt = .1e-10

    script = f"""
    n   := 384
    length := 200e-9

    setgridsize(n,n,1)
    setcellsize(length/n,length/n,1e-9)

    domain := circle(length*.99)
    setgeom(domain)

    Msat = 1e6
    Aex = 10e-12
    Dind = 2.2e-3
    Ku1 = 1e6
    AnisU = vector(0, 0, 1)
    alpha = 0.001

    B_ext = vector(0, 0, {amp} * sin( 2*pi*{freq}*t)) 
    TableAdd(B_ext)

    dia := 20e-9
    sep := 50e-9
    shape_1 := circle(dia).transl(sep, 0, 0)
    shape_2 := circle(dia).transl(-sep, 0, 0)

    m = Uniform(0, 0, 1)
    m.setInShape(shape_1, NeelSkyrmion(1, -1).transl(sep, 0, 0 ))
    m.setInShape(shape_2, NeelSkyrmion(1, -1).transl(-sep, 0, 0 ))


    // Save the topological charge density of a skyrmion
    //saveas(chargeDensity, "chargeDensity.ovf")

    minimize()
    autosave(m,{dt})
    tableautosave({dt})
    run({T})
    """

    return script


def sim_func(freq): 
    
    title = f"Twin Skyrmions f={freq}"
    m,table = run_mumax3(simulation(freq, .002), title)

    return calculate_velocity(m, table, scale=5.28e-10)
    

learner = Learner1D(sim_func, bounds=(.5e9,4e9))
runner = Runner(learner, goal=lambda l: l.loss() < .1, ntasks=1)