import numpy as np
import math
from tick.hawkes import (SimuHawkes, SimuHawkesMulti, HawkesKernelExp,
                         HawkesKernelTimeFunc, HawkesKernelPowerLaw,
                         HawkesKernel0, HawkesSumGaussians)
from tick.base import TimeFunction
import matplotlib.pyplot as plt


def drawImpactFunctions(kernels=None):
    t_values = np.linspace(0,4,50)
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax_list_list = plt.subplots(len(kernels), len(kernels),figsize=(30,20), sharex=False,sharey=False)
    plt.subplots_adjust(left=0.03, right=0.98, top=0.95, bottom=0.03,hspace=0.45)
    for i, ax_list in enumerate(ax_list_list):
        for j, ax in enumerate(ax_list):
            y_values = kernels[j][i].get_values(t_values)
            ax.plot(t_values, y_values, linestyle='-', color="#1C1C1C", linewidth=1.5,label="real")
            ax.set_xlabel(r"$ϕ_{%g,%g}$" % (j+1, i+1),labelpad=-7,size=8)
            ax.set_ylim(-0.05,0.58)
            ax.set_xticks([0,4])
            ax.set_yticks([0,0.5])
            ax.margins(x=0) 
            ax.tick_params(labelsize=8,direction="in",length=0,pad=0.2,colors="#000000")
    plt.show()

if __name__ == '__main__':
    n_nodes =20 #Number of nodes 20/100/200/500/1000
    n_sources = 5 #Number of data sources
    n_nodes_sub=int(n_nodes/n_sources) #Number of nodes per data source
    end_time = 20 #Time until which this point process will be simulated
    n_realizations = 500 #The number of times the Hawkes simulation is performed, i.e., the number of generated sequences
    baseline=[np.random.uniform(0.01,1)/n_nodes for _ in range(n_nodes)] #The baseline of all intensities, also noted μ(t)
    block_rate=0.5 #The proportion of causal semantic pathways that are blocked
    schemaMatrix_source = np.random.choice([1, 0], size=(n_sources,n_sources), p=[1-block_rate,block_rate])  #Source-level connectivity matrix
    schemaMatrix_node = np.kron(schemaMatrix_source, np.ones((n_nodes_sub,n_nodes_sub))) #Node-level connectivity matrix
    kernelMatrix = np.array([[None]*n_nodes for _ in range(n_nodes)]) #A 2-dimensional arrays of kernels, also noted ϕij
    #Eleven candidate impact functions can be selected.
    #SineLikeKernel_0
    t_values = np.linspace(0, 4, 50)
    y_values = np.maximum(0., np.sin(t_values) / 4)
    SineLikeKernel_0 = HawkesKernelTimeFunc(t_values=t_values,y_values=y_values)
    #SineLikeKernel_1
    t_values = np.linspace(0, 4, 50)
    y_values = np.maximum(0., np.sin(t_values-(4-np.pi)) / 4)
    SineLikeKernel_1 = HawkesKernelTimeFunc(t_values=t_values,y_values=y_values)
    #ExponentialKernel
    ExponentialKernel=HawkesKernelExp(.2, 2.)
    #PowerLawKernel
    PowerLawKernel=HawkesKernelPowerLaw(.2, .5, 1.3)
    #GaussianKernel_0
    t_values_gaussian = np.linspace(0, 4, 50)
    landmark_gaussian=3
    sigma_gaussian2=0.3**2
    y_values_gaussian=0.4*1 / np.sqrt(2 * np.pi * sigma_gaussian2) * \
                          np.exp(-0.5 * ((t_values_gaussian - landmark_gaussian) ** 2) / sigma_gaussian2)
    GaussianKernel_0=HawkesKernelTimeFunc(t_values=t_values_gaussian,y_values=y_values_gaussian)
    #GaussianKernel_1
    t_values_gaussian = np.linspace(0, 4, 50)
    landmark_gaussian=1
    sigma_gaussian2=0.3**2
    y_values_gaussian=0.4*1 / np.sqrt(2 * np.pi * sigma_gaussian2) * \
                          np.exp(-0.5 * ((t_values_gaussian - landmark_gaussian) ** 2) / sigma_gaussian2)
    GaussianKernel_1=HawkesKernelTimeFunc(t_values=t_values_gaussian,y_values=y_values_gaussian)
    #PiecewiseConstantKernel_0
    t_values = np.array([0, 0.5, 1.5], dtype=float)
    y_values = np.array([0, 0.4, 0], dtype=float)
    tf = TimeFunction([t_values, y_values],inter_mode=TimeFunction.InterConstRight, dt=0.1)
    PiecewiseConstantKernel_0 = HawkesKernelTimeFunc(tf)
    #PiecewiseConstantKernel_1
    t_values = np.array([0, 2, 4], dtype=float)
    y_values = np.array([0, 0.2, 0], dtype=float)
    tf = TimeFunction([t_values, y_values],inter_mode=TimeFunction.InterConstRight, dt=0.1)
    PiecewiseConstantKernel_1 = HawkesKernelTimeFunc(tf)
    # Custom Kernel
    CustomKernal_0 = HawkesKernelTimeFunc(
        t_values=np.array([0., .7, 2.5, 3., 4.]),
        y_values=np.array([.3, .03, .03, .2, 0.]))
    #Custom Kernel
    CustomKernal_1 = HawkesKernelTimeFunc(
        t_values=np.array([0., 1., 2., 3., 4.]),
        y_values=np.array([0., .2, 0, .3, 0.]))
    #ZeroKernel
    ZeroKernel=HawkesKernel0()
    KernelList=[SineLikeKernel_0,SineLikeKernel_1,ExponentialKernel,PowerLawKernel,GaussianKernel_0,GaussianKernel_1,
                PiecewiseConstantKernel_0,PiecewiseConstantKernel_1,CustomKernal_0,CustomKernal_1,ZeroKernel]

    #The impact functions of the blocked paths are set to ZeroKernel
    indexCollection_block = np.where(schemaMatrix_node == 0)
    for row, col in zip(indexCollection_block[0], indexCollection_block[1]):
        kernelMatrix[row][col]=ZeroKernel

    #The impact functions of the opened path are selected from the list of candidate impact functions.
    indexCollection_open = np.where(schemaMatrix_node > 0)
    p=[1/(len(KernelList)) for _ in KernelList]
    for row, col in zip(indexCollection_open[0], indexCollection_open[1]):
        kernelMatrix[row][col]=np.random.choice(KernelList,1,p=p,replace=False)[0]

    #Start generating data.
    hawkes = SimuHawkes(baseline=baseline, kernels=kernelMatrix, end_time=end_time,max_jumps=200,
                        verbose=False, seed=1039,force_simulation=False)
    multi = SimuHawkesMulti(hawkes, n_simulations=n_realizations,n_threads=0)
    multi.simulate()
    
    #Draw impact functions.
    drawImpactFunctions(kernels=kernelMatrix)