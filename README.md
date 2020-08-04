# CS Roofline Toolkit #

Welcome to the CS Roofline Toolkit Bitbucket site.  The Empirical Roofline Tool and Roofline Visualizer are currently available here.
In the future additional tools will be hosted here.  

For an overview of the Roofline Performance Model and this software's place in that context see
[https://crd.lbl.gov/departments/computer-science/performance-and-algorithms-research/research/roofline/](https://crd.lbl.gov/departments/computer-science/performance-and-algorithms-research/research/roofline/).

## Empirical Roofline Tool ##

The Empirical Roofline Tool, ERT, automatically generates a roofline data
for a given computer.  This includes the maximum bandwidth for the various
levels of the memory hierarchy and the maximum gflop rate.  This data is
obtained using a variety of "micro-kernels".

The ERT comes with a set of configuration files for a number of
computers/architectures.  These configuration file can be adapted to your
local environment and needs to better measure the roofline parameters of
your computer(s).

This is version 1.1.0 of the ERT -- the second public release.

For details about the ERT, please refer to the User's Manual in the
repository under the "Empirical_Roofline_Tool-1.1.0" directory.

## Roofline Visualizer ##

The Roofline Visualizer can visualize the roofline performance data
generated locally by the ERT or stored on a remote Roofline repository.

This is version 1.1.0 of the Roofline Visualizer -- the second public release.

For details about the Roofline Visualizer, please refer to the "README.md"
file in the repository under the "Roofline_Visualizer-1.1.0" directory.

## Contact Information ##

Please contact [Charlene Yang](mailto:CJYang@lbl.gov) with any questions, problems, corrections, suggestions, etc.

## SYCL Update ##

Functionality to use ERT through SYCL has been added. The functionality is similar to OpenCL. The Config file must 
include the flag ERT_SYCL with value set to True, and the flag ERT_SYCL_SIZES with global_size:local_size pairs in a comma
separated list. The SYCL funcionality is intended to be used on a GPU and thus uses SYCL's `gpu_selector` to choose a device to execute on. A sample config file has been included. The kernel run in the ERT_SYCL mode is always the single-source SYCL kernel found in `driver1.cxx`.
