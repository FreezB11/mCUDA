# mCUDA
CUDA C++ extends C++ by allowing the programmer to define C++ functions, called kernels, that, when called are executed N times in parallel by N different CUDA threads.
<br>
A kernel is defined using the __global__ declaration specifies and the number so CUDA threads that execure that kernel for a given kernal call is specifiead using a new <<<...>>>. Each thread that executes the kernel is given a unique ID that is accessible within the kernel
[src/umm.cu](src/umm.cu)<br>

 #### Thread Hierachy
 the <mark style="background-color: rgba(255,255,255,0.5);">'threadIdx' </mark> is a 3-component vector, so that threads can be identified using a 1D, 2D OR 3D thread index forming a 1D,2D or 3D block of threads, called a thread block.<br>
 The index of a thread and its thread ID relate to each other in a straightforward way: For a 1d block ther ar the same for 2d block of size (dx,dy), the threadsID of a thread of index (x,y) is *(x+y*dx); for 3d block of size(dx,dy,dz) the threadID of a thread of index(x,y,z) is (x+y*dx+z*dx*dy).
 [src/mat.cu](src/mat.cu)<br>
 There is a limit to the number of threads per block, since all threads of a block are expected to reside
on the same streaming multiprocessor core and must share the limited memory resources of that
core. On current GPUs, a thread block may contain up to 1024 threads.<br>
However, a kernel can be executed by multiple equally-shaped thread blocks, so that the total number
of threads is equal to the number of threads per block times the number of blocks.<br>
Blocks are organized into a one-dimensional, two-dimensional, or three-dimensional grid of thread
blocks. The number of thread blocks in a grid is usually dictated by the size
of the data being processed, which typically exceeds the number of processors in the system.
The number of threads per block and the number of blocks per grid specified in the <<<...>>> syntax
can be of type int or dim3. Two-dimensional blocks or grids can be specified as in the example above.
Each block within the grid can be identified by a one-dimensional, two-dimensional, or threedimensional unique index accessible within the kernel through the built-in blockIdx variable. The
dimension of the thread block is accessible within the kernel through the built-in blockDim variable.
Extending the previous MatAdd() example to handle multiple blocks, the code becomes as follows.
from line 15 [src/mat.cu](src/mat.cu)

#### Thread Block Cluster
the CUDA programming model introduces an optional level of hierarchy called Thread Block Clusters that are made up of thread blocks. Similar to how threads in a thread block are guaranteed to be co-scheduled on a streaming multiprocessor, thread blocks in a cluster are also guaranteed to be co-scheduled on a GPU Processing Cluster (GPC) in the GPU.<br>
The number of thread blocks in a cluster can be user-defined, and a maximum of 8 thread blocks in a cluster is supported as a portable cluster size in CUDA. Note that on GPU hardware or MIG configurations which are too small to support 8 multiprocessors the maximum cluster size will be reduced accordingly. Identification of these smaller configurations, as well as of larger configurations supporting a thread block cluster size beyond 8, is architecture-specific and can be queried using the cudaOccupancyMaxPotentialClusterSize API.<br>
A thread block cluster can be enabled in a kernel either using a compile-time kernel attribute using __cluster_dims__(X,Y,Z) or using the CUDA kernel launch API cudaLaunchKernelEx. The example below shows how to launch a cluster using a compile-time kernel attribute. The cluster size using kernel attribute is fixed at compile time and then the kernel can be launched using the classical <<< , >>>. If a kernel uses compile-time cluster size, the cluster size cannot be modified when launching the kernel.   [src/TBC.cu](src/TBC.cu)<br>
A thread block cluster size can also be set at runtime and the kernel can be launched using the CUDA kernel launch API cudaLaunchKernelEx. The code example below shows how to launch a cluster kernel using the extensible API.[src/TBC.cu](src/TBC.cu)