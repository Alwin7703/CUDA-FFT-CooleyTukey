# CUDA-FFT-CooleyTukey
# Parallelization of Cooley-Tukey FFT Algorithm using CUDA

This project implements the **Cooley-Tukey Fast Fourier Transform (FFT)** algorithm in parallel using CUDA for high-performance computation. The FFT is widely used in signal processing, image processing, and scientific computing for efficient transformation of data between time and frequency domains.

---

## Concept

The **Cooley-Tukey FFT algorithm** is a divide-and-conquer method for computing the Discrete Fourier Transform (DFT). By breaking down a DFT of size `N` into smaller DFTs of size `N/2`, the algorithm efficiently computes the result in `O(N log N)` time complexity compared to the naive `O(N^2)` method.

In this project:
- The input sequence is represented as complex numbers (real and imaginary parts).
- CUDA is used to parallelize the FFT computations by dividing the workload across multiple threads.
- Each thread processes a "butterfly operation" (combining even and odd elements with a twiddle factor).

Key CUDA concepts include:
- **Thread-level parallelism**: Each thread computes part of the FFT.
- **Memory transfer**: Input data is copied to GPU memory for computation and results are transferred back to CPU memory.
- **Synchronization**: Ensures all threads complete before moving to the next stage.

---

## Features
- Efficient parallel FFT implementation.
- GPU acceleration using CUDA.
- Handles complex inputs.
- Easily extendable for larger datasets.

---

## Prerequisites

Ensure you have the following:
- **CUDA Toolkit** (Version 10.0 or above)
- An **NVIDIA GPU** with CUDA support.
- **Google Colab** or a local environment with CUDA installed.
- Basic knowledge of terminal and Git.

---

## Getting Started

### Steps to Run Locally

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/CUDA-FFT-CooleyTukey.git
   cd CUDA-FFT-CooleyTukey


## Setup

1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/CUDA-FFT-CooleyTukey.git
   cd CUDA-FFT-CooleyTukey

2. Develop the Code
- Create a IPYNB File: In your project directory (i.e.CUDA-FFT-CooleyTukey), create a file named CooleyTukeyFFT_Cuda_Project.ipynb .
- Write the Code: In IPYNB file, write a CUDA program to perform Parallelization of Cooley-Tukey FFT Algorithm

## Write CUDA code in a cell within Colab notebook 
%%writefile my_cuda_program.cu
// CUDA code here

## Compile the CUDA code in Google colab
! nvcc -o fft_cuda fft_cuda.cu

## Execute the compiled program in Google colab
! ./fft_cuda

