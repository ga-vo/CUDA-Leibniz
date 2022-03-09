# CUDA implementation of Leibniz series for PI calculation

## Compile
Use  `nvcc -arch=<GPU_ARCHITECTURE_CODE> -ccbin g++ -o leibniz leibniz_GPU.cu`

You can check the *GPU_ARCHITECTURE_CODE* corresponding to your GPU architecture <a href="https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/" target="_blank">here</a>
<br>

## RUN 
Use `./leibniz` 
<br>
