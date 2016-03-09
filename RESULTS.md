#Torch7 Profiling 
# RESULTS TABLE

# AlexNet OWT 
operations: 1.43 G

image size: 224 x 224

All results are averaged over 100 runs unless otherwise mentioned

### Macbook Pro 15in Late 2013 CPU intel i7
31.90 ms

### Macbook Pro 15in Late 2013 GPU GT 750M 
25.18 ms

### Intel(R) Xeon(R) CPU E5-1620 0 @ 3.60GHz (GPU2)
462.37 ms (1-core)

### nVidia GeForce GTX 980 (GPU2)
2.99 ms

### nVidia TX1 CPU
114.66 ms

### nVidia TX1 GPU 32 bits
25.73 ms

### nVidia TX1 CUDNN 4, FP32 thnets:

|      Batch Size     |   1   |   2   |   4   |   8   |   16  |   32*  |
|:-------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Time (ms per batch) | 0.054 | 0.057 | 0.069 | 0.093 | 0.137 | 0.216 |
| Time (ms per frame) | 0.054 | 0.028 | 0.017 | 0.012 | 0.008 | 0.007 |

*batch > 32 gets worse

### nVidia TX1 CUDNN 4, FP16 thnets:


|      Batch Size     |   1   |   2   |   4  |   8   |   16  |   32  |
|:-------------------:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|
| Time (ms per batch) | 0.028 | 0.033 | 0.04 |  0.07 | 0.135 | 0.593 |
| Time (ms per frame) | 0.028 | 0.016 | 0.01 | 0.009 | 0.008 | 0.018 |


### nVidia TX1 CPU thnets:

batch 1 0.316170 ms

(batch > 1 is not better in performance)


### nVidia TX1 nVidia TX1 thnets cudnn 4

| Input Resolution | Perf. (ms) CPU FP32 | Perf. (ms) GPU FP32 | Perf. (ms) GPU FP16 |
|:----------------:|:-------------------:|:-------------------:|:-------------------:|
|   VGA (640x480)  |         1.27        |        0.094        |        0.058        |
|  WXGA (1280x720) |         4.4         |         0.3         |        0.203        |
|  FHD (1920x1080) |        11.23        |         0.67        |        0.434        |
