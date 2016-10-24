#Torch7 Profiling 
# RESULTS TABLE

# ResNet 18

720p (720x1280) 

67.7 Gops/frame

###TitanX

35.40 ms

###GTX 1080

25.48 ms

# AlexNet OWT 
operations: 1.43 G

image size: 224 x 224

All results are averaged over 100 runs unless otherwise mentioned

### Macbook Pro 15in Late 2013 CPU intel i7
31.90 ms

### Intel Core i7 4710HQ (Gigabyte P35x V4)
25.30 ms (4C8T)

### Macbook Pro 15in Late 2013 GPU GT 750M 
25.18 ms

### Intel(R) Xeon(R) CPU E5-1620 0 @ 3.60GHz (GPU2)
462.37 ms (1-core)

### nVidia GeForce GTX 980M (Gigabyte P35X v4)
3.74 ms

### nVidia GeForce GTX 980 (GPU2)
2.99 ms

### nVidia GeForce GTX Titan X (GPU3)
2.57 ms

### nVidia GeForce GTX 1080 (GPU1)
2.00 ms

### nVidia TX1 CPU
114.66 ms

### nVidia TX1 GPU 32 bits
25.73 ms

### nVidia TX1 CUDNN 4, FP32 thnets:

|      Batch Size     |   1   |   2   |   4   |   8   |   16  |   32*  |
|:-------------------:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Time (ms per batch) | 54 | 57 | 69 | 93 | 137 | 216 |
| Time (ms per frame) | 54 | 28 | 17 | 12 | 8 | 7 |

*batch > 32 gets worse

### nVidia TX1 CUDNN 4, FP16 thnets:


|      Batch Size     |   1   |   2   |   4  |   8   |   16  |   32  |
|:-------------------:|:-----:|:-----:|:----:|:-----:|:-----:|:-----:|
| Time (ms per batch) | 28 | 33 | 40 |  70 | 135 | 593 |
| Time (ms per frame) | 28 | 16 | 10 | 9 | 8 | 18 |


### nVidia TX1 CPU thnets:

batch 1 31.6170 ms

(batch > 1 is not better in performance)

### nVidia TX1 nVidia TX1 thnets cudnn 4

| Input Resolution | Perf. CPU FP32* (ms) | Perf. GPU FP32 (ms) | Perf. GPU FP16 (ms) |
|:----------------:|:--------------------:|:-------------------:|:-------------------:|
|   VGA (640x480)  |         1272        |        95        |        58         |
|  WXGA (1280x720) |         4406        |         308      |        203        |
|  FHD (1920x1080) |        11237        |         673      |        434        |

*CPU results averaged over 10 runs
