# Torch7-profiling

Repository of neural network models for feed-forward speed and number of operation profiling.


## Running the application

The application can profile both a written definition of a network or an already trained network (saved as ether `ascii` or `binary`). Pass in the location of the model using the `-m` or its long equivalent `--model`.

To profile the written definition of the model it must be defined in a specially formatted table saved to a file with the `.lua` extension. Examples of which can be found in the 'models' directory.

```
th profile-model.lua --model <'model name'>
```

To profile the already trained network, pass in the file name by again using the `-m` flag. If the network file has standard extensions the application will auto detect if the network is saved as ether an `ascii` or `binary` network and load appropriately. In most cases you will also need to pass in the network `eye` value and in some rare case the input channel number.

```
th profile-model.lua --net <'model name'> --eye <eye number value>
```

Profiling the network speed on different platforms is also possible. Currently the default platform is `cpu` but if available the profiler can be targeted to run the networks using `cuda` or the `nnx` hardware.

```
th profile-model.lua --model <'model name'> --platform <'cpu'|'cuda'|'nnx'>
```


## Running notes on specific cores

Note that the ODroid has 8 cores, 4 slow and 4 fast, for best performance use only the fast cores, this is done using `taskset`.

```
taskset -c 4-7 th profile-model.lua ...
```
