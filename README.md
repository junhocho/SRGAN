# Torch7-profiling

Repository of neural network models for feed-forward speed and number of
operation profiling.


## Running the application

The application can profile both a written definition of a network or an
already trained network (saved as ether 'ascii' or 'binary').

To profile the written definition of the model it must be defined in a specially
formatted table, examples of which can be found in the 'models' directory.
Select the model definition using the '-m' or its long equivalent '--model'.

```
th profile-model.lua --model <'model name'>
```

To profile the already trained network, pass in the file name of the network
using the '-n' or its long equivalent '--net'. In most cases you will also need
to pass in the network 'eye' value. If the network has standard extensions the
application will auto detect if the network is saved as ether a 'ascii' or
'binary' network and load appropriately.

```
th profile-model.lua --net <'model name'> --eye <eye number value>
```

Profiling the network speed on different platforms is also possible. Currently
the default platform is 'cpu' but if available the profiler can be targeted
running the networks using 'cuda' or the 'nnx' hardware.


```
th profile-model.lua --model <'model name'> --platform <'cpu'|'cuda'|'nnx'>
```


## Running notes on specific cores

Note that the ODroid has 8 cores, 4 slow and 4 fast, for best performance use
only the fast cores, this is done using 'taskset'.


```
taskset -c 4-7 th profile-model.lua ...
```
