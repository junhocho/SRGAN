# Torch7-Network Profiler

Repository to calculate feed-forward time and number of operations taken by a neural network.


## Running the application

The application can profile both a written definition of a network or an already trained network (saved as ether `ascii` or `binary`). Pass in the location of the model using the `-m` or its long equivalent `--model`.

To profile the written definition of the model it must be defined in a specially formatted table saved to a file with the `.lua` extension. Examples of which can be found in the [models](models) directory.

```
th profile-model.lua --model <'path/filename.lua'> --res 1x3x231x231
```

To profile the already trained network, pass in the path and file name by again using the `-m/model` flag. If the network file has standard extensions the application will auto detect if the network is saved as ether an `ascii` or `binary` network and load appropriately.

Profiling the network speed on different platforms is also possible. Currently the default platform is `cpu` but if available the profiler can be targeted to run the networks using `cuda`.

```
th profile-model.lua --model <'path/filename.lua'> --platform <'cpu'|'cuda'>
```

### License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary here: http://creativecommons.org/licenses/by-nc/4.0/
