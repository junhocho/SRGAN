# Torch7-profiling

Repository of neural network models for feed-forward speed profiling

## Running the code

```
th profile-model.lua --table <modelName>
th profile-model.lua --net <modelName>
th profile-model.lua --array <modelName>
```
The flags can be also shortened: `--net` -> `-n` and `--cuda` -> `-c`.

## Running notes on specific cores
8 cores, 4 slow and 4 fast, slow performance fix with `sudo taskset -c 4-7 th ...`
