# State Of The Art (SOTA) models 

Repository of SOTA deep neural network models for feed-forward speed profiling

## Running notes on Odroid XU3
8 cores, 4 slow and 4 fast, slow performance (4172.40 ms) fix with `sudo taskset -c 4-7 th test.lua`

## VGG (ILSVRC 14 classification winner)

| Device      | Time [ms] |
|-------------|----------:|
| MacBook Pro | 418.33    |
| Odroid XU3  | 3237.75   |
