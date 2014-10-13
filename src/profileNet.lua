--------------------------------------------------------------------------------
-- Profiling network (forward only)
-- Computational time and estimation of number of operations
--------------------------------------------------------------------------------
-- Alfredo Canziani, Oct 14
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'xlua'

local function time(name, model, nFeatureMaps, mapSize, iterations)
   print(string.format('Profiling %s, %d iterations', name, iterations))

   -- Input definition ---------------------------------------------------------
   local input = torch.Tensor(nFeatureMaps[0], mapSize[0], mapSize[0])

   local timer = torch.Timer()
   for i = 1, iterations do
      xlua.progress(i, iterations)
      model:forward(input)
      if cuda then cutorch.synchronize() end
   end

   time = timer:time().real/iterations
   print(string.format('Forward average time %.2f ms\n', time * 1e3))

   return time

end

local function ops(nFeatureMaps, filterSize, convPadding, convStride, poolSize,
   poolStride, hiddenUnits, mapSize, time)

end

profileNet = {time = time, ops = ops}
