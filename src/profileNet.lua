--------------------------------------------------------------------------------
-- Profiling network (forward only)
-- Computational time and estimation of number of operations
--------------------------------------------------------------------------------
-- Alfredo Canziani, Oct 14
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'xlua'
require 'sys'

-- Local definitions -----------------------------------------------------------
local pf = function(...) print(string.format(...)) end
local r = sys.COLORS.red
local n = sys.COLORS.none

local function time(name, model, nFeatureMaps, mapSize, iterations)
   pf('Profiling %s, %d iterations', r..name..n, iterations)

   -- Input definition ---------------------------------------------------------
   local input = torch.Tensor(nFeatureMaps[0], mapSize[0][1], mapSize[0][1])

   local timer = torch.Timer()
   for i = 1, iterations do
      xlua.progress(i, iterations)
      model:forward(input)
      if cuda then cutorch.synchronize() end
   end

   time = timer:time().real/iterations
   pf('Forward average time: %.2f ms\n', time * 1e3)

   return time

end

local function ops(nFeatureMaps, filterSize, convPadding, convStride, poolSize,
   poolStride, hiddenUnits, mapSize, time)

   local convOps = torch.Tensor(#nFeatureMaps)
   local poolOps = torch.zeros(#nFeatureMaps)
   for i, nbMap in ipairs(nFeatureMaps) do
      convOps[i] = 2 * nFeatureMaps[i-1] * nbMap * filterSize[i]^2 *
         mapSize[i][1]^2 + 2 * mapSize[i][1] -- bias + ReLU
      if poolSize[i] > 1 then
         poolOps[i] = poolSize[i]^2 * mapSize[i][2]^2
      end
   end
   local MLPOps = torch.Tensor(#hiddenUnits)
   local neurons = model.neurons
   for i, hidden in ipairs(hiddenUnits) do
      MLPOps[i] = 2 * neurons[#nFeatureMaps+i-1] * neurons[#nFeatureMaps+i] +
         2 * neurons[#nFeatureMaps+i] -- bias + ReLU
   end

   --print(convOps, poolOps, MLPOps)
   local totOps = convOps:sum() + poolOps:sum() + MLPOps:sum()
   pf('Total number of operations: %.2f G-Ops', totOps * 1e-9)
   pf('conv/pool/MLP: %.2fG/%.2fk/%.2fM-Ops\n',
      convOps:sum() * 1e-9, poolOps:sum() * 1e-3, MLPOps:sum() * 1e-6)
   pf('Performance for %sTHIS%s machine: %.2f G-Ops/s', r, n, totOps * 1e-9 / time)

end

profileNet = {time = time, ops = ops}
