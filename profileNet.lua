--------------------------------------------------------------------------------
-- Profiling network (forward only)
--------------------------------------------------------------------------------
-- Alfredo Canziani, Oct 14
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'xlua'

function profileNet(model, nFeatureMaps, mapSize, iterations)
   -- Input definition ---------------------------------------------------------
   local input = torch.Tensor(nFeatureMaps[0], mapSize[0], mapSize[0])

   local timer = torch.Timer()
   for i = 1, iterations do
      xlua.progress(i, iterations)
      model:forward(input)
      if cuda then cutorch.synchronize() end
   end

   time = timer:time().real/iterations
   print(string.format('Forward time %.2f ms\n', time))

   return time

end
