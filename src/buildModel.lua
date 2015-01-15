--------------------------------------------------------------------------------
-- Build model, given its parameters
-- No <dropout> since only forward model is considered
--------------------------------------------------------------------------------
-- Alfredo Canziani, Oct 14
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'sys'

-- Local definitions -----------------------------------------------------------
local pf = function(...) io.write(string.format(...)); io.flush() end
local r = sys.COLORS.red
local n = sys.COLORS.none

-- Public function -------------------------------------------------------------
function buildModel(name, nFeatureMaps, filterSize, convPadding, convStride,
   poolSize, poolStride, hiddenUnits, mapSize, cuda, save)
   pf('Building %s model...\n', r..name..n)
   collectgarbage()

   -- Computing useful figures -------------------------------------------------
   -- Feature maps size and neurons number
   local neurons = {}
   neurons.real = {}
   neurons.pool = {}
   mapSize.real = {[0] = mapSize[0]}
   mapSize.pool = {[0] = mapSize[0]}
   mapSize[0]   = nil
   local f = math.floor
   for i = 1, #nFeatureMaps do
      mapSize.real[i] = f((mapSize.pool[i-1] + 2 * convPadding[i] - filterSize[i]) /
         convStride[i]) + 1
      mapSize.pool[i] = f((mapSize.real[i] - poolSize[i]) / poolStride[i]) + 1
      neurons.real[i] = mapSize.real[i]^2 * nFeatureMaps[i]
      neurons.pool[i] = mapSize.pool[i]^2 * nFeatureMaps[i]
   end
   for _, h in ipairs(hiddenUnits) do
      table.insert(neurons.real, h)
      table.insert(neurons.pool, h)
   end

   -- Model definition ---------------------------------------------------------
   -- Convolution container
   local convBlock = nn.Sequential()

   for i = 1, #nFeatureMaps do

      -- Convolution
      if not cuda and convStride[i] > 1 then -- non Cuda MM stride not supported
         p = convPadding[i]
         if p > 0 then
            convBlock:add(nn.SpatialZeroPadding(p,p,p,p))
         end
         convBlock:add(
            nn.SpatialConvolution(
               nFeatureMaps[i-1], nFeatureMaps[i],
               filterSize[i], filterSize[i],
               convStride[i], convStride[i],
               convPadding[i])
            )
      else
         convBlock:add(
            nn.SpatialConvolutionMM(
               nFeatureMaps[i-1], nFeatureMaps[i],
               filterSize[i], filterSize[i],
               convStride[i], convStride[i],
               convPadding[i])
            )
      end

      -- Non linearity
      convBlock:add(nn.ReLU())

      -- Pooling
      if poolSize[i] > 1 then
         convBlock:add(
            nn.SpatialMaxPooling(
               poolSize[i], poolSize[i],
               poolStride[i], poolStride[i])
            )
      end

   end

   -- MLP
   -- Defining classifier
   local classifier = nn.Sequential()

   classifier:add(nn.Reshape(neurons.pool[#nFeatureMaps]))

   for i = 1, #hiddenUnits do
      classifier:add(
         nn.Linear(neurons.pool[#nFeatureMaps+i-1],neurons.pool[#nFeatureMaps+i])
         )
      if i < #hiddenUnits then
         classifier:add(nn.ReLU())
      else
         classifier:add(nn.LogSoftMax())
      end
   end

   -- Full model
   -- Defining container
   local model = nn.Sequential()
   model:add(convBlock)
   model:add(classifier)
   model.neurons = neurons

   pf('   Total number of neurons: %d\n', torch.Tensor(neurons.pool):sum())
   pf('   Total number of trainable parameters: %d\n',
      model:getParameters():size(1))

   if save == 'a' then
      pf('Saving model as model.net.ascii... ')
      torch.save('model.net.ascii', model, 'ascii')
      pf('Done.\n')
   elseif save == 'b' then
      pf('Saving model as model.net... ')
      torch.save('model.net', model)
      pf('Done.\n')
   end

   if cuda then
      require 'cunn'
      model:cuda()
   end

   return model

end
