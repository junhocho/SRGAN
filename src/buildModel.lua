--------------------------------------------------------------------------------
-- Build model, given its parameters
-- No <dropout> since only forward model is considered
--------------------------------------------------------------------------------
-- Alfredo Canziani, Oct 14
--------------------------------------------------------------------------------

function buildModel(nFeatureMaps, filterSize, convPadding, convStride, poolSize,
   poolStride, hiddenUnits, mapSize)

   -- Computing useful figures -------------------------------------------------
   -- Feature maps size and neurons number
   local neurons = {}
   for i = 1, #nFeatureMaps do
      mapSize[i] = (mapSize[i-1] + 2*convPadding[i] - filterSize[i]) / convStride[i] + 1
      mapSize[i] = (mapSize[i] - poolSize[i]) / poolStride[i] + 1
      neurons[i] = mapSize[i]^2 * nFeatureMaps[i]
   end
   for _, h in ipairs(hiddenUnits) do
      table.insert(neurons, h)
   end

   -- Model definition ---------------------------------------------------------
   -- Convolution container
   local convBlock = nn.Sequential()

   for i, nbMap in ipairs(nFeatureMaps) do

      -- Convolution
      convBlock:add(
         nn.SpatialConvolutionMM(
            nFeatureMaps[i-1], nbMap,
            filterSize[i], filterSize[i],
            convStride[i], convStride[i],
            convPadding[i])
         )

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

   classifier:add(nn.Reshape(neurons[#nFeatureMaps]))

   for i = 1, #hiddenUnits do
      classifier:add(
         nn.Linear(neurons[#nFeatureMaps+i-1],neurons[#nFeatureMaps+i])
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
   model.mapSize = mapSize
   model.neurons = neurons

   return model

end
