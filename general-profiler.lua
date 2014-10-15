--------------------------------------------------------------------------------
-- Profiler main file
--------------------------------------------------------------------------------
-- Alfredo Canziani, Oct 14
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'nn'
require 'src/buildModel'
require 'src/profileNet'
require 'pl'

-- Options ---------------------------------------------------------------------
opt = lapp [[
 -n, --net (string) Network to profile (VGG-D,Kriz)
]]
torch.setdefaulttensortype('torch.FloatTensor')

-- Get model -------------------------------------------------------------------
if opt.net == 'VGG-D' then require 'models/VGG-D'
elseif opt.net == 'Kriz' then require 'models/Kriz'
else error('Unknown model') end

-- Building model --------------------------------------------------------------
model = buildModel(name, nFeatureMaps, filterSize, convPadding, convStride,
   poolSize, poolStride, hiddenUnits, mapSize)

-- Profile net (forward only) --------------------------------------------------
operations = profileNet.ops(nFeatureMaps, filterSize, convPadding, convStride,
   poolSize, poolStride, hiddenUnits, mapSize)

iterations = 10; cuda = false
time = profileNet.time(name, model, nFeatureMaps, mapSize, iterations, cuda,
   operations)
