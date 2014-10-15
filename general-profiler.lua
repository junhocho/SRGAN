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
local opt = lapp [[
 -n, --net  (string)     Network to profile (VGG-D,Kriz)
 -c, --cuda              Cuda option, default false
 -i, --iter (default 10) Averaging iterations
]]
torch.setdefaulttensortype('torch.FloatTensor')

-- Get model -------------------------------------------------------------------
if opt.net == 'VGG-D' then require 'models/VGG-D'
elseif opt.net == 'Kriz' then require 'models/Kriz'
else error('Unknown model') end

-- Building model --------------------------------------------------------------
model = buildModel(name, nFeatureMaps, filterSize, convPadding, convStride,
   poolSize, poolStride, hiddenUnits, mapSize, opt.cuda)

-- Profile net (forward only) --------------------------------------------------
operations = profileNet.ops(nFeatureMaps, filterSize, convPadding, convStride,
   poolSize, poolStride, hiddenUnits, mapSize)

time = profileNet.time(name, model, nFeatureMaps, mapSize, opt.iter, opt.cuda,
   operations)
