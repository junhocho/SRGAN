--------------------------------------------------------------------------------
-- Profiler main file
--------------------------------------------------------------------------------
-- Alfredo Canziani, Oct 14
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'nn'
require 'src/buildModel'
require 'src/profileNet'
lapp = require 'pl.lapp'

-- Options ---------------------------------------------------------------------
local opt = lapp [[
 -n, --net  (string)     Network to profile (VGG-D,Kriz)
 -c, --cuda              Cuda option, default false
 -i, --iter (default 10) Averaging iterations
]]
torch.setdefaulttensortype('torch.FloatTensor')

-- Get model -------------------------------------------------------------------
require('models/' .. opt.net)

-- Building model --------------------------------------------------------------
model = buildModel(name, nFeatureMaps, filterSize, convPadding, convStride,
   poolSize, poolStride, hiddenUnits, mapSize, opt.cuda)

-- Profile net (forward only) --------------------------------------------------
operations = profileNet.ops(nFeatureMaps, filterSize, convPadding, convStride,
   poolSize, poolStride, hiddenUnits, mapSize)

time = profileNet.time(name, model, nFeatureMaps, mapSize, opt.iter, opt.cuda,
   operations)
