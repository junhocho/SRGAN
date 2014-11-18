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
 -n, --net  (string)     Network to profile (VGG-D|Kriz|HW04|4-16Test|CamFind1|
                         halfKriz|largeNetTest)
 -c, --cuda              Cuda option, default false
 -i, --iter (default 10) Averaging iterations
 -s, --save (default -)  Save the float model to file as <model.net.ascii> in
                         [a]scii or as <model.net> in [b]inary format (a|b)
]]
torch.setdefaulttensortype('torch.FloatTensor')

-- Get model definition --------------------------------------------------------
require('models/' .. opt.net)

-- Building model --------------------------------------------------------------
model = buildModel(name, nFeatureMaps, filterSize, convPadding, convStride,
   poolSize, poolStride, hiddenUnits, mapSize, opt.cuda, opt.save)

-- Profile net (forward only) --------------------------------------------------
operations = profileNet.ops(nFeatureMaps, filterSize, convPadding, convStride,
   poolSize, poolStride, hiddenUnits, mapSize)

time = profileNet.time(name, model, nFeatureMaps, mapSize, opt.iter, opt.cuda,
   operations)
