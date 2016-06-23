require 'nn'
require 'xlua'
require 'sys'

local lapp = assert(require('pl.lapp'))
local opts = assert(require('opts'))
local profileTime = assert(require('src/modelTimer.lua'))

-- to load a 64-bit model in binary, we override torch.DiskFile if 32bit machine (ARM):
local systembit = tonumber(io.popen("getconf LONG_BIT"):read('*a'))
if systembit == 32 then
   require('libbincompat')
end

local pf = function(...) print(string.format(...)) end
local r = sys.COLORS.red
local g = sys.COLORS.green
local b = sys.COLORS.blue
local n = sys.COLORS.none
local THIS = sys.COLORS.blue .. 'THIS' .. n

-- Parsing input arguemnets
opt = opts.parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')

paths.dofile('src/profiler.lua')
-- Loading model
if string.find(opt.model, '.lua', #opt.model-4) then
   model = assert(require('./'..opt.model))
   pf('Building %s model from model...\n', r..model.name..n)
   net = model:mknet()
   eye = model.eye
elseif string.find(opt.model, '.net', #opt.model-4) then
   model = { channel = 3, name = 'Trained binary network' }
   pf('Loading %s model from binary file...\n', r..model.name..n)
   net = torch.load(opt.model, 'binary')
elseif string.find(opt.model, '.net.ascii', #opt.model-10) then
   model = { channel = 3, name = 'Trained ascii network' }
   pf('Loading %s model from ascii file...\n', r..model.name..n)
   net = torch.load(opt.model, 'ascii')
else
   error('Network named not recognized')
end

local iBatch, iChannel, iWidth, iHeight = string.match(opt.res, '(%d+)x(%d+)x(%d+)x(%d+)')
--                                  or string.match(opt.res, '(%d+)X(%d+)X(%d+)')

iBatch = tonumber(iBatch)
iChannel = tonumber(iChannel)
iWidth = tonumber(iWidth)
iHeight = tonumber(iHeight)

if iChannel ~= 0 then
   model.channel = iChannel
end

eye = eye or 100
if opt.eye ~= 0 then
   eye = opt.eye
end

local batch    = (iBatch ~= 0) and iBatch
local width    = (iWidth ~= 0) and iWidth or eye
local height   = (iHeight ~= 0) and iHeight or width

img = torch.FloatTensor(model.channel, height, width)
imgBatch = torch.FloatTensor(batch, model.channel, height, width)

if opt.save == 'a' then
   pf('Saving model as model.net.ascii... ')
   torch.save('model.net.ascii', net, 'ascii')
   pf('Done.\n')
elseif opt.save == 'b' then
   pf('Saving model as model.net... ')
   torch.save('model.net', net)
   pf('Done.\n')
end

-- calculate the number of operations performed by the network
if not model.def then
   totalOps, layerOps = count_ops(net, imgBatch)
else
   totalOps, layerOps = count_ops(model.def, imgBatch)
end

pf('Operations estimation for image size: %d X %d', width, height)


-- Compute per layer opt counts
print('\n-------------------------------------------------------------------------------------')
pf('%5s %-29s %20s %11s %15s', 'S.No.', 'Module Name', 'Input Resolution', 'Neurons', 'Operations')
print('=====================================================================================')
local opsPerCommonModule = {}
local totalNeurons = 0
for i, info in pairs(layerOps) do
    local name = info['name']
    local ops = info['ops']
    local maps = info['maps']
    local neurons = info['neurons']
    if not opsPerCommonModule[name] then
        opsPerCommonModule[name] = 0
    end
    pf('%5d %s%-29s%s %s%20s%s %11s %s%15s%s Ops', i, g, name, n, r, maps, n, neurons, r, ops, n)
    totalNeurons = totalNeurons + neurons
    opsPerCommonModule[name] = opsPerCommonModule[name] + ops
end

print('-------------------------------------------------------------------------------------')
pf('   %s%s%s %d ', r, 'Total number of trainable parameters:', n, net:getParameters():size(1))
pf('   %s%-37s%s %d', r, 'Total number of neurons:', n, totalNeurons)
print('-------------------------------------------------------------------------------------')
print('Operations per common module')
-- Print total
local ops = opt.MACs and 'MACs' or 'Ops'
for name, count in pairs(opsPerCommonModule) do
    if count > 0 then
        print(string.format('   %-38s%.4e %s', name..':', count, ops))
    end
end
pf('   %s%-38s%.4e %s', b, 'Total:', totalOps, ops)
print('=====================================================================================')

-- time and average over a number of iterations
pf('Profiling %s, %d iterations', r..model.name..n, opt.iter)
time = profileTime:time(net, imgBatch, opt.iter, opt.platform)

local d = g..'CPU'..n
if 'cuda' == opt.platform then
   d = g..'GPU'..n
end

pf('   Forward average time on %s %s: %.2f ms', THIS, d, time.total * 1e3)
if (time.conv ~= 0) and (time.mlp ~= 0) then
   pf('    + Convolution time: %.2f ms', time.conv * 1e3)
   pf('    + MLP time: %.2f ms', time.mlp * 1e3)
end

pf('   Performance for %s %s: %.2f G-Ops/s\n', THIS, d, totalOps * 1e-9 / time.total)


