require 'nn'
require 'xlua'
require 'sys'
local lapp = assert(require('pl.lapp'))
local build = assert(require('src/builder'))
local profile = assert(require('src/profiler'))

local pf = function(...) print(string.format(...)) end
local r = sys.COLORS.red
local g = sys.COLORS.green
local n = sys.COLORS.none
local THIS = sys.COLORS.blue .. 'THIS' .. n

local opt = lapp [[
 -t, --table   (default ./tables/Kriz.lua) Network to profile
 -n, --net     (default '')   Network to profile
 -a, --array   (default '')   Network to profile

 -p, --platform   (default cpu)  Select profiling platform (cpu|cuda|nnx)
 -e, --eye        (default 0)    Network eye
 -i, --iter       (default 10)   Averaging iterations
 -s, --save       (default -)    Save the float model to file as <model.net.ascii>in
                                 [a]scii or as <model.net> in [b]inary format (a|b)
]]
torch.setdefaulttensortype('torch.FloatTensor')



if opt.net ~= '' then
   -- get network definition
   model = assert(require(opt.net))
   pf('Building %s model from network...\n', r..model.name..n)
   net = model:mknet()
   eye = model.eye
elseif opt.array ~= '' then
   -- get network definition
   model = assert(require(opt.array))
   pf('Building %s model from array...\n', r..model.name..n)
   net = model:mknet()
   eye = model.eye
else
   -- get table definition
   model = assert(require(opt.table))
   pf('Building %s model from table...\n', r..model.name..n)
   net, eye = build:cpu(model)
   pf('\n')
end

eye = eye or 100
if opt.eye ~= 0 then
   eye = opt.eye
end
img = torch.FloatTensor(model.channel, eye, eye)

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
ops = profile:ops(net, img)
ops_total = ops.conv + ops.pool + ops.mlp

pf('   Total number of neurons: %d', ops.neurons)
pf('   Total number of trainable parameters: %d', net:getParameters():size(1))
pf('   Operations estimation for square image side: %d', eye)
pf('    + Total: %.2f G-Ops', ops_total * 1e-9)
pf('    + Conv/Pool/MLP: %.2fG/%.2fk/%.2fM(-Ops)\n',
   ops.conv * 1e-9, ops.pool * 1e-3, ops.mlp * 1e-6)


-- time and average over a number of iterations
pf('Profiling %s, %d iterations', r..model.name..n, opt.iter)
time = profile:time(net, img, opt.iter, opt.platform)

local d = g..'CPU'..n
if 'cuda' == opt.platform then
   d = g..'GPU'..n
elseif 'nnx' == opt.platform then
   d = g..'nnX'..n
end

pf('   Forward average time on %s %s: %.2f ms', THIS, d, time.total * 1e3)
if (time.conv ~= 0) and (time.mlp ~= 0) then
   pf('    + Convolution time: %.2f ms', time.conv * 1e3)
   pf('    + MLP time: %.2f ms', time.mlp * 1e3)
end

pf('   Performance for %s %s: %.2f G-Ops/s\n', THIS, d, ops_total * 1e-9 / time.total)
