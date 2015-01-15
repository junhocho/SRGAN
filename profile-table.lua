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
 -n, --net  (default halfKriz)   Network to profile (VGG-D | Kriz | HW04 |
                           4-16Test | CamFind1 | halfKriz | largeNetTest)

 -i, --iter (default 10)   Averaging iterations
 -s, --save (default -)    Save the float model to file as <model.net.ascii> in
                           [a]scii or as <model.net> in [b]inary format (a|b)
]]
torch.setdefaulttensortype('torch.FloatTensor')

-- Get model definition
model = require('tables/' .. opt.net)

pf('Building %s model...', r..model.name..n)
net, eye = build:cpu(model)
eye = eye or 100

if opt.save == 'a' then
   pf('Saving model as model.net.ascii... ')
   torch.save('model.net.ascii', net, 'ascii')
   pf('Done.\n')
elseif opt.save == 'b' then
   pf('Saving model as model.net... ')
   torch.save('model.net', net)
   pf('Done.\n')
end

ops = profile:ops(model, {
   width  = eye,
   height = eye,
})
ops_total = ops.conv + ops.pool + ops.mlp

pf('   Total number of neurons: %d', ops.neurons)
pf('   Total number of trainable parameters: %d', net:getParameters():size(1))
pf('   Operations estimation for square image side: %d', eye)
pf('    + Total: %.2f G-Ops', ops_total * 1e-9)
pf('    + Conv/Pool/MLP: %.2fG/%.2fk/%.2fM(-Ops)\n',
   ops.conv * 1e-9, ops.pool * 1e-3, ops.mlp * 1e-6)


pf('Profiling %s, %d iterations', r..model.name..n, opt.iter)
time, time_conv, time_mlp = profile:time(model, net, opt.iter, {
   width  = eye,
   height = eye,
})

local d = g..'CPU'..n
pf('   Forward average time on %s %s: %.2f ms', THIS, d, time * 1e3)
if (time_conv ~= 0) and (time_mlp ~= 0) then
   pf('    + Convolution time: %.2f ms', time_conv * 1e3)
   pf('    + MLP time: %.2f ms', time_mlp * 1e3)
end

pf('   Performance for %s %s: %.2f G-Ops/s\n', THIS, d, ops_total * 1e-9 / time)
