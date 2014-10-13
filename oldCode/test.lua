--------------------------------------------------------------------------------
-- Testing network
--------------------------------------------------------------------------------
require 'pl'
require 'xlua'

opt = lapp([[
--input     (default float)       Input type, float or cuda
--cudadev   (default 1)           Cuda device #
]])

if opt.input == 'cuda' then
   require 'cunn'
   -- dev:
   cutorch.setDevice(opt.cudadev or 1)
   print('DEVID = ' .. cutorch.getDevice())
end

dofile 'VGG-D.lua'

if opt.input == 'cuda' then
   model:cuda()
end

timer = torch.Timer()
for i = 1, 100 do
   xlua.progress(i,100)
   model:forward(input)
   if opt.input == 'cuda' then cutorch.synchronize() end
end
print(string.format('Forward time %.2f ms\n', timer:time().real*10))

print('{model.modules[]}')
print('{convBlock.modules[]}')
print('{classifier.modules[]}')

