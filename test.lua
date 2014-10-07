--------------------------------------------------------------------------------
-- Testing network
--------------------------------------------------------------------------------

dofile 'VGG-D.lua'
require 'xlua'

timer = torch.Timer()
for i = 1, 100 do
   xlua.progress(i,100)
   model:forward(input)
end
print(string.format('Forward time %.2f ms\n', timer:time().real*10))

print('{model.modules[]}')
print('{convBlock.modules[]}')
print('{classifier.modules[]}')

