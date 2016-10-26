require 'image'
require 'cunn'
require 'cudnn'
x1 = image.scale(image.lena(), 96):cuda()
x2 = image.scale(image.lena(), 96):cuda()

input = torch.cat( x1:view(1, 3, 96, 96),
					x2:view(1, 3, 96, 96) , 1)

net = require 'models.resnet-deconv'

out = net:forward(input)
print(#out)
