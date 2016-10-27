require 'image'
require 'nn'
require 'cunn'
lena = image.lena():cuda():view(1,3,512,512)

vgg = torch.load('VGG/VGG19.t7')

vgg:cuda()

require 'src.vgg-util'
out = vgg:forward(vgg_preprocess(lena))
