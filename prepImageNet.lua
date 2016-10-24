require 'src/util'

local imgBatch = {}
local datasetPath = "/home/junho/data/ImageNet/"
imgBatch.imgPaths, imgBatch.imgNum = prepImageNet(datasetPath) 
print('done prepImageNet and save as imgBatch.t7') 


torch.save('imgBatch.t7', imgBatch)


