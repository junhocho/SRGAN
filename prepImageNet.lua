require 'src/util'

local imgBatch = {}
local datasetPath = "/home/junho/data/ImageNet/"
imgBatch.imgPaths, imgBatch.imgNum = prepImageNetClass(datasetPath) 
print('done prepImageNet and save as imgBatch.t7') 


torch.save('imgBatch-bird.t7', imgBatch)


