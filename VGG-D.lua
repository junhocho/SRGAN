--------------------------------------------------------------------------------
-- ILSVRC 2014 classification winner
-- VGG, single network D
--------------------------------------------------------------------------------
-- Very Deep Convolutional Networks for Large-Scale Image Recognition
-- http://arxiv.org/abs/1409.1556
--------------------------------------------------------------------------------
-- Alfredo Canziani, Oct 14
--------------------------------------------------------------------------------

-- Requires --------------------------------------------------------------------
require 'nn'

-- Options ---------------------------------------------------------------------
torch.setdefaulttensortype('torch.FloatTensor')

-- Input definition ------------------------------------------------------------
input = torch.Tensor(3,224,224)
if opt.input == 'cuda' then input = input:cuda() end

-- Model definition ------------------------------------------------------------
-- Convolution container
convBlock = nn.Sequential()

convBlock:add(nn.SpatialConvolutionMM(3,64,3,3,1,1,1))
convBlock:add(nn.ReLU())
convBlock:add(nn.SpatialConvolutionMM(64,64,3,3,1,1,1))
convBlock:add(nn.ReLU())

convBlock:add(nn.SpatialMaxPooling(2,2,2,2))

convBlock:add(nn.SpatialConvolutionMM(64,128,3,3,1,1,1))
convBlock:add(nn.ReLU())
convBlock:add(nn.SpatialConvolutionMM(128,128,3,3,1,1,1))
convBlock:add(nn.ReLU())

convBlock:add(nn.SpatialMaxPooling(2,2,2,2))

convBlock:add(nn.SpatialConvolutionMM(128,256,3,3,1,1,1))
convBlock:add(nn.ReLU())
convBlock:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1))
convBlock:add(nn.ReLU())
convBlock:add(nn.SpatialConvolutionMM(256,256,3,3,1,1,1))
convBlock:add(nn.ReLU())

convBlock:add(nn.SpatialMaxPooling(2,2,2,2))

convBlock:add(nn.SpatialConvolutionMM(256,512,3,3,1,1,1))
convBlock:add(nn.ReLU())
convBlock:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1))
convBlock:add(nn.ReLU())
convBlock:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1))
convBlock:add(nn.ReLU())

convBlock:add(nn.SpatialMaxPooling(2,2,2,2))

convBlock:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1))
convBlock:add(nn.ReLU())
convBlock:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1))
convBlock:add(nn.ReLU())
convBlock:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1))
convBlock:add(nn.ReLU())

convBlock:add(nn.SpatialMaxPooling(2,2,2,2))

-- MLP
-- Defining classifier
classifier = nn.Sequential()

classifier = nn.Sequential()

classifier:add(nn.Reshape(512 * 7^2))

classifier:add(nn.Linear(512 * 7^2, 4096))
classifier:add(nn.ReLU())

classifier:add(nn.Linear(4096, 4096))
classifier:add(nn.ReLU())

classifier:add(nn.Linear(4096, 1024))
classifier:add(nn.LogSoftMax())

-- Full model
-- Defining container
model = nn.Sequential()
model:add(convBlock)
model:add(classifier)

