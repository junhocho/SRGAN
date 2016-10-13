require 'nn'
require 'image'
require 'cunn'
require 'cudnn'
-- debugger = require 'fb.debugger'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train SRResNet model.')
cmd:text()
cmd:text('Options')

cmd:option('-model_name', '9x9-15res-LR24', 'will save checkpoints in models/model_name/ ')
cmd:option('-checkpoint_start_from', '' , 'start training from checkpoint if given. If not given, train from scratch')
cmd:option('-lr', 10e-4, 'learning rate')
cmd:option('-beta', 0.9 , 'beta')
-- cmd:option('-iter_start', 1, 'not to overwrite previous trained model when resumed. ')
cmd:option('-iter_end', 10e6, 'iter to end training')

cmd:text()

local opt = cmd:parse(arg or {})
print(opt)

-- Load checkpoint if given OR train from scratch
if string.len(opt.checkpoint_start_from) > 0 then 
	local loaded_checkpoint = torch.load(opt.checkpoint_start_from) -- resume training
	model = loaded_checkpoint.model
	iter_start = loaded_checkpoint.iter + 1
else
	model = require 'models.resnet-deconv2' -- train from scratch
	iter_start = 1
end
model:cuda()
-- -- model = torch.load('models/resnet-deconv30000.t7') 
-- model = require 'models.resnet-deconv2'
-- model:cuda

local saveCheckpointPath = paths.concat('models/', opt.model_name)

-- loss function
local loss = nn.MSECriterion():cuda()
local theta, gradTheta = model:getParameters()

-- config to adam
local config = {}
config.learningRate = opt.lr -- 10e-4
config.optim_beta = opt.beta --0.9 -- 0.999/
--config.optim_alpha = 0.9
--config.optim_epsilon = 10e-8

local optim_state = {}

require 'optim'
require 'src.util'

local imgBatch = {} -- input SR, LR
imgBatch.batchNum = 16

-- VOC
-- local datasetPath = "/home/junho/data/VOCdevkit/VOC2012/JPEGImages/"
-- imgBatch.imgPaths, imgBatch.imgNum = prepImgs(datasetPath)

local datasetPath = "/home/junho/data/ImageNet/"
-- imgBatch.imgPaths, imgBatch.imgNum = prepImageNet(datasetPath)
-- Save paths
--torch.save('imgBatch.t7', imgBatch)


local imgBatch = torch.load('imgBatch.t7')

-- print(imgBatch.imgPaths)
print('ImageNet loaded, # of imgs:' .. imgBatch.imgNum)

function feval(theta)
	gradTheta:zero()
	-- print(imgBatch.LR:cuda())
	local X = imgBatch.LR 
	local h_x = model:forward(X)
	local J = loss:forward(h_x, imgBatch.SR)
	-- print(#h_x)
	local dJ_dh_x = loss:backward(h_x, imgBatch.SR)
	print(J)
	model:backward(X, dJ_dh_x)
	return J, gradTheta
end

require 'optim'
-- all images in datasetPath
for iter = iter_start, opt.iter_end do -- start from checkpoint.iter +1    -- 1,10e6 do -- 3e4+1, 1e6 do
	setBatch(imgBatch)
	print('iter:' .. iter) -- debug
	optim.adam(feval, theta, config, optim_state)

	if iter % 10000 == 0 then 
		local checkpoint = {}
		checkpoint.opt = opt
		checkpoint.iter = iter
		checkpoint.model = model

		print('saving model' .. iter)
		if paths.mkdir(saveCheckpointPath) then print(saveCheckpointPath .. ': new folder to save model') end
		torch.save(saveCheckpointPath .. '/' .. iter .. '.t7', checkpoint) --model)
		print('saved model, next will be: ' .. iter+1)
	end
end


