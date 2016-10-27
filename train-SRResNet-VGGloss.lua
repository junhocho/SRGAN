require 'nn'
require 'image'
require 'cunn'
require 'cudnn'


debugger = require 'fb.debugger'

-- Debugging
display = require 'display'



cmd = torch.CmdLine()
cmd:text()
cmd:text('Train SRResNet model.')
cmd:text()
cmd:text('Options')

cmd:option('-model_name', '9x9-15res-LR24', 'will save checkpoints in checkpoints/model_name/ ')
cmd:option('-checkpoint_start_from', '' , 'start training from checkpoint if given. If not given, train from scratch')
cmd:option('-arch', '', 'if checkpoint not and arch is given, use the architecture')

cmd:option('-lr', 10e-5, 'learning rate')
cmd:option('-beta', 0.9 , 'beta')
-- cmd:option('-iter_start', 1, 'not to overwrite previous trained model when resumed. ')
cmd:option('-iter_end', 10e6, 'iter to end training')
cmd:option('-checkpoint_save_iter', 10000, 'saver period')
cmd:text()

local opt = cmd:parse(arg or {})
print(opt)

-- Load checkpoint if given OR train from scratch
if string.len(opt.checkpoint_start_from) > 0 then 
	local loaded_checkpoint = torch.load(opt.checkpoint_start_from) -- resume training
	model = loaded_checkpoint.model
	iter_start = loaded_checkpoint.iter + 1
else
	if string.len(opt.arch) > 0 then
		model = require(opt.arch)
		iter_start = 1
	else
		model = require 'models.resnet-deconv2' -- train from scratch
		iter_start = 1
	end
end
print("resnet loaded")
model:cuda()


-- 1. Decide feature map in VGG19 for Euclidean loss.
-- 2. ~~Forward imgBatch.imgNum x 2 (Generated and GT)to VGG19.~~
--   --> model:add(VGG) and seperate VGG.
-- 3. Divide into Feature map of Generated and GT.
-- 4. Compute MSE and backpropagate into Generator.

-- VGG model load
VGG19 = torch.load('VGG/VGG19.t7')

local VGGloss_type = '2,2' -- '5,4'
if VGGloss_type == '2,2' then
	for _ = 1,28 do 
		VGG19:remove()
	end
elseif VGGloss_type == '5,4' then
	VGG19:remove()
end
print("VGG loaded")
VGG19:cuda()

-- -- model = torch.load('models/resnet-deconv30000.t7') 
-- model = require 'models.resnet-deconv2'
-- model:cuda

local saveCheckpointPath = paths.concat('checkpoints/', opt.model_name)

-- loss function
local loss = nn.MSECriterion():cuda()
local theta, gradTheta = model:getParameters()
local theta_vgg, gradTheta_vgg = VGG19:getParameters()

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

-- VOC
-- local datasetPath = "/home/junho/data/VOCdevkit/VOC2012/JPEGImages/"
-- imgBatch.imgPaths, imgBatch.imgNum = prepImgs(datasetPath)

local do_prepImageNet = false

if do_prepImageNet then
	local datasetPath = "/home/junho/data/ImageNet/"
	imgBatch.imgPaths, imgBatch.imgNum = prepImageNet(datasetPath)
	print('prepImageNet')
	-- Save paths
	torch.save('imgBatch.t7', imgBatch)
else
	imgBatch = torch.load('imgBatch.t7')
end



imgBatch.batchNum = 32
imgBatch.res = 96 --192 -- 288-- 288
-- print(imgBatch.imgPaths)
print('ImageNet loaded, # of imgs:' .. imgBatch.imgNum)


local vgg_mean = {103.939, 116.779, 123.68} 
function vgg_preprocess(img)
	local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
	local perm = torch.LongTensor{3, 2, 1}
	return img:index(2, perm):mul(255):add(-1, mean) 
end

function vgg_deprocess(img)
	local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
	local perm = torch.LongTensor{3, 2, 1}
	return (img + mean):div(255):index(2, perm)
end

function feval(theta)
	gradTheta:zero()
	gradTheta_vgg:zero()
	-- print(imgBatch.LR:cuda())
	local X = imgBatch.LR 
	local h_x = model:forward(X)

	-- VGG feature on GT
	local vgg_GT = VGG19:forward(vgg_preprocess(imgBatch.SR)):clone() -- output is pointer
	-- VGG feature on genSR
	local hp_x = vgg_preprocess(h_x) -- hp_x is preprocessof h_x
	local vgg_hp_x = VGG19:forward(hp_x)
	-- VGG loss
	local J = loss:forward(vgg_hp_x, vgg_GT)
	local dJ_dvgg_hp_x = loss:backward(vgg_hp_x, vgg_GT)

	print(J)

	local dJ_dhp_x = VGG19:backward(hp_x, dJ_dvgg_hp_x)
    local dJ_dh_x = dJ_dhp_x:div(255):index(2, torch.LongTensor{3,2,1}) -- deprocess gradient.
	
	-- debugger.enter()
	
	model:backward(X, dJ_dh_x) 

	return J, gradTheta
end

require 'optim'
-- all images in datasetPath
for iter = iter_start, opt.iter_end do -- start from checkpoint.iter +1    -- 1,10e6 do -- 3e4+1, 1e6 do
	setBatch(imgBatch)
	print('iter:' .. iter) -- debug
	optim.adam(feval, theta, config, optim_state)

	-- Visualize
	if iter % 10 == 0 then
		local X = imgBatch.LR[1]
		local GT = imgBatch.SR[1]:view(3,96,96)
		local Gen = model:forward(X:view(1,3,24,24)):view(3,96,96)
		display.image(torch.cat(GT, Gen))
	end

	if iter % opt.checkpoint_save_iter == 0 then 
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


