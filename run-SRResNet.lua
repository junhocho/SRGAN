require 'nn'
require 'image'
require 'cunn'
require 'cudnn'

require 'src.util'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Run SRResNet model.')
cmd:text()
cmd:text('Options')

cmd:option('-checkpoint_path', 'models/9x9-15res-LR24/resnet-deconv-9x9-15res-LR24-80000.t7', 'checkpoint path to run')
cmd:option('-dataset', 'BSD100', 'which dataset to use? BSD100|Set14|Set5')
cmd:option('-result_path', 'imgs/results', 'path to output results. default: imgs/results. directory will be made if not exists.')
cmd:text()

local opt = cmd:parse(arg or {})
print(opt)
local loaded_checkpoint_path = torch.load(opt.checkpoint_path)
local model = loaded_checkpoint_path.model:cuda()
local datasetName = opt.dataset
local savePath = opt.result_path


-- -- model = torch.load('models/resnet-deconv-9x9-20000.t7')  -- require 'models.resnet-deconv'
-- model = torch.load('models/9x9-15res-LR24/resnet-deconv-9x9-15res-LR24-80000.t7')  -- require 'models.resnet-deconv'
-- model:cuda()
-- local datasetName = 'BSD100'
-- local savePath = 'imgs/9x9-LR24-80K' --'imgs/9x9/' --'imgs/9x9-LR24/'




local datasetPath
if datasetName == 'Set5' then
	datasetPath = '/home/junho/data/Set5/image_SRF_4/LR/'
elseif datasetName == 'Set14' then 
	datasetPath = '/home/junho/data/Set14/image_SRF_4/LR/'
elseif datasetName == 'BSD100' then
	datasetPath = '/home/junho/data/BSD100/image_SRF_4/LR/'
else
	error('no such dataset')
end

-- generate path for each dataset result
if paths.mkdir(savePath) then print(savePath .. ': new folder') end
if paths.mkdir(paths.concat(savePath, datasetName)) then print(datasetName .. ': new folder made in above folder') end

-- Lena Test. 
local do_lena = false
if do_lena then 
	-- lena
	local LRimg = image.scale(image.lena(), 96,96)
	local imgBatch =LRimg:cuda():view(1,3, LRimg:size(2), LRimg:size(3)):cuda() -- make 1 image batch for BN
	
	-- local SRimg = model:forward(imgBatch) -- forwardpass
	local ok, SRimg = pcall(function() return model:forward(imgBatch) end) -- forward pass
	
	if ok then
		local imgSaveName = 'lena' --'img_004_SRF_4_'
		image.save(savePath .. imgSaveName .. 'LR24.png', LRimg) --LRimg:view(3, LRimg:size(3), LRimg:size(4)))
		image.save(savePath .. imgSaveName .. 'SR24Resnet.png', SRimg:view(3, SRimg:size(3), SRimg:size(4)))
		-- local dataset_path = "/home/junho/data/VOCdevkit/VOC2012/JPEGImages/"
	else
		print('oom')
	end
end

-- Dataset Test
local Hmin = 80
-- local do_limitInput = false
print('Testing datasetPath: ' .. datasetPath)
for imgFile in paths.iterfiles(datasetPath) do
	local imPath = datasetPath .. imgFile
	local LRimg = image.load(imPath) -- Already LR images	
	-- print(imPath)
	-- print(#LRimg)
	
	-- local LRimg = image.scale(LRimg, Hmin) -- Temprarily downsize...
	
	-- if ~do_limitInput and LRimg:size(2) < Hmin then 
	local ok, SRimg = pcall(function() 
		local imgBatch = LRimg:cuda():view(1,3, LRimg:size(2), LRimg:size(3))
		return model:forward(imgBatch) 
	end)
	if ok then
		local saveFilePath = paths.concat(savePath, datasetName, imgFile .. '_SRResnet.png')
		image.save(saveFilePath , SRimg:view(3, SRimg:size(3), SRimg:size(4)))
	else
		print('oom')	
	end
	-- end 
end

print('run test done')
