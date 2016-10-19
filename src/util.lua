require 'image'


function crop_SR_LR_patches(imgPath, res)
  local r = 4
  local w_SRPatch = res -- 96 -- 288 -- 96 -- 384
  local h_SRPatch = res -- 96 -- 288 -- 96 -- 384

  local w_LRPatch = w_SRPatch/r
  local h_LRPatch = h_SRPatch/r

  -- local img = image.lena() --jh: Too BIG Mistake. Never loaded different image.
  local img = image.load(imgPath)
  local c = img:size(1)
  local w = img:size(2)
  local h = img:size(3)
  if c == 1 then
	  img = image.lena() 
	  print("load BW image. load Lena instead")
  elseif w < w_SRPatch or h < h_SRPatch then
	  img = image.lena()
	  print("Image has too low resolution. Load Lena istead")
  end
  local h = img:size(2)
  local w = img:size(3)
  local Xmin = math.floor(torch.uniform(0,w - w_SRPatch))
  local Ymin = math.floor(torch.uniform(0,h - h_SRPatch))
  -- local Xmin = 0
  -- local Ymin = 0
  local Xmax = Xmin + w_SRPatch
  local Ymax = Ymin + h_SRPatch
  -- print(imgPath)
  -- print(w, h, Xmin, Ymin, Xmax, Ymax)
  local SRPatch = image.crop(img, Xmin, Ymin, Xmax , Ymax)
  local LRPatch = image.scale(SRPatch, w_LRPatch, h_LRPatch)
  return SRPatch, LRPatch
end

function setBatch(B) -- imgPaths, imgNum) -- s, batchNum) -- TODO img --> imgs & batchNum
	local imgs = {}

	-- random batchNum ima:w
	-- ges
	for i = 1, B.batchNum do
		local imgidx = math.floor(torch.uniform(1, B.imgNum + 1))
		-- print('imgidx: ' .. imgidx) -- debug
		imgs[i] = B.imgPaths[imgidx] -- 1 to imgNum
	end

	-- local img = image.lena():cuda()
	local SRPatch , LRPatch
	SRPatch, LRPatch = crop_SR_LR_patches(imgs[1], B.res)
	B.SR = SRPatch:clone():view(1, 3, SRPatch:size(2), SRPatch:size(3))
	B.LR = LRPatch:clone():view(1, 3, LRPatch:size(2), LRPatch:size(3))
	
	-- Debug Sample Patch
	-- print('save Sample SRLRPatch')
	-- image.save('SRPatch_sample.png', SRPatch)
	-- image.save('LRPatch_sample.png', LRPatch)
	for i=2, B.batchNum do -- concat to SRPatch and LRPatch
		SRPatch, LRPatch = crop_SR_LR_patches(imgs[i], B.res)
		B.SR = torch.cat(B.SR,  SRPatch:clone():view(1, 3, SRPatch:size(2), SRPatch:size(3)), 1)
		B.LR = torch.cat(B.LR, LRPatch:clone():view(1, 3, LRPatch:size(2), LRPatch:size(3)), 1)
		-- print(imgs[i]) -- debug
	end
 	B.SR = B.SR:cuda()
	B.LR = B.LR:cuda()
	-- return imgBatch
end

function prepImgs(datasetPath)
	local imgPaths = {}
	local imgNum = 0
	for file in paths.iterfiles(datasetPath) do
		imgNum = imgNum + 1
		imgPaths[imgNum] = file 
	end
	return imgPaths, imgNum
end

function prepImageNet(ImageNetPath)
	local imgPaths = {}
	local imgNum = 0
	for dir in paths.iterdirs(ImageNetPath) do
		local c = 1
		for file in paths.iterfiles(paths.concat(ImageNetPath, dir)) do
			if c > 50 then break end
			local imPath = paths.concat(ImageNetPath, dir, file)
			local img = image.load(imPath)
			if img:size(1) == 3 and img:size(2) > 288 and img:size(3) > 288 then  -- TODO global resolution 
				imgNum = imgNum + 1
				imgPaths[imgNum] = imPath
				c = c+1
				print(imgNum)
			end
		-- print(dir)
		end
	end
	return imgPaths, imgNum
end
