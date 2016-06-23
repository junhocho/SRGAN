local profileTime = {}
local xlua = assert(require('xlua'))

-- convert network for cudnn
local function convertCUDNN(net, create, index)
   create = create or nn.Sequential()
   index = index or 1
   local module = net.modules[index]
   local module_name = module.__typename
   local module_last = index == #net.modules

   while module_name == 'nn.Sequential' do
      -- branching net module
      if module_last then
         return convertCUDNN(module, create)
      end

      create = convertCUDNN(module, create)
      index = index + 1
      module = net.modules[index]
      module_name = module.__typename
      module_last = index == #net.modules
   end

   if module_name == 'nn.SpatialConvolutionMM' then

      local tmp_module = cudnn.SpatialConvolution
         ( module.nInputPlane
         , module.nOutputPlane
         , module.kW, module.kH
         , module.dW, module.dH
         )

      tmp_module.weight = module.weight:float():reshape
         ( module.nOutputPlane
         , module.nInputPlane
         , module.kW, module.kH
         )

      tmp_module.bias = module.bias:float()

      create:add(tmp_module)
   elseif module_name == 'nn.SpatialMaxPooling' then

      local tmp_module = cudnn.SpatialMaxPooling
         ( module.kW, module.kH
         , module.dW, module.dH)

      create:add(tmp_module)
   elseif module_name == 'nn.ReLU' then

      local tmp_module = cudnn.ReLU()

      create:add(tmp_module)
   elseif module_name == 'nn.Dropout' then

      module.train = false

      create:add(module)
   elseif module_name == 'nn.SoftMax' then
      -- do nothing
   elseif module_name == 'nn.LogSoftMax' then
      -- do nothing
   else
      create:add(module)
   end


   if module_last then
      -- construction of spatial net complete
      return create
   end

   return convertCUDNN(net, create, (index+1))
end

local function calc_time_nnx(net, img, iterations)
   local nn_X = assert(require 'nn_X')

   -- parse network for nnx
   local dst, src = nn_X:compile(net, img:type(), img:size())

   local timer = torch.Timer()
   local timing = torch.FloatTensor(iterations)
   local t = 0

   -- iterations plus one to prime the jit
   for i=1, (iterations+1) do
      xlua.progress(i, iterations)

      timer:reset()

      nn_X:forward(img)

      t = timer:time().real
      timing[(i%iterations)+1] = t
   end

   local scale = timing:mean()*1024*1024
   local bandwidth_total = BANDWIDTH.src+BANDWIDTH.dst
   print(string.format('   Bandwidth to memory   [MByte/sec]: %d', BANDWIDTH.dst/scale))
   print(string.format('   Bandwidth from memory [MByte/sec]: %d', BANDWIDTH.src/scale))
   print(string.format('   Bandwidth total       [MByte/sec]: %d', bandwidth_total/scale))

   nn_X:close()

   return timing:mean(), tmp
end

local function calc_time_cuda(net, img, iterations)
   collectgarbage()
   if not sys.execute('uname -a'):find('tegra') then
      assert(require("cunn"))
   else
      assert(require("cudnn"))

      net = convertCUDNN(net)
      print('network has been converted to CUDNN:')
      print(net)
      net = net:cuda()

      cutorch.setDevice(1)
--      print('==> using GPU #' .. cutorch.getDevice())
--      print(cutorch.getDeviceProperties(1))
      cutorch.synchronize()
   end

   local tmp = false
   local timer = torch.Timer()
   local timing = torch.FloatTensor(iterations)
   local t = 0
   net:cuda()

   -- iterations plus one to prime the jit
   for i=1, (iterations+1) do
      xlua.progress(i, iterations)

      timer:reset()

      local img_cuda = img:cuda()
      tmp = net:forward(img_cuda)
      cutorch.synchronize()
      tmp:float()

      t = timer:time().real
      timing[(i%iterations)+1] = t
   end

   return timing:mean(), tmp
end

local function calc_time_cpu(net, img, iterations)
   local tmp = false
   local timer = torch.Timer()
   local timing = torch.FloatTensor(iterations)
   local t = 0

   -- iterations plus one to prime the jit
   for i=1, (iterations+1) do
      xlua.progress(i, iterations)

      timer:reset()

      tmp = net:forward(img)

      t = timer:time().real
      timing[(i%iterations)+1] = t
   end

   return timing:mean(), tmp
end

function profileTime:time(net, img, iterations, platform)
   iterations = iterations or 10
   local time = { total = 0, conv = 0, mlp = 0, }

   if 'cuda' == platform then

      time.total = calc_time_cuda(net, img, iterations)

   elseif 'nnx' == platform then

      time.total = calc_time_nnx(net, img, iterations)

   elseif 2 ~= #net['modules'] then

      time.total = calc_time_cpu(net, img, iterations)

   else
      local tmp = false

      time.conv, tmp = calc_time_cpu(net.modules[1], img, iterations)
      time.mlp       = calc_time_cpu(net.modules[2], tmp, iterations)

      time.total = time.conv + time.mlp
   end

   return time
end

return profileTime
