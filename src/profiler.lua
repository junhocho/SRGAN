local profiler = {}
local xlua = assert(require('xlua'))
local parser = assert(require('./parser'))

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


local function calc_conv(layer, input, map, ops)
   local output = assert(layer.output)
   local k = layer.conv.k
   local p = layer.conv.p or 0
   if ('boolean' == type(p)) and p then
      -- auto generate a padding value to give same output size as input
      p = math.floor((k-1)/2)
   end
   local s = layer.conv.s or 1

   map.width  = math.floor((map.width  + (2 * p) - k + 1) / s)
   map.height = math.floor((map.height + (2 * p) - k + 1) / s)
   local output_map = map.width * map.height
   local ops_kernel = (2 * k^2) -- kernel + comb

   if not layer.columns then
      local ops_bias = output * output_map

      ops.conv = ops.conv + (input * output * output_map * ops_kernel + ops_bias)
   else
      assert('number' == type(layer.columns))
      local input = input / layer.columns
      local output = output / layer.columns
      local ops_bias = output * output_map
      local ops_column = (input * output * output_map * ops_kernel + ops_bias)

      ops.conv = ops.conv + (ops_column * layer.columns)
   end

   if layer.nlmp then
      if not ( ('ReLU' == layer.nlmp)
            or ('Threshold' == layer.nlmp)
            or ('SoftMax' == layer.nlmp)
            or ('LogSoftMax' == layer.nlmp)) then

         error('do not know this non-linear mapper module '..layer.nlmp)
      else
         local ops_nlmp = output * output_map
         ops.conv = ops.conv + ops_nlmp
      end
   end

   if not layer.pool then
      -- calculate neurons for conv without pooling
      ops.neurons = ops.neurons + (output * output_map)
   else
      local size = layer.pool
      local stride = layer.pool
      if 'table' == type(layer.pool) then
         size   = layer.pool.size
         stride = layer.pool.stride
      end

      map.width  = math.floor((map.width  - size) / stride) + 1
      map.height = math.floor((map.height - size) / stride) + 1
      local output_map = map.width * map.height

      ops.pool = ops.pool + (size^2 * output_map)

      -- calculate neurons for conv with pooling
      ops.neurons = ops.neurons + (output * output_map)
   end

   return output
end

local function calc_linear(layer, input, ops)
   local output = assert(layer.linear)
   local ops_weights = (2 * input * output)
   local ops_bias = output
   ops.mlp = ops.mlp + ops_weights + ops_bias

   if layer.nlmp then
      if not ( ('ReLU' == layer.nlmp)
            or ('Threshold' == layer.nlmp)
            or ('SoftMax' == layer.nlmp)
            or ('LogSoftMax' == layer.nlmp)) then

         error('do not know this non-linear mapper module '..layer.nlmp)
      else
         local ops_nlmp = output
         ops.mlp = ops.mlp + ops_nlmp
      end
   end

   -- calculate neurons for linear
   ops.neurons = ops.neurons + output

   return output
end

local function calc_transform(layer, input)
   local transform = layer.transform
   local output = 0
   local height = 0
   local width = 0
   if type(transform.size) == 'table' then
      height = transform.size.height
      width = transform.size.width
   else
      height = transform.size
      width = transform.size
   end

   if 'Reshape' == transform.name then
      -- calculate new output after reshape
      output = (width * height * input)
   elseif 'View' == transform.name then
      -- calculate new output after reshape
      output = (width * height * input)
   else
      error('do not know this transform module '..transform.name)
   end

   return output
end

function profiler:calc_ops(def, input, map, pos, ops)
   pos = pos or 1
   ops = ops or {conv = 0, pool = 0, mlp = 0, neurons = 0}
   local layer = assert(def[pos], 'no layer at position')

   while 0 ~= #layer do
      ops, input, map = self:calc_ops(layer, input, map, 1, ops)

      if (#def == pos) then
         return ops, input, map
      end

      pos = pos+1
      layer = assert(def[pos], 'no layer at position')
   end

   if layer.conv then
      -- calculate ops for conv
      output = calc_conv(layer, input, map, ops)
   elseif layer.linear then
      -- calculate ops for linear
      output = calc_linear(layer, input, ops)
   elseif layer.transform then
      -- calculate ops for transfrom
      output = calc_transform(layer, input)
   else
      error('unknown layer type')
   end

   if (#def == pos) then
      return ops, output, map
   end

   return self:calc_ops(def, output, map, pos+1, ops)
end

function profiler:ops(net, img)
   assert(img:dim() == 3, 'ops image needs to have 3 dimensions')
   local channel = img:size(1)
   local map = { width  = img:size(3), height = img:size(2), }
   local def = parser:network(net, img)

   return self:calc_ops(def, channel, map)
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

function profiler:time(net, img, iterations, platform)
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

return profiler
