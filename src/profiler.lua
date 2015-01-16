local profiler = {}
local xlua = assert(require('xlua'))
local parser = assert(require('./parser'))


function profiler:calc_ops(def, input, map, pos, ops)
   pos = pos or 1
   ops = ops or {conv = 0, pool = 0, mlp = 0, neurons = 0}

   local layer = assert(def[pos], 'no layer at position')
   local output = layer.output or input

   while 0 ~= #layer do
      ops, input, map = self:calc_ops(layer, input, map, 1, ops)

      if (#def == pos) then
         return ops, input, map
      end

      pos = pos+1
      layer = assert(def[pos], 'no layer at position')
      output = layer.output or input
   end

   -- calculate ops for conv
   if layer.conv then
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
      local ops_bias = output * output_map

      ops.conv = ops.conv + (input * output * output_map * ops_kernel + ops_bias)

      if layer.nlmp then
         if ('ReLU' == layer.nlmp)
               or ('Threshold' == layer.nlmp)
               or ('LogSoftMax' == layer.nlmp) then

            local ops_nlmp = output * output_map
            ops.conv = ops.conv + ops_nlmp
         else
            error('do not know this non-linear mapper module', layer.nlmp)
         end
      end

      if not layer.pool then
         -- calculate neurons for conv without pooling
         ops.neurons = ops.neurons + (output * output_map)
      end
   end

   -- calculate ops for pool
   if layer.pool then
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

   -- calculate ops for linear
   if layer.linear then
      output = layer.linear
      local ops_weights = (2 * input * output)
      local ops_bias = output
      ops.mlp = ops.mlp + ops_weights + ops_bias

      if layer.nlmp then
         if ('ReLU' == layer.nlmp)
               or ('Threshold' == layer.nlmp)
               or ('LogSoftMax' == layer.nlmp) then

            local ops_nlmp = output
            ops.mlp = ops.mlp + ops_nlmp
         else
            error('do not know this non-linear mapper module', layer.nlmp)
         end
      end

      -- calculate neurons for linear
      ops.neurons = ops.neurons + output
   end

   -- calculate new output
   if layer.reshape then
      output = (layer.reshape * layer.reshape * input)
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

function profiler:time(model, net, iterations,  map)
   local img = torch.FloatTensor(model.channel, map.height, map.width)
   local time = 0
   local time_conv = 0
   local time_mlp = 0

   if 2 ~= #model.def then

      time = calc_time_cpu(net, img, iterations)

   else
      local tmp = false

      time_conv, tmp = calc_time_cpu(net.modules[1], img, iterations)
      time_mlp       = calc_time_cpu(net.modules[2], tmp, iterations)

      time = time_conv + time_mlp
   end

   return time, time_conv, time_mlp
end

return profiler
