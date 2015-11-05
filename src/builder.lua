local builder = {}

require 'nn'

local eye_ops = {}
local eye_output = 0
local function eye_calc(pos, output)
   local ops = eye_ops[pos]
   local input = output

   if 'conv' == ops.name then
      input = (output * ops.s) + (ops.k - (2 * ops.p) - 1)
   end

   if 'pool' == ops.name then
      input = ((output - 1) * ops.stride) + ops.size
   end

   if 1 == pos then
      return input
   end

   return eye_calc(pos-1, input)
end

local function generate_conv(layer, net, input)
   local output = assert(layer.output)
   local conv_k = assert(layer.conv.k, 'conv def needs kernel def at min')
   local conv_p = layer.conv.p or 0
   if ('boolean' == type(conv_p)) and conv_p then
      -- auto generate a padding value to give same output size as input
      conv_p = math.floor((conv_k-1)/2)
   end
   local conv_s = layer.conv.s or 1

   print('net:add(nn.SpatialConvolutionMM('
      ..input..',  '..output..' , '
      ..conv_k..', '..conv_k..' , '
      ..conv_s..', '..conv_s..' , '
      ..conv_p..', '..conv_p..'))')

   net:add( nn.SpatialConvolutionMM
      ( input,  output
      , conv_k, conv_k
      , conv_s, conv_s
      , conv_p, conv_p
      )
   )

   -- save eye calculation operation
   table.insert(eye_ops, {
      name = 'conv',
      k = conv_k,
      p = conv_p,
      s = conv_s,
   })

   if layer.pool then
      local size = layer.pool
      local stride = layer.pool
      if 'table' == type(layer.pool) then
         size   = layer.pool.size
         stride = layer.pool.stride
      end

      print('net:add(nn.SpatialMaxPooling('
         ..size..', '..size..', '..stride..', '..stride..'))')

      net:add( nn.SpatialMaxPooling (
           size,   size
         , stride, stride
         )
      )

      -- save eye calculation operation
      table.insert(eye_ops, {
         name   = 'pool',
         size   = size,
         stride = stride,
      })
   end

   if layer.nlmp then
      if 'ReLU' == layer.nlmp then
         print('net:add(nn.ReLU())')
         net:add(nn.ReLU())
      elseif 'Threshold' == layer.nlmp then
         print('net:add(nn.Threshold())')
         net:add(nn.Threshold())
      elseif 'SoftMax' == layer.nlmp then
         print('net:add(nn.SoftMax())')
         net:add(nn.SoftMax())
      elseif 'LogSoftMax' == layer.nlmp then
         print('net:add(nn.LogSoftMax())')
         net:add(nn.LogSoftMax())
      else
         error('do not know this non-linear mapper module '..layer.nlmp)
      end
   end

   return net, output
end

local function generate_linear(layer, net, input)
   local output = assert(layer.linear)
   print('net:add(nn.Linear('..input..', '..output..'))')
   net:add(nn.Linear(input, output))

   if layer.nlmp then
      if 'ReLU' == layer.nlmp then
         print('net:add(nn.ReLU())')
         net:add(nn.ReLU())
      elseif 'Threshold' == layer.nlmp then
         print('net:add(nn.Threshold())')
         net:add(nn.Threshold())
      elseif 'SoftMax' == layer.nlmp then
         print('net:add(nn.SoftMax())')
         net:add(nn.SoftMax())
      elseif 'LogSoftMax' == layer.nlmp then
         print('net:add(nn.LogSoftMax())')
         net:add(nn.LogSoftMax())
      else
         error('do not know this non-linear mapper module '..layer.nlmp)
      end
   end

   return net, output
end

local function generate_transform(layer, net, input)
   local transform = layer.transform
   local output = 0

   if 'Reshape' == transform.name then
      output = (transform.size * transform.size * input)
      print('net:add(nn.Reshape('..output..'))')
      net:add(nn.Reshape(output))

      -- save transform size for eye calculation
      eye_output = transform.size
   else
      error('do not know this transform module '..transform.name)
   end

   return net, output
end

local function parse_cpu(def, pos, net, input)
   local layer = def[pos]
   local output = layer.output or input

   while 0 ~= #layer do
      local nest_net, nest_output = parse_cpu(layer, 1, nn.Sequential(), input)
      net:add(nest_net)

      if (#def == pos) then
         return net, nest_output
      end

      input = nest_output
      pos = pos+1
      layer = def[pos]
      output = layer.output or input
   end

   if layer.conv then
      net, output = generate_conv(layer, net, input)
   elseif layer.linear then
      net, output = generate_linear(layer, net, input)
   elseif layer.transform then
      net, output = generate_transform(layer, net, input)
   end

   if (#def == pos) then
      return net, output
   end

   return parse_cpu(def, pos+1, net, output)
end

function builder:cpu(model)
   eye_ops = {}
   eye_output = 0
   local eye

   local net = parse_cpu(model.def, 1, nn.Sequential(), model.channel)
   if eye_output ~= 0 then
      eye = eye_calc(#eye_ops, eye_output)
   end

   print('\n')
   return net, eye
end

return builder
