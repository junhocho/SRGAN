local op_count
local op_used
local multiply_adds = opt.MACs

function count_ops(network, input)
    op_count = 0
    op_used = {}
    network:apply(intercept_updateOutput)
    inputImg = input[1]
    network:forward(input)
    network:apply(restore_updateOutput)
    return op_count, op_used
end

-- Intercept updateOutput. At each call increment op_count appropriately.
function intercept_updateOutput(module)
    module.updateOutput_original = module.updateOutput
    module.updateOutput = function(self, input)
        compute_ops(module, input)
        return module:updateOutput_original(input)
    end
end

-- Restore original network behaviour
function restore_updateOutput(module)
    assert(module.updateOutput_original,
        "restore_updateOutput should be called after intercept_updateOutput!")
    module.updateOutput = module.updateOutput_original
    module.updateOutput_original = nil
end

-- Compute #flops that specified module needs to process an input.
-- module_handlers table is at the bottom of this file
function compute_ops(module, input)
    module_name = torch.type(module)
    handler = module_handlers[module_name]
    assert(handler, string.format("No handler for module %s!", module_name))
    local ops = handler(module, input)
    op_count = op_count + ops
    local maps = 0
    local neurons = 0
    if torch.type(module) ~= 'nn.JoinTable' and torch.type(module) ~= 'nn.CAddTable' then
       for i = 1, input:dim() do
          if i == 1 then
             maps = input:size(1)
             neurons = input:size(1)
          else
             maps = maps .. ' x ' .. input:size(i)
             neurons = neurons * input:size(i)
          end
       end
    else
    end
    table.insert(op_used, {name = torch.type(module),
                           ops = ops,
                           maps = maps,
                           neurons = neurons})
end

--------------------------------------------------------------------------------
------------------------------- Module handlers --------------------------------
--------------------------------------------------------------------------------

local function ops_nothing(module, input)
    return 0
end

local function ops_linear(module, input)
    local batch_size = input:dim() == 2 and input:size(1) or 1
    local weight_ops = module.weight:nElement() * (multiply_adds and 1 or 2)
    local bias_ops = module.bias:nElement()
    local ops_per_sample = weight_ops + bias_ops
    return batch_size * ops_per_sample
end

local function ops_logsoftmax(module, input)
    local batch_size = input:dim() == 2 and input:size(1) or 1
    local input_dim = input:dim() == 2 and input:size(2) or input:size(1)
    local expminusapprox_ops = 1 -- around 8 in Torch
    -- +2 for accumulation and substraction in two loops
    local ops_per_elem = expminusapprox_ops + 1 + 1
    local ops_per_sample = input_dim * ops_per_elem
    return batch_size * ops_per_sample
end

-- WARNING: an oversimplified version
local function ops_nonlinearity(module, input)
    return input:nElement()
end

local function ops_convolution(module, input)
    assert(input:dim() == 4, "ops_convolution supports only batched inputs!")
    assert(input:size(2) == module.nInputPlane, "number of input planes doesn't match!")
    local batch_size = input:size(1)
    local input_planes = input:size(2)
    local input_height = input:size(3)
    local input_width = input:size(4)

    -- ops per output element
    local kernel_ops = module.kH * module.kW * input_planes * (multiply_adds and 1 or 2)
    local bias_ops = 1
    local ops_per_element = kernel_ops + bias_ops

    local output_width = math.floor((input_width + 2 * module.padW - module.kW) / module.dW + 1)
    local output_height = math.floor((input_height + 2 * module.padH - module.kH) / module.dH + 1)

    return batch_size * module.nOutputPlane * output_width * output_height * ops_per_element
end

local function ops_fullconvolution(module, input)
    assert(input:dim() == 4, "ops_fullconvolution supports only batched inputs!")
    assert(input:size(2) == module.nInputPlane, "number of input planes doesn't match!")
    local batch_size = input:size(1)
    local input_planes = input:size(2)
    local input_height = input:size(3)
    local input_width = input:size(4)

    -- ops per input element
    local single_kernel_ops = module.kH * module.kW * input_planes * (multiply_adds and 1 or 2)
    local sample_kernel_ops = input_planes * input_width * input_height * single_kernel_ops

    local output_width = (input_width - 1) * module.dW - 2 * module.padW + module.kW + module.adjW
    local output_height = (input_height - 1) * module.dH - 2 * module.padW + module.kH + module.adjH
    local bias_ops = output_width * output_height * module.nOutputPlane

    return batch_size * (sample_kernel_ops + bias_ops)
end

local function ops_dilatedconvolution(module, input)
    assert(input:dim() == 4, "ops_convolution supports only batched inputs!")
    assert(input:size(2) == module.nInputPlane, "number of input planes doesn't match!")
    local batch_size = input:size(1)
    local input_planes = input:size(2)
    local input_height = input:size(3)
    local input_width = input:size(4)
    local dilW = module.dilationW
    local dilH = module.dilationH

    -- ops per output element
    local kernel_ops = module.kH * module.kW * input_planes * (multiply_adds and 1 or 2)
    local bias_ops = 1
    local ops_per_element = kernel_ops + bias_ops

    local output_width = math.floor(
                         (input_width + 2 * module.padW - dilW * (module.kW - 1) + 1)
                         /module.dW + 1)
    local output_height = math.floor(
                          (input_height + 2 * module.padH - dilH * (module.kH - 1) + 1)
                          /module.dH + 1)

    return batch_size * module.nOutputPlane * output_width * output_height * ops_per_element
end

local function ops_pooling(module, input)
    assert(input:dim() == 4, "ops_averagepooling supports only batched inputs!")
    local batch_size = input:size(1)
    local input_planes = input:size(2)
    local input_height = input:size(3)
    local input_width = input:size(4)

    local kernel_ops = module.kH * module.kW

    local output_width = math.floor((input_width + 2 * module.padW - module.kW) / module.dW + 1)
    local output_height = math.floor((input_height + 2 * module.padH - module.kH) / module.dH + 1)

    return batch_size * input_planes * output_width * output_height * kernel_ops
end

local function ops_unpooling(module, input)
    assert(input:dim() == 4, "ops_unpooling supports only batched inputs!")
    local batch_size = input:size(1)
    local input_planes = input:size(2)
    local input_height = input:size(3)
    local input_width = input:size(4)


    local output_width = (input_width - 1) * module.pooling.dW - (2 * module.pooling.padW - module.pooling.kW)
    local output_height = (input_height - 1) * module.pooling.dH - (2 * module.pooling.padH - module.pooling.kH)

    return batch_size * input_planes * output_width * output_height
end

local function ops_caddtable(module, input)
    assert(torch.type(input) == 'table', "ops_caddtable input should be a table!")
    return input[1]:nElement() * #input
end

local function ops_batchnorm(module, input)
    return input:nElement() * (multiply_adds and 1 or 2)
end

local function ops_sum(module, input)
   assert(not module.nInputDims, 'nInputDims mode of nn.Sum not supported.')
   local ops = 1
   for d = 1, input:dim() do
      local s = input:size(d)
      ops = d ~= module.dimension and ops * s or ops * (s - 1)
   end
   return ops
end

module_handlers = {
    -- Containers
    ['nn.Sequential'] = ops_nothing,
    ['nn.Parallel'] = ops_nothing,
    ['nn.Concat'] = ops_nothing,
    ['nn.gModule'] = ops_nothing,
    ['nn.Identity'] = ops_nothing,
    ['nn.DataParallelTable'] = ops_nothing,
    ['nn.Contiguous'] = ops_nothing,
    ['nn.ConcatTable'] = ops_nothing,
    ['nn.JoinTable'] = ops_nothing,
    ['nn.Padding'] = ops_nothing,

    -- Nonlinearities
    ['nn.ReLU'] = ops_nonlinearity,
    ['nn.PReLU'] = ops_nonlinearity,
    ['nn.LogSoftMax'] = ops_logsoftmax,
    ['nn.SoftMax'] = ops_logsoftmax,   --TODO Update it with correct ops calculator
    ['cudnn.ReLU'] = ops_nonlinearity,
    ['cudnn.PReLU'] = ops_nonlinearity,

    -- Basic modules
    ['nn.Linear'] = ops_linear,
    ['nn.Sum'] = ops_sum,

    -- Spatial Modules
    ['nn.SpatialConvolution'] = ops_convolution,
    ['nn.SpatialConvolutionMM'] = ops_convolution,
    ['nn.SpatialDilatedConvolution'] = ops_dilatedconvolution,
    ['nn.SpatialFullConvolution'] = ops_fullconvolution,
    ['nn.SpatialMaxPooling'] = ops_pooling,
    ['nn.SpatialAveragePooling'] = ops_pooling,
    ['nn.SpatialMaxUnpooling'] = ops_unpooling,
    ['nn.SpatialZeroPadding'] = ops_nothing,
    ['nn.SpatialBatchNormalization'] = ops_nothing, -- Can be squashed

    ['cudnn.SpatialConvolution'] = ops_convolution,
    ['cudnn.SpatialConvolutionMM'] = ops_convolution,
    ['cudnn.SpatialDilatedConvolution'] = ops_dilatedconvolution,
    ['cudnn.SpatialMaxPooling'] = ops_pooling,
    ['cudnn.SpatialAveragePooling'] = ops_pooling,
    ['cudnn.SpatialBatchNormalization'] = ops_nothing, -- Can be squashed

    -- Table modules
    ['nn.CAddTable'] = ops_caddtable,

    -- Various modules
    ['nn.View'] = ops_nothing,
    ['nn.Reshape'] = ops_nothing,
    ['nn.Dropout'] = ops_nothing, -- Is turned off in inference
    ['nn.SpatialDropout'] = ops_nothing, -- Is turned off in inference
    ['nn.Concat'] = ops_nothing,
}
