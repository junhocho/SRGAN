----------------------------------------------------------------------
-- ImageNet-like network
--
-- E. Culurciello, with portions from Clement Farabet
----------------------------------------------------------------------

require 'torch'
require 'image'
require 'nnx'
--require 'RespNorm'

torch.setnumthreads(8)
torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')

opt={}
----------------------------------------------------------------------
print '==> define parameters'

-- load classes:
--local classes = require 'classes'
classes = torch.Tensor(1000)

-- checks:
opt.receptive=224
--assert(opt.receptive==224 and opt.size==256, 'input size must be 256 and receptive field 224')
print('WARNING - removed receptive fields check!!!!')

-- hidden units & filter sizes:
local mapsizes = {[0] = opt.receptive}
local nfeatures 
local dropout_mult = 2
nfeatures= {[0]=3,48,128,192,192,128*2,4096,4096,1000}--#classes}
  
local filtsizes = {9,9,9,3,3,1,1,1}
local paddings = {4,4,2,2,2}
local strides = {1,1,1,1,1,1,1,1}
local poolsizes = {2,2,2,1,2}
local kersizes = poolsizes
if opt.overpooling then
   kersizes = {1,3,3,1,3}
end
opt.ncolumns = 1 
local ncolumns = opt.ncolumns

-- map sizes:
for i = 1,#nfeatures do
   if filtsizes[i] == 1 then
      mapsizes[i] = 1
   else
      mapsizes[i] = mapsizes[i-1] + paddings[i] - filtsizes[i]
      if strides[i] then
         mapsizes[i] = mapsizes[i] / strides[i]
      end
      mapsizes[i] = mapsizes[i] + 1
      if poolsizes[i] then
         mapsizes[i] = (mapsizes[i] - kersizes[i]) / poolsizes[i] + 1
      end
      mapsizes[i] = math.floor(mapsizes[i])
   end
end

-- nb of hidden units per layer:
local nunits = {}
for k,mapsize in pairs(mapsizes) do
   nunits[k] = mapsizes[k]^2 * nfeatures[k]
end

total_ops = 0 -- total network operations
for i = 0,#mapsizes do
   print(string.format(
      '==> model layer %02d  -  spatial extent: %03dx%03d  |  unique features: %04d  |  hidden units: %05d',
      i, mapsizes[i], mapsizes[i], nfeatures[i], nunits[i]
   ))
   if i < #mapsizes then 
      ops = 2 * nfeatures[i] * nfeatures[i+1] * (mapsizes[i]-filtsizes[i+1]+1)^2/strides[i+1] * filtsizes[i+1]^2
   end
   print('Operations in this layer: ', ops)
   total_ops = total_ops + ops
end

----------------------------------------------------------------------
print '==> construct model'

local model = nn.Sequential()

--model:add(nn.Transpose({1,4},{1,3},{1,2}))
---model:get(1).updateGradInput = function() end

-- split model into N columns
concat = nn.Concat(2)
local column1, column2 = nil, nil

for i = 1,ncolumns do 
   -- submodel
   local submodel = nn.Sequential()
   
   -- stage 1: conv+max
   submodel:add(nn.SpatialConvolutionMM(nfeatures[0], nfeatures[1], filtsizes[1], filtsizes[1], strides[1], strides[1], paddings[1]))
   submodel:add(nn.Threshold(0,0))
   submodel:add(nn.SpatialMaxPooling(kersizes[1],kersizes[1],poolsizes[1],poolsizes[1]))
   
   -- stage 2: conv+max
   submodel:add(nn.SpatialConvolutionMM(nfeatures[1], nfeatures[2], filtsizes[2], filtsizes[2], strides[2], strides[2], paddings[2]))
   submodel:add(nn.Threshold(0,0))
   submodel:add(nn.SpatialMaxPooling(kersizes[2],kersizes[2],poolsizes[2],poolsizes[2]))
   
   -- stage 3: conv
   submodel:add(nn.SpatialConvolutionMM(nfeatures[2], nfeatures[3], filtsizes[3], filtsizes[3], strides[3], strides[3], paddings[3]))
   submodel:add(nn.Threshold(0,0))
   submodel:add(nn.SpatialMaxPooling(kersizes[3],kersizes[3],poolsizes[3],poolsizes[3]))
   
   -- stage 4: conv
   -- submodel:add(nn.SpatialConvolutionMM(nfeatures[3], nfeatures[4], filtsizes[4], filtsizes[4], strides[4], strides[4], paddings[4]))
   -- submodel:add(nn.Threshold(0,0))
   -- submodel:add(nn.SpatialMaxPooling(kersizes[4],kersizes[4],poolsizes[4],poolsizes[4]))
   
   -- -- stage 5: conv
   -- submodel:add(nn.SpatialConvolutionMM(nfeatures[4], nfeatures[5], filtsizes[5], filtsizes[5], strides[5], strides[5], paddings[5]))
   -- submodel:add(nn.Threshold(0,0))
   -- submodel:add(nn.SpatialMaxPooling(kersizes[5],kersizes[5],poolsizes[5],poolsizes[5]))
   
   -- reshape
   --submodel:add(nn.Transpose({4,1},{4,2},{4,3}))
   submodel:add(nn.Reshape(nunits[5]))

   -- add submodel
   concat:add(submodel)
   
   -- optimizations
   submodel:get(1).updateGradInput = function(input, gradOutput) return gradOutput end

   if i == 1 then
      column1 = submodel
   else
      column2 = submodel
   end
end
model:add(concat)

-- -- stage 6: linear
-- local dropout2 = nn.Dropout(opt.dropout) 
-- local start_counter 
-- model:add(nn.Linear(nunits[5]*ncolumns, nunits[6]))
-- model:add(nn.Threshold(0,0))

-- -- stage 7: linear
-- local dropout1 = nn.Dropout(opt.dropout) 
-- model:add(nn.Linear(nunits[6], nunits[7]))

-- -- if maxout flag is on then replace RELU with maxout
-- if opt.maxout then
--    -- add maxout layer
--    model:add(nn.Reshape(nunits[7]/dropout_mult, dropout_mult))
--    model:add(nn.Max(3))
--    model:add(nn.Reshape(nunits[7]/dropout_mult))
--    -- stage 8: linear (classifier)
--    model:add(nn.Linear(nunits[7]/dropout_mult, nunits[8]))
-- else
--    model:add(nn.Threshold(0,0))
--    if opt.dropouts == 0 then
--       start_counter = #model.modules
--    end
--    dropout0 = nn.Dropout(opt.dropout)
--    model:add(dropout0)
--    -- stage 8: linear (classifier)
--    model:add(nn.Linear(nunits[7], nunits[8]))
-- end


batch_size = 1 -- for MM tests
testbatch = torch.Tensor(batch_size,3,224,224)

-- speed test:
sys.tic()
result = model:forward(testbatch)
t=sys.toc()

print('Time taken per batch [ms]: ', t*1000)
print('Time taken per image [ms]: ', t*1000/batch_size)
print('Network operations per batch: [G]', total_ops*batch_size/1e9)
print('Network operations per image: [G]', total_ops/1e9)
print('G flops/s: ', total_ops*batch_size/t/1e9)



