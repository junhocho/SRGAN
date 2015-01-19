local parser = {}
require 'nn'

local process_node = {
   ['nn.SpatialZeroPadding'] = function(node, img, sequence, layer)
      assert(node.pad_l == node.pad_r, 'padding should be same all over')
      assert(node.pad_t == node.pad_b, 'padding should be same all over')
      assert(node.pad_l == node.pad_b, 'padding should be same all over')
      if next(layer) ~= nil then
         -- add pending layer to sequence
         table.insert(sequence, layer)
         layer = {}
      end

      layer.conv = {
         p = node.pad_l
      }

      img = node:forward(img)
      return img, sequence, layer
   end,
   ['nn.SpatialConvolutionMM'] = function(node, img, sequence, layer)
      assert(node.kH == node.kW, 'conv kernel should be square')
      assert(node.dH == node.dW, 'conv stride should be equal')
      if ((next(layer) ~= nil) and ((not layer.conv) or layer.conv.k)) then
         -- add pending layer to sequence
         table.insert(sequence, layer)
         layer = {}
      end

      layer.conv = layer.conv or {}
      layer.conv.k = node.kH
      layer.conv.s = node.dH
      layer.conv.p = layer.conv.p or node.padding
      layer.output = node.nOutputPlane

      img = node:forward(img)
      return img, sequence, layer
   end,
   ['nn.SpatialConvolution'] = function(node, img, sequence, layer)
      assert(node.kH == node.kW, 'conv kernel should be square')
      assert(node.dH == node.dW, 'conv stride should be equal')
      if ((next(layer) ~= nil) and ((not layer.conv) or layer.conv.k)) then
         -- add pending layer to sequence
         table.insert(sequence, layer)
         layer = {}
      end

      layer.conv = layer.conv or {}
      layer.conv.k = node.kH
      layer.conv.s = node.dH
      layer.conv.p = layer.conv.p or node.padding
      layer.output = node.nOutputPlane

      img = node:forward(img)
      return img, sequence, layer
   end,
   ['nn.ReLU'] = function(node, img, sequence, layer)
      assert(not layer.nlmp, "shouldn't have two non-linears in same layer")

      layer.nlmp = 'ReLU'

      img = node:forward(img)
      return img, sequence, layer
   end,
   ['nn.Threshold'] = function(node, img, sequence, layer)
      assert(not layer.nlmp, "shouldn't have two non-linears in same layer")

      layer.nlmp = 'Threshold'

      img = node:forward(img)
      return img, sequence, layer
   end,
   ['nn.LogSoftMax'] = function(node, img, sequence, layer)
      assert(not layer.nlmp, "shouldn't have two non-linears in same layer")

      layer.nlmp = 'LogSoftMax'

      img = node:forward(img)
      return img, sequence, layer
   end,
   ['nn.SpatialMaxPooling'] = function(node, img, sequence, layer)
      assert(node.kH == node.kW, 'pooling area should be square')
      assert(node.dH == node.dW, 'pooling strides should be equal')

      if node.kH == node.dH then
         layer.pool = node.kH
      else
         layer.pool = {
            size   = node.kH,
            stride = node.dH,
         }
      end

      img = node:forward(img)
      return img, sequence, layer
   end,
   ['nn.Reshape'] = function(node, img, sequence, layer)
      assert(img:dim() == 3, 'reshape input should have 3 dimensions')
      assert(img:size(2) == img:size(3), 'reshape input maps should be square')
      if next(layer) ~= nil then
         -- add pending layer to sequence
         table.insert(sequence, layer)
         layer = {}
      end

      layer.transform = {
         name = 'Reshape',
         size = img:size(2),
      }

      img = node:forward(img)
      return img, sequence, layer
   end,
   ['nn.Linear'] = function(node, img, sequence, layer)
      if next(layer) ~= nil then
         -- add pending layer to sequence
         table.insert(sequence, layer)
         layer = {}
      end

      layer.linear = node.weight:size(1)

      img = node:forward(img)
      return img, sequence, layer
   end,
   ['nn.Dropout'] = function(node, img, sequence, layer)
      img = node:forward(img)
      return img, sequence, layer
   end,
}

function parser:network(net, img, pos, sequence, layer)
   pos = pos or 1
   layer = layer or {}
   sequence = sequence or {}

   local node = assert(net['modules'][pos], 'no node at position')
   local node_name = node.__typename
   local node_last = (#net['modules'] == pos)

   while 'nn.Sequential' == node_name do
      local sub_sequence, tmp = self:network(node, img)
      table.insert(sequence, sub_sequence)

      if node_last then
         return sequence, tmp
      end

      img = tmp
      pos = pos+1
      node = assert(net['modules'][pos], 'no node at position')
      node_name = node.__typename
      node_last = (#net['modules'] == pos)
   end

   local process_fun = process_node[node_name]
   if process_fun then
      img, sequence, layer = process_fun(node, img, sequence, layer)
   else
      print('WARNING network module ignored:', node_name)
   end

   if node_last then
      table.insert(sequence, layer)

      return sequence, img
   end

   return self:network(net, img, pos+1, sequence, layer)
end

return parser
