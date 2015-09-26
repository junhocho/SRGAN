-- make network spatial
local function net_spatial(net, src, create, index)
   assert(net, '<net_spatial> no network to convert')
   assert(src, '<net_spatial> needs an image')

   create = create or nn.Sequential()
   index = index or 1
   local module = net.modules[index]
   local module_name = module.__typename
   local module_last = index == #net.modules

   while module_name == 'nn.Sequential' do
      -- branching net module
      if module_last then
         return net_spatial(module, src, create)
      end

      create, src = net_spatial(module, src, create)
      index = index + 1
      module = net.modules[index]
      module_name = module.__typename
      module_last = index == #net.modules
   end

   if module_name == 'nn.Linear' then

      if (#src:size() == 3) then
         tmp_module = nn.SpatialConvolutionMM
            ( src:size(1)
            , module.weight:size(1)
            , src:size(2)
            , src:size(3)
            )
      else
         tmp_module = nn.SpatialConvolutionMM
            ( module.weight[1]
            , module.weight[2]
            , 1
            , 1
            )
      end

      tmp_module.weight:copy(module.weight):resize(tmp_module.weight:size())
      tmp_module.bias:copy(module.bias)
      create:add(tmp_module)
      src = tmp_module:forward(src)
   elseif module_name == 'nn.Dropout' then
      -- do nothing
   elseif module_name == 'nn.SoftMax' then
      -- do nothing
   elseif module_name == 'nn.LogSoftMax' then
      -- do nothing
   elseif module_name == 'nn.Reshape' then
      -- do nothing
   elseif module_name == 'nn.View' then
      -- do nothing
   else
      create:add(module)
      src = module:forward(src)
   end


   if module_last then
      -- construction of spatial net complete
      return create, src
   end

   return net_spatial(net, src, create, (index+1))
end


-- classifier only 1st layer is SptialConvMM, other layers remain linear
local function net_spatial_mlp(net, src, create, index)
   assert(net, '<net_spatial_mlp> no network to convert')
   assert(src, '<net_spatial_mlp> needs an image')

   create = create or nn.Sequential()
   index = index or 1
   local module = net.modules[index]
   local module_name = module.__typename
   local module_last = index == #net.modules

   while module_name == 'nn.Sequential' do
      -- branching net module
      if module_last then
         return net_spatial_mlp(module, src, create)
      end

      create, src = net_spatial_mlp(module, src, create)
      index = index + 1
      module = net.modules[index]
      module_name = module.__typename
      module_last = index == #net.modules
   end


   if (module_name == 'nn.Reshape') or (module_name == 'nn.View') then
      index = index + 1
      local next_module = net.modules[index]
      while next_module.__typename ~= 'nn.Linear' do
         -- find first linear
         index = index + 1
         next_module = net.modules[index]
      end

      if #src:size() ~= 3 then
         error('ERROR <spatial> MLP src dimension must be 3')
      end

      local kernel = 0
      if module_name == 'nn.Reshape' then
         kernel = math.sqrt(module.nelement/src:size(1))
      else
         kernel = math.sqrt(module.numElements/src:size(1))
      end

      -- additional convolution layer for
      local conv_module = nn.SpatialConvolutionMM
            ( src:size(1)
            , next_module.weight:size(1)
            , kernel
            , kernel
            )

      conv_module.weight:copy(next_module.weight):resize(conv_module.weight:size())
      conv_module.bias:copy(next_module.bias)
      create:add(conv_module)
      src = conv_module:forward(src)

      -- absorb non-linear
      local peak_module = net.modules[index + 1]
      if (peak_module.__typename == 'nn.ReLU') or
         (peak_module.__typename == 'nn.Threshold') then

         index = index + 1
         create:add(peak_module)
      end

      -- transform module
      local transform_module = nn.Reshape(src:size(1))
      create:add(transform_module)
      src = transform_module:forward(src)

      module_last = index == #net.modules

   elseif module_name == 'nn.Dropout' then
      -- do nothing
   elseif module_name == 'nn.SoftMax' then
      -- do nothing
   elseif module_name == 'nn.LogSoftMax' then
      -- do nothing
   else
      create:add(module)
      src = module:forward(src)
   end


   if module_last then
      -- construction of spatial net complete
      return create, src
   end

   return net_spatial_mlp(net, src, create, (index+1))
end


return {

   -- make network spatial
   net_spatial = net_spatial,

   -- classifier only 1st layer is SptialConvMM, other layers remain linear
   net_spatial_mlp = net_spatial_mlp,

}
