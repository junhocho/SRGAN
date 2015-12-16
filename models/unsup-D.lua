return {
   name = 'Net Unsup D',
   channel = 3,
   eye = 64,
   mknet = function(self)
      local model = nn.Sequential()
      local nc = 3
      local ndf = 32

      -- convolution layer
      -- input is nc x 64 x 64
      model:add(nn.SpatialConvolution(nc, ndf, 4, 4, 2, 2, 1, 1))
      model:add(nn.ReLU()) 
      -- state size: ndf x 32 x 32
      model:add(nn.SpatialConvolution(ndf, ndf*2, 4, 4, 2, 2, 1, 1))
      model:add(nn.ReLU())
      -- state size: 2*ndf x 16 x 16
      model:add(nn.SpatialConvolution(ndf*2, ndf*4, 4, 4, 2, 2, 1, 1))
      model:add(nn.ReLU())
      -- state size: 4*ndf x 8 x 8
      model:add(nn.SpatialConvolution(ndf*4, ndf*8, 4, 4, 2, 2, 1, 1))
      model:add(nn.ReLU())
      -- state size: 8*ndf x 4 x 4
      model:add(nn.SpatialConvolution(ndf*8, 1, 4, 4)) -- state size: 8*ndf x 4 x 4
      model:add(nn.Sigmoid())
      -- state size: 1 x 1 x 1

      -- fully connected layer
      -- model:add(nn.View(64*2*2))
      -- model:add(nn.Dropout(0.5))
      -- model:add(nn.Linear(64*2*2, 128))
      -- model:add(nn.ReLU())
      -- model:add(nn.Dropout(0.5))
      -- model:add(nn.Linear(128, 17))
      -- model:add(nn.LogSoftMax())

      return model
   end
}
