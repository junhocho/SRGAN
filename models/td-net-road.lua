return {
   name = 'Net Road1',
   channel = 3,
   eye = 64, -- or 96 if desired
   mknet = function(self)
      local model = nn.Sequential()

      -- convolution layer
      model:add(nn.SpatialConvolutionMM(3, 32, 9, 9, 4, 4))
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(32, 64, 5, 5, 1, 1))
      model:add(nn.SpatialMaxPooling(2,2,2,2))
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1))
      -- model:add(nn.SpatialMaxPooling(2,2,2,2)) -- uncomment this for eye size of 96
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1))
      model:add(nn.ReLU())

      -- fully connected layer
      model:add(nn.View(64))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(64, 128))
      model:add(nn.ReLU())
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(128, 17))
      model:add(nn.LogSoftMax())

      return model
   end
}
