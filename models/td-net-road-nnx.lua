return {
   name = 'Net Road1',
   channel = 3,
   eye = 56,
   mknet = function(self)
      local model = nn.Sequential()

      -- convolution layer
      model:add(nn.SpatialConvolutionMM(3, 32, 9, 9, 1, 1))
      model:add(nn.SpatialMaxPooling(4,4,4,4))
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(32, 64, 5, 5, 1, 1))
      model:add(nn.SpatialMaxPooling(4,4,4,4))
      model:add(nn.ReLU())

      -- fully connected layer
      model:add(nn.View(64*2*2))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(64*2*2, 128))
      model:add(nn.ReLU())
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(128, 17))
      model:add(nn.LogSoftMax())

      return model
   end
}
