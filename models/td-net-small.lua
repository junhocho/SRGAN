return {
   name = 'Net Small',
   channel = 3,
   eye = 231,
   mknet = function(self)
      local model = nn.Sequential()

      -- convolution layer
      model:add(nn.SpatialConvolutionMM(3, 48, 9, 9, 4, 4, 4))
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(48, 64, 5, 5, 1, 1))
      model:add(nn.SpatialMaxPooling(2,2,2,2))
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1))
      model:add(nn.SpatialMaxPooling(2,2,2,2))
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1))
      model:add(nn.SpatialMaxPooling(2,2,2,2))
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(64, 32, 3, 3, 1, 1))
      model:add(nn.ReLU())

      -- fully connected layer
      --model:add(nn.Reshape(32*3*3))
      model:add(nn.View(32*3*3))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(32*3*3, 128))
      model:add(nn.ReLU())
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(128, 128))
      model:add(nn.ReLU())
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(128, 17))
      model:add(nn.LogSoftMax())

      return model
   end
}
