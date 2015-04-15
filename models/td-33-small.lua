return {
   name = 'TeraDeep Net 33 Small',
   channel = 3,
   eye = 231,
   mknet = function(self)
      local model = nn.Sequential()

      -- convolution layer
      model:add(nn.SpatialConvolutionMM(3, 64, 3, 3, 1, 1))
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1))
      model:add(nn.SpatialMaxPooling(2,2,2,2))
      model:add(nn.ReLU())

      model:add(nn.SpatialConvolutionMM(128, 128, 3, 3, 1, 1))
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(128, 256, 3, 3, 1, 1))
      model:add(nn.SpatialMaxPooling(4,4,4,4))
      model:add(nn.ReLU())
      
      model:add(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1))
      model:add(nn.ReLU())
      model:add(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1))
      model:add(nn.SpatialMaxPooling(4,4,4,4))
      model:add(nn.ReLU())

      -- fully connected layer
      model:add(nn.View(256*5*5))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(256*5*5, 1024))
      model:add(nn.ReLU())
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(1024, 1024))
      model:add(nn.ReLU())
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(1024, 17))
      model:add(nn.LogSoftMax())

      return model
   end
}
