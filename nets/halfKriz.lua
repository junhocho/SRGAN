return {
   name = 'Half Krizhevsky',
   channel = 3,
   eye = 226,
   mknet = function(self)
      local net = nn.Sequential()

      net:add(nn.SpatialConvolutionMM(3,  48 , 11, 11 , 4, 4 , 4))
      net:add(nn.ReLU())
      net:add(nn.SpatialConvolutionMM(48,  128 , 5, 5 , 1, 1 , 4))
      net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      net:add(nn.ReLU())
      net:add(nn.SpatialConvolutionMM(128,  192 , 3, 3 , 1, 1 , 2))
      net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      net:add(nn.ReLU())
      net:add(nn.SpatialConvolutionMM(192,  192 , 3, 3 , 1, 1 , 2))
      net:add(nn.ReLU())
      net:add(nn.SpatialConvolutionMM(192,  256 , 3, 3 , 1, 1 , 2))
      net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      net:add(nn.ReLU())
      net:add(nn.Reshape(25600))
      net:add(nn.Linear(25600, 4096))
      net:add(nn.ReLU())
      net:add(nn.Linear(4096, 4096))
      net:add(nn.ReLU())
      net:add(nn.Linear(4096, 1000))
      net:add(nn.LogSoftMax())

      return net
   end
}
