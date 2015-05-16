return {
   name = 'OverFeat-accurate',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 96,
            conv = { k = 7, p = 0, s = 2, },
            pool = { size = 3, stride = 3, },
            nlmp = 'ReLU',
         },
         [2] = {
            output = 256,
            conv = { k = 7, p = 0, },
            pool = { size = 2, stride = 2, },
            nlmp = 'ReLU',
         },
         [3] = {
            output = 512,
            conv = { k = 3, p = 1, },
            nlmp = 'ReLU',
         },
         [4] = {
            output = 512,
            conv = { k = 3, p = 1, },
            nlmp = 'ReLU',
         },
         [5] = {
            output = 1024,
            conv = { k = 3, p = 1, },
            pool = { size = 3, stride = 3, },
            nlmp = 'ReLU',
         },
      },
      [2] = {
         [1] = {
            transform = {
               name = 'Reshape',
               size = 5,
            },
         },
         [2] = {
            linear = 4096,
            nlmp = 'ReLU',
         },
         [3] = {
            linear = 4096,
            nlmp = 'ReLU',
         },
         [4] = {
            linear = 1000,
            nlmp = 'LogSoftMax',
         },
      },
   },
   mknet = function(self)
      local build = assert(require('src/builder'))
      local net, eye = build:cpu(self)
      self.eye = eye

      return net
   end
}
