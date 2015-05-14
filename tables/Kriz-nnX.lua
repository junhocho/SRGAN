return {
   name = 'Krizhevsky',
   channel = 3,
   eye = 231,
   def = {
      [1] = {
         output = 96,
         conv = { k = 9, p = 2, s = 4, },
         pool = { size = 2, stride = 2, },
         nlmp = 'ReLU',
      },
      [2] = {
         output = 256,
         conv = { k = 5, p = 1, },
         pool = { size = 2, stride = 2, },
         nlmp = 'ReLU',
      },
      [3] = {
         output = 384,
         conv = { k = 3, p = 1, },
         nlmp = 'ReLU',
      },
      [4] = {
         output = 384,
         conv = { k = 3, p = 1, },
         nlmp = 'ReLU',
      },
      [5] = {
         output = 256,
         conv = { k = 3, p = 1, },
         pool = { size = 2, stride = 2, },
         nlmp = 'ReLU',
      },
      [6] = {
         transform = {
            name = 'Reshape',
            size = 6,
         },
      },
      [7] = {
         linear = 4096,
         nlmp = 'ReLU',
      },
      [8] = {
         linear = 4096,
         nlmp = 'ReLU',
      },
      [9] = {
         linear = 1000,
         nlmp = 'LogSoftMax',
      },
   },
   mknet = function(self)
      local build = assert(require('src/builder'))
      local net, eye = build:cpu(self)
      self.eye = self.eye or eye

      return net
   end
}
