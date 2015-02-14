return {
   name = 'HW04',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 16,
            conv = { k = 9, s = 4, },
            nlmp = 'ReLU',
         },
         [2] = {
            output = 32,
            conv = { k = 9, },
            pool = 2,
            nlmp = 'ReLU',
         },
         [3] = {
            output = 32,
            conv = { k = 9, },
            pool = 2,
            nlmp = 'ReLU',
         },
      },
      [2] = {
         [1] = {
            transform = {
               name = 'Reshape',
               size = 2,
            },
         },
         [2] = {
            linear = 128,
            nlmp = 'ReLU',
         },
         [3] = {
            linear = 128,
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
