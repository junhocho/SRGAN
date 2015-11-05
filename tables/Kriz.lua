return {
   name = 'Krizhevsky',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 64,
            conv = { k = 11, p = 3, s = 4, },
            pool = { size = 3, stride = 2, },
            nlmp = 'ReLU',
         },
         [2] = {
            columns = 2,
            output = 192,
            conv = { k = 5, p = 2, },
            pool = { size = 3, stride = 2, },
            nlmp = 'ReLU',
         },
         [3] = {
            output = 384,
            conv = { k = 3, p = 1, },
            nlmp = 'ReLU',
         },
         [4] = {
            columns = 2,
            output = 256,
            conv = { k = 3, p = 1, },
            nlmp = 'ReLU',
         },
         [5] = {
            columns = 2,
            output = 256,
            conv = { k = 3, p = 1, },
            pool = { size = 3, stride = 2, },
            nlmp = 'ReLU',
         },
      },
      [2] = {
         [1] = {
            transform = {
               name = 'Reshape',
               size = 6,
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
