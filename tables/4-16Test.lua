return {
   name = '4-16 Test',
   channel = 4,
   def = {
      [1] = {
         output = 16,
         conv = { k = 10, },
         pool = { size = 2, stride = 1, },
         nlmp = 'ReLU',
      },
   },
   mknet = function(self)
      local build = assert(require('src/builder'))
      local net, eye = build:cpu(self)
      self.eye = eye

      return net
   end
}
