return {
   name = 'VGG-D',
   channel = 3,
   def = {},

   mknet = function(self)
      local build = assert(require('src/builder'))

      table.insert(self.def, {
         output = 64,
         conv = { k = 3, p = 1 },
         pool = 1,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 64,
         conv = { k = 3, p = 1 },
         pool = 2,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 128,
         conv = { k = 3, p = 1 },
         pool = 1,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 128,
         conv = { k = 3, p = 1 },
         pool = 2,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 256,
         conv = { k = 3, p = 1 },
         pool = 1,
         nlmp = 'ReLU',
      })


      table.insert(self.def, {
         output = 256,
         conv = { k = 3, p = 1 },
         pool = 1,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 256,
         conv = { k = 3, p = 1 },
         pool = 2,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 512,
         conv = { k = 3, p = 1 },
         pool = 1,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 512,
         conv = { k = 3, p = 1 },
         pool = 1,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 512,
         conv = { k = 3, p = 1 },
         pool = 2,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 512,
         conv = { k = 3, p = 1 },
         pool = 1,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 512,
         conv = { k = 3, p = 1 },
         pool = 1,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         output = 512,
         conv = { k = 3, p = 1 },
         pool = 2,
         nlmp = 'ReLU',
      })
---[[
      table.insert(self.def, {
         transform = {
            name = 'Reshape',
            size = 7,
         },
      })

      table.insert(self.def, {
         linear = 4096,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         linear = 4096,
         nlmp = 'ReLU',
      })

      table.insert(self.def, {
         linear = 1000,
         nlmp = 'LogSoftMax',
      })
--]]
      local net, eye = build:cpu(self)
      self.eye = eye

      return net
   end
}
