return {
   name = '4-16 Test',
   channel = 4,
   def = {
      [1] = {
         [1] = {
            output = 16,
            conv = { k = 10, },
            pool = { size = 2, stride = 1, },
            nlmp = 'ReLU',
         },
      },
      [2] = {
         [1] = {
            transform = {
               name = 'Reshape',
               size = 502,
            },
         },
         [2] = {
            linear = 1,
            nlmp = 'ReLU',
         },
         [3] = {
            linear = 1,
            nlmp = 'LogSoftMax',
         },
      },
   },
}
