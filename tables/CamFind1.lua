return {
   name = 'CamFind1',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 48,
            conv = { k = 10, p = 3, s = 4, },
            nlmp = 'ReLU',
         },
         [2] = {
            output = 64,
            conv = { k = 5, },
            pool = 2,
            nlmp = 'ReLU',
         },
         [3] = {
            output = 64,
            conv = { k = 3, },
            pool = 2,
            nlmp = 'ReLU',
         },
         [4] = {
            output = 64,
            conv = { k = 3, p = 1, },
            pool = 2,
            nlmp = 'ReLU',
         },
         [5] = {
            output = 32,
            conv = { k = 3, p = 1, },
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
            linear = 128,
            nlmp = 'ReLU',
         },
         [3] = {
            linear = 128,
            nlmp = 'ReLU',
         },
         [4] = {
            linear = 17,
            nlmp = 'LogSoftMax',
         },
      },
   },
}
