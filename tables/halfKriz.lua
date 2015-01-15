return {
   name = 'Half Krizhevsky',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 48,
            conv = { k = 11, p = 4, s = 4, },
            nlmp = 'ReLU',
         },
         [2] = {
            output = 128,
            conv = { k = 5, p = 4, },
            pool = 2,
            nlmp = 'ReLU',
         },
         [3] = {
            output = 192,
            conv = { k = 3, p = 2, },
            pool = 2,
            nlmp = 'ReLU',
         },
         [4] = {
            output = 192,
            conv = { k = 3, p = 2, },
            nlmp = 'ReLU',
         },
         [5] = {
            output = 256,
            conv = { k = 3, p = 2, },
            pool = 2,
            nlmp = 'ReLU',
         },
      },
      [2] = {
         [1] = {
            reshape = 10,
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
}
