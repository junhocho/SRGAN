return {
   name = 'VGG-D',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 64,
            conv = { k = 3, p = 1 },
            pool = 1,
            nlmp = 'ReLU',
         },
         [2] = {
            output = 64,
            conv = { k = 3, p = 1 },
            pool = 2,
            nlmp = 'ReLU',
         },
         [3] = {
            output = 128,
            conv = { k = 3, p = 1 },
            pool = 1,
            nlmp = 'ReLU',
         },
         [4] = {
            output = 128,
            conv = { k = 3, p = 1 },
            pool = 2,
            nlmp = 'ReLU',
         },
         [5] = {
            output = 256,
            conv = { k = 3, p = 1 },
            pool = 1,
            nlmp = 'ReLU',
         },
         [6] = {
            output = 256,
            conv = { k = 3, p = 1 },
            pool = 1,
            nlmp = 'ReLU',
         },
         [7] = {
            output = 256,
            conv = { k = 3, p = 1 },
            pool = 2,
            nlmp = 'ReLU',
         },
         [8] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 1,
            nlmp = 'ReLU',
         },
         [9] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 1,
            nlmp = 'ReLU',
         },
         [10] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 2,
            nlmp = 'ReLU',
         },
         [11] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 1,
            nlmp = 'ReLU',
         },
         [12] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 1,
            nlmp = 'ReLU',
         },
         [13] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 2,
            nlmp = 'ReLU',
         },
      },
      [2] = {
         [1] = {
            reshape = 7,
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
