return {
   name = 'VGG-D',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 64,
            conv = { k = 3, p = 1 },
            pool = 1,
            relu = true,
         },
         [2] = {
            output = 64,
            conv = { k = 3, p = 1 },
            pool = 2,
            relu = true,
         },
         [3] = {
            output = 128,
            conv = { k = 3, p = 1 },
            pool = 1,
            relu = true,
         },
         [4] = {
            output = 128,
            conv = { k = 3, p = 1 },
            pool = 2,
            relu = true,
         },
         [5] = {
            output = 256,
            conv = { k = 3, p = 1 },
            pool = 1,
            relu = true,
         },
         [6] = {
            output = 256,
            conv = { k = 3, p = 1 },
            pool = 1,
            relu = true,
         },
         [7] = {
            output = 256,
            conv = { k = 3, p = 1 },
            pool = 2,
            relu = true,
         },
         [8] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 1,
            relu = true,
         },
         [9] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 1,
            relu = true,
         },
         [10] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 2,
            relu = true,
         },
         [11] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 1,
            relu = true,
         },
         [12] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 1,
            relu = true,
         },
         [13] = {
            output = 512,
            conv = { k = 3, p = 1 },
            pool = 2,
            relu = true,
         },
      },
      [2] = {
         [1] = {
            reshape = 7,
         },
         [2] = {
            linear = 4096,
            relu = true,
         },
         [3] = {
            linear = 4096,
            relu = true,
         },
         [4] = {
            linear = 1000,
            lsmax = true,
         },
      },
   },
}
