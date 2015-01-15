return {
   name = 'Krizhevsky',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 96,
            conv = { k = 11, p = 3, s = 4, },
            pool = { size = 3, stride = 2, },
            relu = true,
         },
         [2] = {
            output = 256,
            conv = { k = 5, p = 2, },
            pool = { size = 3, stride = 2, },
            relu = true,
         },
         [3] = {
            output = 384,
            conv = { k = 3, p = 1, },
            relu = true,
         },
         [4] = {
            output = 384,
            conv = { k = 3, p = 1, },
            relu = true,
         },
         [5] = {
            output = 256,
            conv = { k = 3, p = 1, },
            pool = { size = 3, stride = 2, },
            relu = true,
         },
      },
      [2] = {
         [1] = {
            reshape = 6,
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
