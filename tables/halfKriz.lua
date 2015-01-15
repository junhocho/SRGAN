return {
   name = 'Half Krizhevsky',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 48,
            conv = { k = 11, p = 4, s = 4, },
            relu = true,
         },
         [2] = {
            output = 128,
            conv = { k = 5, p = 4, },
            pool = 2,
            relu = true,
         },
         [3] = {
            output = 192,
            conv = { k = 3, p = 2, },
            pool = 2,
            relu = true,
         },
         [4] = {
            output = 192,
            conv = { k = 3, p = 2, },
            relu = true,
         },
         [5] = {
            output = 256,
            conv = { k = 3, p = 2, },
            pool = 2,
            relu = true,
         },
      },
      [2] = {
         [1] = {
            reshape = 10,
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
