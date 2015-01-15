return {
   name = 'Large-Net-Test',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 48,
            conv = { k = 9, },
            pool = 2,
            relu = true,
         },
         [2] = {
            output = 128,
            conv = { k = 9, },
            pool = 2,
            relu = true,
         },
         [3] = {
            output = 192,
            conv = { k = 9, },
            pool = { size = 2, stride = 1, },
            relu = true,
         },
      },
      [2] = {
         [1] = {
            reshape = 41,
         },
         [2] = {
            linear = 1,
            relu = true,
         },
         [2] = {
            linear = 1,
            lsmax = true,
         },
      },
   },
}
