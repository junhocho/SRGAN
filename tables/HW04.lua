return {
   name = 'HW04',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 16,
            conv = { k = 9, s = 4, },
            relu = true,
         },
         [2] = {
            output = 32,
            conv = { k = 9, },
            pool = 2,
            relu = true,
         },
         [3] = {
            output = 32,
            conv = { k = 9, },
            pool = 2,
            relu = true,
         },
      },
      [2] = {
         [1] = {
            reshape = 2,
         },
         [2] = {
            linear = 128,
            relu = true,
         },
         [3] = {
            linear = 128,
            relu = true,
         },
         [4] = {
            linear = 1000,
            lsmax = true,
         },
      },
   },
}
