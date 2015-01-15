return {
   name = 'CamFind1',
   channel = 3,
   def = {
      [1] = {
         [1] = {
            output = 48,
            conv = { k = 10, p = 3, s = 4, },
            relu = true,
         },
         [2] = {
            output = 64,
            conv = { k = 5, },
            pool = 2,
            relu = true,
         },
         [3] = {
            output = 64,
            conv = { k = 3, },
            pool = 2,
            relu = true,
         },
         [4] = {
            output = 64,
            conv = { k = 3, p = 1, },
            pool = 2,
            relu = true,
         },
         [5] = {
            output = 32,
            conv = { k = 3, p = 1, },
            relu = true,
         },
      },
      [2] = {
         [1] = {
            reshape = 6,
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
            linear = 17,
            lsmax = true,
         },
      },
   },
}
