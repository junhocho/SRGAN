return {
   name = 'Large-Net-Test',
   channel = 3,
   def = {
      [1] = {
         output = 48,
         conv = { k = 9, },
         pool = 2,
         nlmp = 'ReLU',
      },
      [2] = {
         output = 128,
         conv = { k = 9, },
         pool = 2,
         nlmp = 'ReLU',
      },
      [3] = {
         output = 192,
         conv = { k = 9, },
         pool = { size = 2, stride = 1, },
         nlmp = 'ReLU',
      },
   },
}
