return {
   name = '4-16 Test',
   channel = 4,
   def = {
      [1] = {
         output = 16,
         conv = { k = 10, },
         pool = { size = 2, stride = 1, },
         nlmp = 'ReLU',
      },
   },
}
