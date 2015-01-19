return {
   name = 'Half Krizhevsky',
   channel = 3,
   eye = 226,
   mknet = function(self, cuda)
      require("../src/buildModel.lua")

      local nFeatureMaps = {48, 128, 192, 192, 256}
      local filterSize   = {11,   5,   3,   3,   3}
      local convPadding  = { 4,   4,   2,   2,   2}
      local convStride   = { 4,   1,   1,   1,   1}
      local poolSize     = { 1,   2,   2,   1,   2}
      local poolStride   = { 1,   2,   2,   1,   2}
      local hiddenUnits  = {4096, 4096, 1000}

      -- Set colour input
      local mapSize     = {}
      mapSize[0]        = self.eye
      nFeatureMaps[0]   = 3

      local net = buildModel
         ( nFeatureMaps
         , filterSize
         , convPadding
         , convStride
         , poolSize
         , poolStride
         , hiddenUnits
         , mapSize
         , cuda
         )

      return net
   end
}
