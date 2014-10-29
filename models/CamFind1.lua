--------------------------------------------------------------------------------
-- TeraDeep CamFind network #1
name = 'CamFind1'
--------------------------------------------------------------------------------

-- Network parameters ----------------------------------------------------------
nFeatureMaps = {48, 64, 64, 64, 32}
filterSize   = {10,  5,  3,  3,  3}
convPadding  = { 3,  0,  0,  1,  1}
convStride   = { 4,  1,  1,  1,  1}
poolSize     = { 1,  2,  2,  2,  1}
poolStride   = { 1,  2,  2,  2,  1}
hiddenUnits  = {128, 128, 17}

-- Set colour input
mapSize         = {}
mapSize[0]      = 231
nFeatureMaps[0] = 3
