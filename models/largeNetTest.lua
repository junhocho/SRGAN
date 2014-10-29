--------------------------------------------------------------------------------
-- TeraDeep Large network test for nn-X hardware: a highly modified subset of:
-- note: 9x9 filters, convolutional Stride of 1 and pooling of 2!!!
name = 'Large-Net-Test'
--------------------------------------------------------------------------------

-- Network parameters ----------------------------------------------------------
nFeatureMaps = {48, 128, 192}
filterSize   = { 9,   9,   9}
convPadding  = { 0,   0,   0}
convStride   = { 1,   1,   1}
poolSize     = { 2,   2,   2}
poolStride   = { 2,   2,   1}
hiddenUnits  = { 1,   1}

-- Set colour input
mapSize         = {}
mapSize[0]      = 225
nFeatureMaps[0] = 3
