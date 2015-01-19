--------------------------------------------------------------------------------
-- Teradeep4-16 1 layer test
name = '4-16 Test'
--------------------------------------------------------------------------------

-- Network parameters ----------------------------------------------------------
nFeatureMaps = {16}
filterSize   = {10}
convPadding  = { 0}
convStride   = { 1}
poolSize     = { 2}
poolStride   = { 1}
hiddenUnits  = { 1, 1}

-- Set colour input
mapSize         = {}
mapSize[0]      = 512
nFeatureMaps[0] = 4
