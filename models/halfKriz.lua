--------------------------------------------------------------------------------
-- Krizhevsky, double columns
name = 'Half Krizhevsky'
--------------------------------------------------------------------------------

-- Network parameters ----------------------------------------------------------
nFeatureMaps = {48, 128, 192, 192, 256}
filterSize   = {11,   5,   3,   3,   3}
convPadding  = { 4,   4,   2,   2,   2}
convStride   = { 4,   1,   1,   1,   1}
poolSize     = { 1,   2,   2,   1,   2}
poolStride   = { 1,   2,   2,   1,   2}
hiddenUnits  = {4096, 4096, 1000}

-- Set colour input
mapSize         = {}
mapSize[0]      = 224
nFeatureMaps[0] = 3
