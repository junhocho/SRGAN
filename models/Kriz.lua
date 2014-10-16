--------------------------------------------------------------------------------
-- ILSVRC 2012 classification winner
-- Krizhevsky, double columns
name = 'Krizhevsky'
--------------------------------------------------------------------------------
-- ImageNet Classification with Deep Convolutional Neural Networks
-- http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
--------------------------------------------------------------------------------

-- Network parameters ----------------------------------------------------------
nFeatureMaps = {96, 256, 384, 384, 256}
filterSize   = {11,   5,   3,   3,   3}
convPadding  = { 3,   2,   1,   1,   1}
convStride   = { 4,   1,   1,   1,   1}
poolSize     = { 3,   3,   1,   1,   3}
poolStride   = { 2,   2,   1,   1,   2}
hiddenUnits  = {4096, 4096, 1000}

-- Set colour input
mapSize         = {}
mapSize[0]      = 224
nFeatureMaps[0] = 3
