--------------------------------------------------------------------------------
-- e-Lab nn-X hardware model 04
name = 'HW04'
--------------------------------------------------------------------------------
-- ImageNet Classification with Deep Convolutional Neural Networks
-- http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
--------------------------------------------------------------------------------

-- Network parameters ----------------------------------------------------------
nFeatureMaps = {16, 32, 32}
filterSize   = { 9,  9,  9}
convPadding  = { 0,  0,  0}
convStride   = { 4,  1,  1}
poolSize     = { 1,  2,  2}
poolStride   = { 1,  2,  2}
hiddenUnits  = {128, 128, 1000}

-- Set colour input
mapSize         = {}
mapSize[0]      = 149
nFeatureMaps[0] = 3
