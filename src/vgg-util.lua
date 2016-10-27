require 'torch'

local vgg_mean = {103.939, 116.779, 123.68} 
function vgg_preprocess(img)
    local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
    local perm = torch.LongTensor{3, 2, 1}
    return img:index(2, perm):mul(255):add(-1, mean) 
end

function vgg_deprocess(img)
  local mean = img.new(vgg_mean):view(1, 3, 1, 1):expandAs(img)
  local perm = torch.LongTensor{3, 2, 1}
  return (img + mean):div(255):index(2, perm)
end
