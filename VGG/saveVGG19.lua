require 'loadcaffe'

model = loadcaffe.load('VGG_ILSVRC_19_layers_deploy.prototxt', 'VGG_ILSVRC_19_layers.caffemodel', 'nn')

for _ = 1,10 do
	model:remove() -- Remove FCs
end

torch.save('VGG19.t7', model)
