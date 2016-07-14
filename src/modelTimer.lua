local profileTime = {}
local xlua = assert(require('xlua'))

local function calc_time_cuda(net, img, iterations)
   collectgarbage()

   cutorch.setDevice(1)
   cudnn.convert(net, cudnn, function(m) return torch.type(m):find('MaxPooling') end)
   net:cuda()

   print('==> using GPU #' .. cutorch.getDevice())
   cutorch.synchronize()

   local tmp = false
   local timer = torch.Timer()
   local timing = torch.FloatTensor(iterations)
   local t = 0

   -- iterations plus one to prime the jit
   for i=1, (iterations+1) do
      xlua.progress(i, iterations)

      timer:reset()

      tmp = net:forward(img:cuda())
      cutorch.synchronize()
      tmp:float()

      t = timer:time().real
      timing[(i%iterations)+1] = t
   end

   return timing:mean(), tmp
end

local function calc_time_cpu(net, img, iterations)
   local tmp = false
   local timer = torch.Timer()
   local timing = torch.FloatTensor(iterations)
   local t = 0

   -- iterations plus one to prime the jit
   for i=1, (iterations+1) do
      xlua.progress(i, iterations)

      timer:reset()

      tmp = net:forward(img)

      t = timer:time().real
      timing[(i%iterations)+1] = t
   end

   return timing:mean(), tmp
end

function profileTime:time(net, img, iterations, platform)
   iterations = iterations or 10
   local time = { total = 0, conv = 0, mlp = 0, }

   if platform == 'cuda' then

      time.total = calc_time_cuda(net, img, iterations)

   else

      time.total = calc_time_cpu(net, img, iterations)

   end

   return time
end

return profileTime
