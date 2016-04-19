require 'caffe'

net=caffe.Net('/data/gpuuser/CAFFE/caffe/models/bvlc_alexnet/deploy.prototxt','/data/gpuuser/CAFFE/caffe/models/bvlc_alexnet/bvlc_alexnet.caffemodel','test')
input=torch.FloatTensor(10,3,227,227)
output=net:forward( input )

print(output)

--gradOutput=torch.FloatTensor(10,1000,1,1)
--gradInput=net:backward(input,gradOutput)

--print(gradInput)
