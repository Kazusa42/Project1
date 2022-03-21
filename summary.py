from thop import profile
import torch
import nets.densenet
from torchvision.models import resnet50

net1 = nets.densenet.densenet121()
net3 = nets.densenet.densenet169()
net2 = resnet50()

input = torch.randn(1, 3, 416, 416)
flops1, params1 = profile(net1, inputs=(input, ))
flops2, params2 = profile(net2, inputs=(input, ))
flops3, params3 = profile(net3, inputs=(input, ))

print('resnet50')
print('flops: %s; params: %s' % (flops2, params2))
print('densenet121')
print('flops: %s; params: %s' % (flops1, params1))
print('densenet169')
print('flops: %s; params: %s' % (flops3, params3))
