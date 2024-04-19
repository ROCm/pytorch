# Before running, set the below flags and then run
# export MIOPEN_ENABLE_LOGGING=1
# export MIOPEN_ENABLE_LOGGING_CMD=1
# export MIOPEN_LOG_LEVEL=7
# python batchnorm-miopen.py 

import torch

m = torch.nn.BatchNorm2d(100, device='cuda:0') #.to("cuda:0")

# random NCHW tensor on GPU
input = torch.randn(20, 100, 5, 4).to("cuda:0")
input2 = input.clone()

# change tensor dims to NHWC
input = input.to(memory_format=torch.channels_last) # <-- comment/uncomment this line

# Observations:
#
# 1) If you comment line 15, memory format is NCHW and MIOpen is invoked. 
# But CK kernel is not selected/called. See portion of the log that gets generated.
# MIOpen(HIP): Info2 [SearchForSolutions] BnCKFwdTraining: Not applicable
# MIOpen(HIP): Info2 [SearchForSolutions] BnFwdTrainingSpatialSingle: Success.
# MIOpen(HIP): Info2 [PrepareInvoker] Preparing kernel: MIOpenBatchNormFwdTrainSpatial
#
#
# 2) If you leave line 15 uncommented, then memory format is NHWC. 
# The expectation is that MIOpen will be invoked and select the CK kernel. 
# However, I notice that MIOpen itself seems not to be invoked at all. 
# < NO MIOPEN LOG GETS GENERATED >
print('input nhwc')
print(input.shape)
print(input.stride())

print('input2')
print(input2.shape)
print(input2.stride())

# call batch norm
output = m(input)
print("%%%%%%%%%%%%%%%%%% output")
print(output.shape)
print(output.stride())

print("%%%%%%%%%%%%%%%%%% output2")

output2= m(input2)
print(output2.shape)
print(output2.stride())

o1 = output.to(memory_format=torch.contiguous_format)
# print(f"EQ={torch.eq(o1, output2)}")

# MIOpenDriver bnorm -n 20 -c 100 -H 5 -W 4 -M 1 --forw 1 -b 0 -s 1 -r 1
