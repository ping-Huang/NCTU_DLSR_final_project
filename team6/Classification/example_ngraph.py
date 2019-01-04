from benchmark import benchmarking
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.resnet import resnet18
import os
import sys
#data_dir = os.environ['TESTDATADIR']
#assert data_dir is not None, "No data directory"
from models import resnet20_cifar
from collections import OrderedDict
import time
import numpy as np

import onnx
import ngraph as ng
from ngraph_onnx.onnx_importer.importer import import_onnx_model

ONNX_MODEL = "resnet20_cinic_9000.onnx"
im_h = 32
im_w = 32
#ngraph
onnx_protobuf = onnx.load(ONNX_MODEL)
ng_model = import_onnx_model(onnx_protobuf)[0]
runtime = ng.runtime(backend_name='CPU')
resnet = runtime.computation(ng_model['output'], *ng_model['inputs'])
#tensorrt
'''
TRT_LOGGER = rt.Logger(rt.Logger.WARNING)
def build_engine():
    with rt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, rt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_workspace_size = 2**30
        builder.max_batch_size = 32
        #builder.fp16_mode = True
        #builder.strict_type_constraints = True
        
        with open(ONNX_MODEL,'rb') as model:
            parser.parse(model.read())
        
        return builder.build_cuda_engine(network)
engine = build_engine()
batch_size = 100
def do_inference(engine,img):
    with engine.create_execution_context() as context:
        
        img = img.astype(np.float32)
        output = np.empty((batch_size,10), dtype = np.float32)
        d_input = cuda.mem_alloc(img.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)
        stream = cuda.Stream()
        
        cuda.memcpy_htod_async(d_input, img, stream)
        context.execute_async(batch_size=batch_size, bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()
    return output


'''
model = resnet20_cifar()
inputs = []
targets = []
@benchmarking(team=6, task=0, model=model, preprocess_fn=None)
def inference(model, data_loader,**kwargs):
    total = 0
    correct = 0
    assert kwargs['device'] != None, 'Device error'
    device = kwargs['device']
    if device == "cpu":
        #for inputs, targets in testloader:
        for idx in range(len(test_loader)):
            inputs[idx], targets[idx] = inputs[idx].to(device).numpy(), targets[idx].to(device).numpy()
            outputs = resnet(inputs[idx])
            pred = np.argmax(outputs,axis=1)
            total += len(targets[idx])
            if(targets[idx].shape != pred.shape):
                correct += np.equal(targets[idx],pred[0:len(targets)]).sum()
            else:
                correct += np.equal(targets[idx],pred).sum()
    else:
        model.to(device)
        checkpoint = torch.load('ckpt.t7')
        new_state_dict = OrderedDict()
        for k, v in checkpoint['net'].items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        # load params
        model.load_state_dict(new_state_dict)
        model.eval()

        with torch.no_grad():
            #for inputs, targets in testloader:
            for idx in range(len(test_loader)):
                inputs[idx], targets[idx] = inputs[idx].to(device), targets[idx].to(device)
                outputs = model(inputs[idx])
                _, predicted = outputs.max(1)
                total += targets[idx].size(0)
                correct += predicted.eq(targets[idx]).sum().item()
        '''
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device).numpy(), targets.to(device).numpy()
            outputs = do_inference(engine,inputs)
            pred = np.argmax(outputs,axis=1)
            total += len(targets)
            if(targets.shape != pred.shape):
                correct += np.equal(targets,pred[0:len(targets)]).sum()
            else:
                correct += np.equal(targets,pred).sum()
        '''
    acc = 100.*correct/total
    print(acc)
    return acc

if __name__=='__main__':

    cinic_directory = os.environ['TESTDATADIR']
    assert cinic_directory is not None, "No data directory"
    #cinic_directory = './data/cinic-10'
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=cinic_mean, std=cinic_std)])

    testset = torchvision.datasets.ImageFolder(root=(cinic_directory + '/test'), transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=9000, shuffle=False, num_workers=4)
    for ins, ts in testloader:
        inputs.append(ins)
        targets.append(ts)
    inference(model, testloader)
