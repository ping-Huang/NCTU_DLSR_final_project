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

onnx_protobuf = onnx.load('resnet20_cinic_100.onnx')
ng_model = import_onnx_model(onnx_protobuf)[0]
runtime = ng.runtime(backend_name='CPU')
resnet = runtime.computation(ng_model['output'], *ng_model['inputs'])



model = resnet20_cifar()
@benchmarking(team=6, task=0, model=model, preprocess_fn=None)
def inference(model, data_loader,**kwargs):
    total = 0
    correct = 0
    assert kwargs['device'] != None, 'Device error'
    device = kwargs['device']
    if device == "cpu":
        for batch_idx, (inputs, targets) in enumerate(testloader):
            t0 = time.time()
            inputs, targets = inputs.to(device).numpy(), targets.to(device).numpy()
            outputs = resnet(inputs)
            pred = np.argmax(outputs,axis=1)
            total += len(targets)
            if(targets.shape != pred.shape):
                correct += np.equal(targets,pred[0:len(targets)]).sum()
            else:
                correct += np.equal(targets,pred).sum()
            print(time.time()-t0)
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
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
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
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)
    inference(model, testloader)
