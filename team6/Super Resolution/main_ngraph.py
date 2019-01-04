import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

## onnx
from torch.autograd import Variable
import torchvision
import torch
import torch.nn as nn
import torch.onnx
import onnx
###
'''
## trt
import tensorrt as rt
import pycuda.driver as cuda
import pycuda.autoinit
##

## build trt engine

import numpy as np
import os
TRT_LOGGER = rt.Logger(rt.Logger.WARNING)
ONNX_MODEL = "./edsr2xNEW.onnx" 
im_h = 678
im_w = 1020
engine_file_path = 'edsr.trt'
def build_engine():
    with rt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, rt.OnnxParser(network, TRT_LOGGER) as parser:
        
        builder.max_workspace_size = 2**10
        
        with open(ONNX_MODEL,'rb') as model:
            parser.parse(model.read())
        engine = builder.build_cuda_engine(network)
        
        with open(engine_file_path, "wb") as f:
                f.write(engine.serialize())
        return engine
'''
'''
if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, rt.Runtime(TRT_LOGGER) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())
else:
    engine = build_engine()
print(engine)
'''
onnx_protobuf = onnx.load('./edsr2xNEW.onnx')
#print(onnx.helper.printable_graph(onnx_protobuf.graph))

# Convert a serialized ONNX model to an ngraph model

from ngraph_onnx.onnx_importer.importer import import_onnx_model
ng_model = import_onnx_model(onnx_protobuf)[0]


# Using an ngraph runtime (CPU backend), create a callable computation
import ngraph as ng
runtime = ng.runtime(backend_name='CPU')
edsr = runtime.computation(ng_model['output'], *ng_model['inputs'])

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if args.data_test == 'video':
    from videotester import VideoTester
    model = model.Model(args, checkpoint)
    t = VideoTester(args, model, checkpoint)
    t.test()
else:
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        #t = Trainer(args, loader, model, loss, checkpoint)
        t = Trainer(args, loader, model, loss, checkpoint, ngraph=edsr)
        while not t.terminate():
            t.train()
            t.test()

        checkpoint.done()

