import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm


## trt
#import tensorrt as rt
#import pycuda.driver as cuda
#import pycuda.autoinit
import numpy as np
##

class Trainer():
    #def __init__(self, args, loader, my_model, my_loss, ckp):
    def __init__(self, args, loader, my_model, my_loss, ckp, my_engine=None, ngraph=None):
        self.engine = my_engine
        self.edsr = ngraph
        
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
    
    ### add do_inference here    
    def do_inference(self, engine, img):
        with engine.create_execution_context() as context:
        
            img = img.astype(np.float32)
            output = np.empty([1, 3, 1356, 2040], dtype = np.float32)
            # Allocate device memory for inputs and outputs.
            d_input = cuda.mem_alloc(img.nbytes)
            d_output = cuda.mem_alloc(output.nbytes)
            # Create a stream in which to copy inputs/outputs and run inference.
            stream = cuda.Stream()
        
            # Transfer input data to the GPU.
            cuda.memcpy_htod_async(d_input, img, stream)
            # Run inference.
            context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
            # Transfer predictions back from the GPU.
            cuda.memcpy_dtoh_async(output, d_output, stream)
            # Synchronize the stream
            stream.synchronize()
            # Return the host output.
        return output
    
    def train(self):
        self.optimizer.schedule()
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _, idx_scale) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch() + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename, _ in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    #sr = self.model(lr, idx_scale)
                    sr = self.model(lr)
                    ######## tensorRT inference #######
                    '''
                    print(self.engine)
                    lrNumpy=lr.data.numpy()
                    output = self.do_inference(self.engine,lrNumpy)  # tensorRT engine here
                    print("RT inference done !")
                    sr=torch.from_numpy(output)  
                    '''
                    ######## tensorRT inference done !#######
                    '''
                    ######## ngraph #############
                    lrNumpy=lr.data.numpy()
                    output = self.edsr(lrNumpy)
                    sr=torch.from_numpy(output)
                    ######## ngraph done#############
                    '''
                    print("sr.shape:",sr.shape)
                    
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)
        return self.ckp.log[-1, idx_data, idx_scale].data.tolist()

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

