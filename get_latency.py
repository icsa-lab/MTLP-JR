import os
import torch
from torchvision import transforms, datasets
import numpy as np
import time
import math
from ofa.tutorial.imagenet_eval_helper import  evaluate_ofa_subnet
import tensorrt as trt
from torch.autograd import Variable
from ofa.model_zoo import ofa_net
from ofa.utils import download_url,AverageMeter
import common
from ofa.tutorial import AccuracyPredictor, LatencyTable, EvolutionFinder
import torch.backends.cudnn as cudnn
import csv
from ofa.utils.pytorch_utils import count_net_flops,count_parameters

TRT_LOGGER = trt.Logger()
bs = 64
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

cuda_available = torch.cuda.is_available()
if cuda_available:
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print('Using GPU.')
else:
    print('Using CPU.')

ofa_network = ofa_net('ofa_mbv3_d234_e346_k357_w1.2', pretrained=True)


def get_engine(onnx_file_path):
    """Takes an ONNX file and creates a TensorRT engine to run inference with"""
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 32  # 4G
        builder.max_batch_size = 1
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        network.get_input(0).shape = [bs, 3, net_config['r'][0], net_config['r'][0]]
        # print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        engine = builder.build_cuda_engine(network)
        print("Completed creating Engine")
        return engine

imagenet_data_path = './imagenet1k/data/imagenet_1k'
if cuda_available:
    if not os.path.isdir(imagenet_data_path):
        os.makedirs(imagenet_data_path, exist_ok=True)
        print('%s is empty. Download a subset of ImageNet for test.' % imagenet_data_path)
        download_url('https://hanlab.mit.edu/files/OnceForAll/ofa_cvpr_tutorial/imagenet_1k.zip', model_dir='data')
    print('The ImageNet dataset files are ready.')


    data_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            root=os.path.join(imagenet_data_path, 'val'),

        ),
        batch_size=bs,  # test batch size
        shuffle=True,
        num_workers=16,  # number of workers for the data loader
        pin_memory=True,
        drop_last=True,
    )


gpu_ava_delay = AverageMeter()
cpu_ava_delay = AverageMeter()
onnxpath = './onnxs/tmp.onnx'
target_hardware = 'note10'
latency_table = LatencyTable(device=target_hardware)
accuracy_predictor = AccuracyPredictor(
    pretrained=True,
    device='cuda:0' if cuda_available else 'cpu'
)



latency_constraint = 60 # ms, suggested range [15, 33] ms
P = 100  # The size of population in each generation
N = 500  # How many generations of population to be searched
r = 0.25  # The ratio of networks that are used as parents for next generation
params = {
    'constraint_type': target_hardware, # Let's do FLOPs-constrained search
    'efficiency_constraint': latency_constraint,
    'mutate_prob': 0.1, # The probability of mutation in evolutionary search
    'mutation_ratio': 0.5, # The ratio of networks that are generated through mutation in generation n >= 2.
    'efficiency_predictor': latency_table, # To use a predefined efficiency predictor.
    'accuracy_predictor': accuracy_predictor, # To use a predefined accuracy_predictor predictor.
    'population_size': P,
    'max_time_budget': N,
    'parent_ratio': r,
}
# build the evolution finder
finder = EvolutionFinder(**params)
a = [2,3,4]
reso = [160, 176, 192, 208, 224]
trverse = []
for i1 in a:
    for i2 in a:
        for i3 in a:
            for i4 in a:
                for i5 in a:
                    for re in reso:
                        trverse.append([[i1,i2,i3,i4,i5],[re]])
csv_f = open('./dataset/1215__test.csv', 'w', encoding='utf-8', newline='')
csv_writer = csv.writer(csv_f)

csv_writer.writerow(['arch_config', 'gpu latency', 'pred_accs', 'flops', 'params','top1'])
for i in range(1215*3):
    i = i % 1215
    net_config, efficiency, acc = finder.random_sample()
    net_config['d'] = trverse[i][0]
    net_config['r'][0] = trverse[i][1][0]
    data_loader.dataset.transform = transforms.Compose([
        transforms.Resize(int(math.ceil(int(net_config['r'][0])/0.875))),
        transforms.CenterCrop(net_config['r'][0]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
    cudnn.benchmark = True
    assert 'ks' in net_config and 'd' in net_config and 'e' in net_config
    assert len(net_config['ks']) == 20 and len(net_config['e']) == 20 and len(net_config['d']) == 5

    ofa_network.set_active_subnet(ks=net_config['ks'], d=net_config['d'], e=net_config['e'])
    subnet = ofa_network.get_active_subnet().to('cuda:1')
    top1 = evaluate_ofa_subnet(ofa_network,  imagenet_data_path,net_config,data_loader,batch_size=30,device='cuda:0')
    flops = count_net_flops(subnet, data_shape=(1, 3, net_config['r'][0], net_config['r'][0]))
    params = count_parameters(subnet)

    input_name = ['input']
    output_name = ['output']
    input = Variable(torch.randn(bs, 3, net_config['r'][0], net_config['r'][0])).cuda('cuda:1')
    torch.onnx.ExportTypes()
    torch.onnx.export(subnet, input, onnxpath, export_params=True,input_names=input_name, output_names=output_name, verbose=False)

    gpu_ava_delay.reset()
    cpu_ava_delay.reset()

    with get_engine(onnxpath) as engine, engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.numpy(), labels.numpy()
            inputs[0].host = images.astype(np.float32)
            t1 = time.time()
            trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
            t2 =time.time()
            if i >5:
                gpu_delay = (t2-t1) * 1000
                gpu_ava_delay.update(gpu_delay)
        print('gpu:{}-pred_acc:{}-flops:{}-params:{}-top1:{}'.format(round(gpu_ava_delay.avg, 4),
                                                                          round(acc, 4),
                                                                          flops,
                                                                          params,
                                                                          top1))
        csv_array = [net_config, round(gpu_ava_delay.avg, 4), round(acc, 4), flops, params,top1]
        csv_writer.writerow(csv_array)
        csv_f.flush()






