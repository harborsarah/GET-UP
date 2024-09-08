import time
import argparse
import datetime
import sys
import os
from datetime import datetime
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.utils as utils
from GCN_dataloader import GCNRADDataLoader
from models.model import *
from models.losses import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch.distributed as dist
import torch.multiprocessing as mp
import random
from pytorch3d.loss import chamfer_distance

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='GET-UP PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--mode',                               type=str,   help='train or test', default='train')
parser.add_argument('--model_name',                         type=str,   help='model name', default='GET_UP_main')
parser.add_argument('--main_path',                          type=str,   help='main path of data', required=True)
parser.add_argument('--train_image_path',                   type=str,   help='path of training image', required=True)
parser.add_argument('--train_radar_path',                   type=str,   help='path of training radar', required=True)
parser.add_argument('--train_ground_truth_path',            type=str,   help='path of D', required=True)
parser.add_argument('--train_ground_truth_nointer_path',    type=str,   help='path of D_acc', required=True)
parser.add_argument('--train_lidar_path',                   type=str,   help='path of single lidar depth', required=True)
parser.add_argument('--test_image_path',                    type=str,   help='path of testing image', required=True)
parser.add_argument('--test_radar_path',                    type=str,   help='path of testing radar', required=True)
parser.add_argument('--test_ground_truth_path',             type=str,   help='path of testing ground truth', required=True)

parser.add_argument('--k',                                  type=int,   help='k nearest neighbor', default=4)
parser.add_argument('--num_heads',                          type=int,   help='number of heads of attention', default=4)
parser.add_argument('--input_height',                       type=int,   help='input height', default=352)
parser.add_argument('--input_width',                        type=int,   help='input width',  default=704)
parser.add_argument('--radar_gcn_channel_in',               type=int,   help='input channels',  default=6)
parser.add_argument('--radar_gcn_channel_out',              type=int,   help='output channels',  default=256)
parser.add_argument('--radar_input_channels',               type=int,   help='number of input radar channels', default=4)
parser.add_argument('--encoder_radar',                      type=str,   help='type of encoder of radar channels, resnet34', default='resnet18')
parser.add_argument('--encoder',                            type=str,   help='type of encoder', default='resnet34_bts')
parser.add_argument('--lidar_points',                       type=int,   help='sampling lidar points',  default=128)
parser.add_argument('--lidar_channel_out',                  type=int,   help='output channels',  default=256)
parser.add_argument('--num_upsample_unit',                  type=int,   help='number of upsampling unit', default=1)
parser.add_argument('--activation',                         type=str,   help='activation function', default='relu')
parser.add_argument('--sparse_conv_type',                   type=str,   help='type of sparse conv', default='distance_aware')
parser.add_argument('--norm_point',                                     help='if set, normalize point cloud', action='store_true')


parser.add_argument('--max_depth',                          type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--weight_decay',                       type=float, help='weight decay factor for optimization', default=1e-2)
parser.add_argument('--learning_rate',                      type=float, help='initial learning rate', default=1e-4)
parser.add_argument('--num_epochs',                         type=int,   help='number of epochs', default=50)
parser.add_argument('--retrain',                                        help='if used with checkpoint_path, will restart training from step zero', action='store_true')
parser.add_argument('--batch_size',                         type=int,   help='batch size', default=4)
parser.add_argument('--adam_eps',                           type=float, help='epsilon in Adam optimizer', default=1e-6)
parser.add_argument('--w_smoothness',                       type=float, help='Weight of local smoothness loss', default=0.00)
parser.add_argument('--w_chamfer',                          type=float, help='Weight of chamfer loss', default=0.00)
parser.add_argument('--w_nointer_depth',                    type=float, help='Weight of no interpolated depth map loss', default=0.00)
parser.add_argument('--reg_loss',                           type=str,   help='loss function for depth regression - l1/silog', default='l1')
parser.add_argument('--end_learning_rate',                  type=float, help='end learning rate', default=-1)

parser.add_argument('--log_directory',                      type=str,   help='directory to save checkpoints and summaries', default='')
parser.add_argument('--checkpoint_path',                    type=str,   help='path to a checkpoint to load', default='')
parser.add_argument('--log_freq',                           type=int,   help='Logging frequency in global steps', default=100)
parser.add_argument('--save_freq',                          type=int,   help='Checkpoint saving frequency in global steps', default=500)

parser.add_argument('--do_online_eval',                                 help='if set, perform online eval in every eval_freq steps', action='store_true')
parser.add_argument('--eval_freq',                          type=int,   help='Online evaluation frequency in global steps', default=500)
parser.add_argument('--eval_summary_directory',             type=str,   help='output directory for eval summary,'
                                                                             'if empty outputs to checkpoint folder', default='')
parser.add_argument('--min_depth_eval',                     type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval',                     type=float, help='maximum depth for evaluation', default=80)

# Multi-gpu training
parser.add_argument('--num_threads',                        type=int,   help='number of threads to use for data loading', default=1)
parser.add_argument('--world_size',                         type=int,   help='number of nodes for distributed training', default=1)
parser.add_argument('--rank',                               type=int,   help='node rank for distributed training', default=0)
parser.add_argument('--dist_url',                           type=str,   help='url used to set up distributed training', default='tcp://127.0.0.1:1234')
parser.add_argument('--dist_backend',                       type=str,   help='distributed backend', default='nccl')
parser.add_argument('--gpu',                                type=int,   help='GPU id to use.', default=None)
parser.add_argument('--multiprocessing_distributed',                    help='Use multi-processing distributed training to launch '
                                                                             'N processes per node, which has N GPUs. This is the '
                                                                             'fastest way to use PyTorch for either single node or '
                                                                             'multi node data parallel training', action='store_true',)



if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

def set_misc(model):
    if 'resne' in args.encoder:
        fixing_layers = ['base_model.conv1', '.bn']
    else:
        fixing_layers = ['conv0', 'norm']
    print("Fixing first conv layer")

    for name, child in model.named_children():
        if not 'encoder' in name:
            continue
        for name2, parameters in child.named_parameters():
            # print(name, name2)
            if any(x in name2 for x in fixing_layers):
                parameters.requires_grad = False

def block_print():
    sys.stdout = open(os.devnull, 'w')

def enable_print():
    sys.stdout = sys.__stdout__


eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'sq_rel', 'log_rms', 'mae', 'd1', 'd2', 'd3']


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    mae = np.mean(np.abs(gt - pred))

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, mae, d1, d2, d3]

def online_eval(model, dataloader_eval, gpu, ngpus):
    if gpu is not None:
        eval_measures = torch.zeros(11).cuda(device=gpu)
    else:
        eval_measures = torch.zeros(11).to(args.device)
    for _, eval_sample_batched in enumerate(tqdm(dataloader_eval.data)):
        with torch.no_grad():
            if gpu is not None:
                image = eval_sample_batched['image'].cuda(device=gpu, non_blocking=True)
                gt_depth = eval_sample_batched['depth'].cuda(device=gpu, non_blocking=True)
                radar_channels = eval_sample_batched['radar_channels'].cuda(device=gpu, non_blocking=True)
                radar_points = eval_sample_batched['radar_points'].cuda(device=gpu, non_blocking=True)
                focal = eval_sample_batched['focal'].cuda(device=gpu, non_blocking=True)
                lidar_points = eval_sample_batched['lidar_points'].cuda(device=gpu, non_blocking=True)
                if args.norm_point:
                    centroid = eval_sample_batched['centroid'].cuda(device=gpu, non_blocking=True)
                    furthest_distance = eval_sample_batched['furthest_distance'].cuda(device=gpu, non_blocking=True)
                else:
                    centroid = None
                    furthest_distance = None
                K = eval_sample_batched['K'].cuda(device=gpu, non_blocking=True)

            else:
                image = eval_sample_batched['image'].to(args.device)
                gt_depth = eval_sample_batched['depth'].to(args.device)
                radar_channels = eval_sample_batched['radar_channels'].to(args.device)
                radar_points = eval_sample_batched['radar_points'].to(args.device)
                focal = eval_sample_batched['focal'].to(args.device)
                lidar_points = eval_sample_batched['lidar_points'].to(args.device)

                if args.norm_point:
                    centroid = eval_sample_batched['centroid'].to(args.device)
                    furthest_distance = eval_sample_batched['furthest_distance'].to(args.device)
                else:
                    centroid = None
                    furthest_distance = None
                K = eval_sample_batched['K'].to(args.device)

            _, _, _, _, pred_depth, _ = model(image, radar_channels, radar_points, focal, K, centroid, furthest_distance)

            pred_depth = pred_depth.cpu().numpy().squeeze()
            gt_depth = gt_depth.cpu().numpy().squeeze()

        pred_depth[pred_depth < args.min_depth_eval] = args.min_depth_eval
        pred_depth[pred_depth > args.max_depth_eval] = args.max_depth_eval
        pred_depth[np.isinf(pred_depth)] = args.max_depth_eval
        pred_depth[np.isnan(pred_depth)] = args.min_depth_eval

        valid_mask = np.logical_and(gt_depth > args.min_depth_eval, gt_depth < args.max_depth_eval)

        measures = compute_errors(gt_depth[valid_mask], pred_depth[valid_mask])

        eval_measures[:-1] += torch.tensor(measures).cuda()
        eval_measures[-1] += 1

    if args.multiprocessing_distributed:
        group = dist.new_group([i for i in range(ngpus)])
        dist.all_reduce(tensor=eval_measures, op=dist.ReduceOp.SUM, group=group)

    if not args.multiprocessing_distributed or gpu == 0:
        eval_measures_cpu = eval_measures.cpu()
        cnt = eval_measures_cpu[-1].item()
        eval_measures_cpu /= cnt
        print('Computing errors for {} eval samples'.format(int(cnt)))
        print("{:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}, {:>7}".format('silog', 'abs_rel', 'log10', 'rms',
                                                                                        'sq_rel', 'log_rms', 'mae', 'd1', 'd2',
                                                                                        'd3'))
        for i in range(9):
            print('{:7.3f}, '.format(eval_measures_cpu[i]), end='')
        print('{:7.3f}'.format(eval_measures_cpu[9]))
        return eval_measures_cpu

    return None

def main_worker(gpu, ngpus_per_node, args):
    
    args.gpu = gpu

    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)

    dataloader = GCNRADDataLoader(args, 'train')
    dataloader_eval = GCNRADDataLoader(args, 'test')

    model = GET_UP(args)

    model.train()
    model.decoder.apply(weights_init_xavier)

    set_misc(model)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in model.parameters() if p.requires_grad])
    print("Total number of learning parameters: {}".format(num_params_update))

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.device == 'cuda':
        model = torch.nn.DataParallel(model)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model)
        model.to(args.device)
    
    if args.distributed:
        print("Model Initialized on GPU: {}".format(args.gpu))
    else:
        print("Model Initialized")

    global_step = 0
    best_eval_measures_lower_better = torch.zeros(7).cpu() + 1e3
    best_eval_measures_higher_better = torch.zeros(3).cpu()
    best_eval_steps = np.zeros(10, dtype=np.int32)

    optimizer = torch.optim.AdamW([{'params': model.module.image_encoder.parameters(), 'weight_decay': args.weight_decay},
                                {'params': model.module.radar_encoder.parameters(), 'weight_decay': args.weight_decay},
                                {'params': model.module.decoder.parameters(), 'weight_decay': 0}],
                                lr=args.learning_rate, eps=args.adam_eps)

    model_just_loaded = False
    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda: {}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
                
            # checkpoint = torch.load(args.checkpoint_path)

            global_step = checkpoint['global_step']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            try:
                best_eval_measures_higher_better = checkpoint['best_eval_measures_higher_better'].cpu()
                best_eval_measures_lower_better = checkpoint['best_eval_measures_lower_better'].cpu()
                best_eval_steps = checkpoint['best_eval_steps']
            except KeyError:
                print("Could not load values for online evaluation")

            print("Loaded checkpoint '{}' (global_step {})".format(args.checkpoint_path, checkpoint['global_step']))
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
        model_just_loaded = True
    
    if args.retrain:
        global_step = 0

    cudnn.benchmark = True

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(args.log_directory + '/' + args.model_name + '/summaries', flush_secs=30)
        if args.do_online_eval:
            if args.eval_summary_directory != '':
                eval_summary_path = os.path.join(args.eval_summary_directory, args.model_name)
            else:
                eval_summary_path = os.path.join(args.log_directory, 'eval')
            eval_summary_writer = SummaryWriter(eval_summary_path, flush_secs=30)

    # loss functions
    if args.reg_loss == 'l1':
        l_depth = l1_loss()
    elif args.reg_loss == 'l2':
        l_depth = l2_loss()
    elif args.reg_loss == 'smoothl1':
        l_depth = smoothl1_loss()
    else:
        print('Not support yet.')   
    smoothness = imgrad_loss()

    start_time = time.time()
    duration = 0

    num_log_images = args.batch_size
    end_learning_rate = args.end_learning_rate if args.end_learning_rate != -1 else 0.1 * args.learning_rate

    steps_per_epoch = len(dataloader.data)
    num_total_steps = args.num_epochs * steps_per_epoch
    start_epoch = global_step // steps_per_epoch

    for epoch in range(start_epoch, args.num_epochs + 1):
        if args.distributed:
            dataloader.train_sampler.set_epoch(epoch)

        with tqdm(dataloader.data, unit='batch') as tepoch:
            for sample_batched in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()
                before_op_time = time.time()

                if args.device == 'cuda':
                    image = sample_batched['image'].cuda(args.gpu, non_blocking=True)
                    depth_gt = sample_batched['depth'].cuda(args.gpu, non_blocking=True)
                    nointer_depth_gt = sample_batched['nointer_depth'].cuda(args.gpu, non_blocking=True)
                    radar_channels = sample_batched['radar_channels'].cuda(args.gpu, non_blocking=True)
                    radar_points = sample_batched['radar_points'].cuda(args.gpu, non_blocking=True)
                    focal = sample_batched['focal'].cuda(args.gpu, non_blocking=True)
                    lidar_points = sample_batched['lidar_points'].cuda(args.gpu, non_blocking=True)
                    if args.norm_point:
                        centroid = sample_batched['centroid'].cuda(args.gpu, non_blocking=True)
                        furthest_distance = sample_batched['furthest_distance'].cuda(args.gpu, non_blocking=True)
                    else:
                        centroid = None
                        furthest_distance = None
                    K = sample_batched['K'].cuda(args.gpu, non_blocking=True)

                else:
                    image = sample_batched['image'].to(args.device)
                    depth_gt = sample_batched['depth'].to(args.device)
                    nointer_depth_gt = sample_batched['nointer_depth'].to(args.device)
                    radar_channels = sample_batched['radar_channels'].to(args.device)
                    radar_points = sample_batched['radar_points'].to(args.device)
                    focal = sample_batched['focal'].to(args.device)
                    lidar_points = sample_batched['lidar_points'].to(args.device)
                    if args.norm_point:
                        centroid = sample_batched['centroid'].to(args.device)
                        furthest_distance = sample_batched['furthest_distance'].to(args.device)
                    else:
                        centroid = None
                        furthest_distance = None
                    K = sample_batched['K'].to(args.device)

                lpg8x8, lpg4x4, lpg2x2, reduc1x1, depth_est, up_pc = model(image, radar_channels, radar_points, focal, K,\
                                                                        centroid, furthest_distance)
                
                # calculate loss for pc upsampling
                loss_chamfer, _ = chamfer_distance(up_pc.permute(0, 2, 1), lidar_points[..., :3])
                loss_chamfer = loss_chamfer * args.w_chamfer

                # calculate loss for depth
                mask = depth_gt > 0.001
                nointer_mask = nointer_depth_gt > 0.001

                loss_depth = l_depth.forward(depth_est, depth_gt, mask.to(torch.bool))
                loss_nointer_depth = l_depth.forward(depth_est, nointer_depth_gt, nointer_mask.to(torch.bool)) * args.w_nointer_depth

                if args.w_smoothness > 0.00:
                    loss_smoothness = smoothness.forward(depth_est, image)
                    loss_smoothness = loss_smoothness * args.w_smoothness
                else:
                    loss_smoothness = 0.0
                
                loss = loss_depth + loss_smoothness + loss_chamfer + loss_nointer_depth
                loss.backward() 

                for param_group in optimizer.param_groups:
                    current_lr = (args.learning_rate - end_learning_rate) * (1 - global_step / num_total_steps) ** 0.9 + end_learning_rate
                    param_group['lr'] = current_lr
                
                optimizer.step()
                
                if not args.do_online_eval and global_step and global_step % args.save_freq == 0:
                    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                        checkpoint = {'global_step': global_step,
                                      'model': model.state_dict(),
                                      'optimizer': optimizer.state_dict()}
                        torch.save(checkpoint, args.log_directory + '/' + args.model_name + '/model-{}'.format(global_step))

                if args.do_online_eval and global_step and global_step % args.eval_freq == 0 and not model_just_loaded:
                    time.sleep(0.1)
                    model.eval()
                    eval_measures = online_eval(model, dataloader_eval, gpu, ngpus_per_node)

                    if eval_measures is not None:
                        for i in range(10):
                            eval_summary_writer.add_scalar(eval_metrics[i], eval_measures[i].cpu(), int(global_step))
                            measure = eval_measures[i]
                            is_best = False
                            if i < 7 and measure < best_eval_measures_lower_better[i]:
                                old_best = best_eval_measures_lower_better[i].item()
                                best_eval_measures_lower_better[i] = measure.item()
                                is_best = True
                            elif i >= 7 and measure > best_eval_measures_higher_better[i-7]:
                                old_best = best_eval_measures_higher_better[i-7].item()
                                best_eval_measures_higher_better[i-7] = measure.item()
                                is_best = True
                            if is_best:
                                old_best_step = best_eval_steps[i]
                                old_best_name = '/model-{}-best_{}_{:.5f}'.format(old_best_step, eval_metrics[i], old_best)
                                model_path = args.log_directory + '/' + args.model_name + old_best_name
                                if os.path.exists(model_path):
                                    command = 'rm {}'.format(model_path)
                                    os.system(command)
                                best_eval_steps[i] = global_step
                                model_save_name = '/model-{}-best_{}_{:.5f}'.format(global_step, eval_metrics[i], measure)
                                print('New best for {}. Saving model: {}'.format(eval_metrics[i], model_save_name))
                                checkpoint = {'global_step': global_step,
                                            'model': model.state_dict(),
                                            'optimizer': optimizer.state_dict(),
                                            'best_eval_measures_higher_better': best_eval_measures_higher_better,
                                            'best_eval_measures_lower_better': best_eval_measures_lower_better,
                                            'best_eval_steps': best_eval_steps
                                            }
                                torch.save(checkpoint, args.log_directory + '/' + args.model_name + model_save_name)
                        eval_summary_writer.flush()
                    model.train()
                    block_print()
                    set_misc(model)
                    enable_print()

                model_just_loaded = False
                global_step += 1

                tepoch.set_postfix(loss=loss.item(), l_chamfer=loss_chamfer.item(), l_depth=loss_depth.item(), l_nointer=loss_nointer_depth.item())

                
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        writer.close()
        if args.do_online_eval:
            eval_summary_writer.close()


def main():
    if args.mode != 'train':
        print('main.py is only for training. Use test.py instead.')
        return -1
    runtime = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    args.model_name = runtime + '_' + args.model_name

    torch.cuda.empty_cache()
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node > 1 and not args.multiprocessing_distributed:
        print("This machine has more than 1 gpu. Please specify --multiprocessing_distributed, or set \'CUDA_VISIBLE_DEVICES=0\'")
        return -1

    if args.do_online_eval:
        print("You have specified --do_online_eval.")
        print("This will evaluate the model every eval_freq {} steps and save best models for individual eval metrics."
              .format(args.eval_freq))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device


    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)

if __name__ == '__main__':
    main()