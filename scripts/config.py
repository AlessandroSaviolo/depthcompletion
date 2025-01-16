import argparse

def parse_args():
    print('Parsing arguments ...')
    parser = argparse.ArgumentParser(description='Arguments for depth completion')

    parser.add_argument('--keep_existing',         type=bool,     default=True)
    parser.add_argument('--da_input_width',        type=int,      default=320)
    parser.add_argument('--da_input_height',       type=int,      default=240)
    parser.add_argument('--da_encoder_type',       type=str,      default='vits')
    parser.add_argument('--da_seed',               type=int,      default=17)
    parser.add_argument('--da_precision',          type=int,      default=32)
    parser.add_argument('--min_depth',             type=float,    default=0.1)
    parser.add_argument('--max_depth',             type=float,    default=10.0)


    # # training
    # parser.add_argument('-R', '--run_id',          type=int,      default=0)
    # parser.add_argument('-D', '--gpu_id',          type=int,      default=0)
    # parser.add_argument('-E', '--epochs',          type=int,      default=10)
    # parser.add_argument('-B', '--batch_size',      type=int,      default=8)
    # parser.add_argument('--warmup_lr',             type=float,    default=1e-2)
    # parser.add_argument('--cosine_lr',             type=float,    default=3e-4)
    # parser.add_argument('--warmup_steps',          type=int,      default=50)
    # parser.add_argument('--cosine_steps',          type=int,      default=1000000)
    # parser.add_argument('--gradient_clip_val',     type=float,    default=1.0)
    # parser.add_argument('--weight_decay',          type=float,    default=1e-2)
    # parser.add_argument('--optimizer_name',        type=str,      default='AdamW')
    # parser.add_argument('--adam_beta1',            type=float,    default=0.9)
    # parser.add_argument('--adam_beta2',            type=float,    default=0.999)
    # parser.add_argument('--adam_eps',              type=float,    default=1e-12)
    # parser.add_argument('--num_devices',           type=int,      default=1)
    # parser.add_argument('--num_workers',           type=int,      default=1)
    # parser.add_argument('--export_new_models',     type=str2bool, default=False)

    # # model
    # parser.add_argument('--da_version',            type=int,      default=2)
    # parser.add_argument('--num_components',        type=int,      default=16) # need to check this name

    # parser.add_argument('--weights_name',          type=str,      default='')
    # parser.add_argument('--feature_extractor',     type=str,      default='ResidualDownsampling')
    # parser.add_argument('--predictor',             type=str,      default='UpProj')
    # parser.add_argument('--dyn_hidden_sizes',      type=list,     default='48 24 24')
    # parser.add_argument('--dyn_dropout',           type=float,    default=0.0)

    # # losses
    # parser.add_argument('--reconstruction_loss',   type=str,      default='mse')
    # parser.add_argument('--distribution_loss',     type=str,      default='kld')
    # parser.add_argument('--epsilon',               type=float,    default=1e-7)
    # parser.add_argument('--huber_delta',           type=float,    default=1.0)
    # parser.add_argument('--silog_alpha',           type=float,    default=0.15)
    # parser.add_argument('--silog_beta',            type=float,    default=10)
    # parser.add_argument('--kl_mix',                type=int,      default=64)
    # parser.add_argument('--kl_min',                type=float,    default=0.01)
    # parser.add_argument('--kl_max',                type=float,    default=10.0)
    # parser.add_argument('--kl_temperature',        type=float,    default=0.75)

    # # logging
    # parser.add_argument('--logging',               type=int,      default=1)
    # parser.add_argument('--valid_frequency',       type=int,      default=1)

    # # data
    # parser.add_argument('--normalization',         type=str,      default='')
    # parser.add_argument('--shuffle',               type=str2bool, default=True)

    # # augmentations
    # parser.add_argument('--aug_motionblur',        type=str2bool, default=False)
    # parser.add_argument('--aug_artifacts',         type=str2bool, default=False)
    # parser.add_argument('--aug_jitter',            type=str2bool, default=False)
    # parser.add_argument('--aug_color',             type=str2bool, default=False)

    # # hdf5
    # parser.add_argument('-H', '--dataset_name',    type=str,      default='')
    # parser.add_argument('--input_window',          type=int,      default=4)
    # parser.add_argument('--unroll_window',         type=int,      default=1)
    # parser.add_argument('--time_stride',           type=int,      default=1)
    # parser.add_argument('--thrust_size',           type=int,      default=4)
    # parser.add_argument('--weight_samples',        type=str2bool, default=False)
    # parser.add_argument('--weight_normalized',     type=str2bool, default=False)

    # # images to predicted depths

    # parser.add_argument('--da_input_height',       type=int,      default=266)
    # parser.add_argument('--da_input_width',        type=int,      default=350)

    args = parser.parse_args()
    for arg in vars(args):
      value = getattr(args, arg)
      if isinstance(value, list):
        setattr(args, arg, [int(e) for e in ''.join(value).split()])
    return args

def save_args(args, file_path):
    print('Saving arguments ...')
    with open(file_path, 'w') as f:
      for arg in vars(args):
        arg_name = arg
        arg_type = type(getattr(args, arg)).__name__
        arg_value = str(getattr(args, arg))
        f.write(arg_name)
        f.write(';')
        f.write(arg_type)
        f.write(';')
        f.write(arg_value)
        f.write('\n')

def load_args(file_path):
    print('Loading arguments ...')
    parser = argparse.ArgumentParser(description='Arguments for depth completion')
    with open(file_path, 'r') as f:
      for arg in f.readlines():
        arg_name = arg.split(';')[0]
        arg_type = arg.split(';')[1]
        arg_value = arg.split(';')[2].replace('\n', '')
        if arg_type == 'str':
          parser.add_argument('--' + arg_name, type=str, default=arg_value)
        elif arg_type == 'bool':
          parser.add_argument('--' + arg_name, type=bool, default=arg_value)
        elif arg_type == 'int':
          parser.add_argument('--' + arg_name, type=int, default=arg_value)
        elif arg_type == 'float':
          parser.add_argument('--' + arg_name, type=float, default=arg_value)
        elif arg_type == 'list':
          arg_value = [int(e) for e in arg_value[1:-1].split(', ')]
          parser.add_argument('--' + arg_name, type=list, default=arg_value)
        elif arg_type == 'tuple':
          arg_value = [int(e) for e in arg_value[1:-1].split(', ')]
          parser.add_argument('--' + arg_name, type=tuple, default=arg_value)

    return parser.parse_args()

def log_args(args):
  print("Arguments")
  for arg, value in sorted(vars(args).items()):
    print(f"{arg}: {value}")

def assert_arg(arg, values):
  assert arg in values, (
    f'Invalid value for argument: {arg}. Choose among {values}'
  )

def str2bool(value):
  if isinstance(value, bool):
    return value
  if value.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif value.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected')

def size(value):
  try:
    return int(value)
  except ValueError:
    pass

  try:
    values = value.split(',')
    if len(values) != 2:
      raise ValueError
    return tuple(int(v) for v in values)
  except Exception:
    raise argparse.ArgumentTypeError(f'Invalid value: {value}. Must an integer or two integers separated by a comma!')
