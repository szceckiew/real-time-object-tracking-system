model_dir = 'D:/intelliJ/pycharm_projects/mobilenet/training_ssd-mobilenet-v2-fpnlite-320/'

# TensorBoard initialization
from tensorboard import program
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', model_dir])
tb.main()
