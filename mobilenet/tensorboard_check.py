from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

logdir = 'D:/intelliJ/pycharm_projects/mobilenet/training/'

event_acc = EventAccumulator(logdir)
event_acc.Reload()

# Pobierz dostępne tagi
tags = event_acc.Tags()['scalars']
print("Dostępne metryki (tagi):", tags)
