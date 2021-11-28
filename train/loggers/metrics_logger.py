from pytorch_lightning import Callback

class MetricsLogger(Callback):
	
	def __init__(self):
		pass
	
	def on_validation_batch_end(trainer, module, outputs, ...):
		vacc = outputs['val_acc'] # you can access them here
		self.collection.append(vacc) # track them

	def on_validation_epoch_end(trainer, module):
		pass

	
