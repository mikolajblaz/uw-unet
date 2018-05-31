from unet import UnetTrainer

trainer = UnetTrainer(optimizer='momentum')
trainer.visualize_predictions()
