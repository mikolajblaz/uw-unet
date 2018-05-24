from unet import UnetTrainer

trainer = UnetTrainer(optimizer='momentum')
trainer.train(1000)
