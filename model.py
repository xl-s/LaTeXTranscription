import torch
from torch import nn, optim
from transforms import clean_transform
from utils import load_data
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from PIL import Image

labels = pd.read_csv("meta.csv").to_dict()["formula"]

class SingleCharacterClassifier(nn.Module):
	def __init__(self):
		super(SingleCharacterClassifier, self).__init__()
		# Architecture: 3x Conv layers => Max Pooling => 3x FC layers
		self.layers = nn.Sequential(
			nn.Conv2d(1, 5, 5),
			nn.Conv2d(5, 10, 5),
			nn.Conv2d(10, 15, 5),
			nn.MaxPool2d(5),
			nn.Flatten(),
			nn.LeakyReLU(),
			nn.Linear(15 * 15 * 15, 2048),
			nn.LeakyReLU(),
			nn.Linear(2048, 1024),
			nn.LeakyReLU(),
			nn.Linear(1024, 405)
		)

	def forward(self, x):
		for layer in self.layers:
			x = layer(x)
		return x

	def predict(self, loc, is_image=False):
		self.eval()
		device = "cuda" if next(self.parameters()).is_cuda else "cpu"
		img = Image.open(loc) if is_image else loc
		img_t = clean_transform(img).to(device)
		out = self.forward(torch.unsqueeze(img_t, 0))
		return nn.functional.softmax(out, dim=1)[0]

	def classify(self, loc, top=5):
		prediction = self.predict(loc)
		_, indices = torch.sort(prediction, descending=True)
		return [(labels[ind.item()], prediction[ind.item()].item()) for ind in indices[:top]]


def train(model, device, train_loader, optimizer, epoch):
	model.train()
	loss_fn = nn.CrossEntropyLoss()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		pred = model(data)
		loss = loss_fn(pred, target)
		loss.backward()
		optimizer.step()

		if batch_idx % 100 == 0:
			print("Epoch:", epoch, "loss:", round(loss.item(), 5))

def test(model, device, test_loader, plot=False):
	model.eval()

	correct = 0
	exampleSet = False
	example_data = np.zeros([10, 90, 90])
	example_pred = np.zeros(10)
	example_target = np.zeros(10)

	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)

			pred = torch.argmax(model(data), dim=1)
			correct += sum((pred - target) == 0).item()

			if not exampleSet:
				for i in range(10):
					example_data[i] = data[i][0].to("cpu").numpy()
					example_pred[i] = pred[i].to("cpu").numpy()
					example_target[i] = target[i].to("cpu").numpy()
					exampleSet = True

	print('Test set accuracy: ', round(100. * correct / len(test_loader.dataset), 3), '%')

	if not plot: return
	for i in range(10):
		plt.subplot(2,5,i+1)
		data = (example_data[i] - example_data[i].min()) * (example_data[i].max() - example_data[i].min())
		plt.imshow(data, cmap='gray', interpolation='none')
		corr = int(example_pred[i]) == int(example_target[i])
		plt.title(labels[int(example_pred[i])] + (" ✔" if corr else " ✖"))
		plt.xticks([])
		plt.yticks([])
	plt.show()

def load_model(load, device):
	model = SingleCharacterClassifier().to(device)
	if load: model.load_state_dict(torch.load(load))
	return model

def run(N_EPOCH, L_RATE, load=None, save=None):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_loader = load_data(replications=100)
	test_loader = load_data(replications=1)

	model = load_model(load, device)
	optimizer = optim.Adam(model.parameters(), lr=L_RATE)

	for epoch in range(N_EPOCH):
		test(model, device, test_loader)
		train(model, device, train_loader, optimizer, epoch + 1)

	test(model, device, test_loader)

	if save: torch.save(model.state_dict(), save)

