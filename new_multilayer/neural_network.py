import matplotlib.pyplot as plt

from neuron import Initialisation, Model, Loss, Gradients, Update, Predict

class NeuralNetwork:
	def __init__(self, X_train, y_train, n1, learning_rate, n_iter):
		self.X_train = X_train
		self.y_train = y_train
		self.n1 = n1
		self.learning_rate = learning_rate
		self.n_iter = n_iter

	def neural_network(self):
		# Initialize weights and bias
		n0 = self.X_train.shape[0]
		n2 = self.y_train.shape[0]
		parameters = Initialisation(n0, self.n1, n2).initialisation()

		train_loss = []
		train_acc = []

		for i in range(self.n_iter):
			# Forward propagation
			activations = Model(self.X_train, parameters).forward_propagation()
			# Back propagation
			gradients = Gradients(self.X_train, self.y_train, activations, parameters).back_propagation()
			# Update weights and bias
			parameters = Update(gradients, parameters, self.learning_rate).update()

			if i % 10 == 0:
				# Compute loss
				train_loss.append(Loss(A, self.y_train).log_loss())
				# Compute accuracy
				y_pred = Predict(self.X_train, W, b).predict()
				current_accuracy = accuracy_score(self.y_train, y_pred)
				train_acc.append(current_accuracy)

		plt.figure(figsize=(14, 4))

		plt.subplot(1, 2, 1)
		plt.plot(train_loss, label='Training Loss')
		plt.legend()

		plt.subplot(1, 2, 2)
		plt.plot(train_acc, label='Training Accuracy')
		plt.legend()

		plt.show()

		return (W, b)
