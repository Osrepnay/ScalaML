package io.github.osrepnay.scalaml.mlp.train

import breeze.linalg.{DenseMatrix, DenseVector}
import io.github.osrepnay.scalaml.mlp.{Activation, Layer, MLP, MLPState}

import scala.annotation.tailrec

/** An object for training a network using stochastic gradient descent. */

object SGD {

	/** Train a network for a specified number of epochs.
	 *
	 * @param mlp          The network to train
	 * @param inputs       A list of inputs to test
	 * @param expected     The expected results for the inputs
	 * @param learningRate The learning rate of the network
	 * @param epochs       The number of epochs to train for on the inputs
	 * @param batchSize    The size of each batch
	 * @return The trained network
	 */

	@tailrec
	def train(mlp: MLP, inputs: Array[Array[Double]], expected: Array[Array[Double]], learningRate: Double,
		epochs: Int,
		batchSize: Int): MLP = {
		if(epochs == 0) {
			mlp
		} else {
			val newMLP = train(mlp, inputs, expected, learningRate, batchSize)
			train(newMLP, inputs, expected, learningRate, epochs - 1, batchSize)
		}
	}

	/** Train a network.
	 *
	 * @param mlp          The network to train
	 * @param inputs       A list of inputs to test
	 * @param expected     The expected results for the inputs
	 * @param learningRate The learning rate of the network
	 * @param batchSize    The size of each batch
	 * @return The trained network
	 */

	@tailrec
	def train(mlp: MLP, inputs: Array[Array[Double]], expected: Array[Array[Double]], learningRate: Double,
		batchSize: Int): MLP = {

		def backprop(inputs: Array[DenseVector[Double]], expected: Array[DenseVector[Double]]): MLP = {
			val changes = inputs.indices.map(idx => backpropChanges(inputs(idx), expected(idx))).unzip
			val changesAvg = (
				changes._1.head.indices
					.map(layerIdx => changes._1
						.map(_ (layerIdx))
						.reduce(_ + _) /:/ changes._1.length.toDouble),
				changes._2.head.indices
					.map(layerIdx => changes._2
						.map(_ (layerIdx))
						.reduce(_ + _) /:/ changes._2.length.toDouble)
			)
			val newWeights = changesAvg._1.indices
				.map(layerIdx => mlp.layers(layerIdx).weights - changesAvg._1(layerIdx)).toArray
			val newBiases = changesAvg._2.indices
				.map(layerIdx => mlp.layers(layerIdx).biases - changesAvg._2(layerIdx)).toArray
			MLP(newWeights.indices
				.map(layerIdx => Layer(newWeights(layerIdx), newBiases(layerIdx), mlp.layers(layerIdx).activation))
				.toArray,
				mlp.loss)
		}

		def backpropChanges(inputs: DenseVector[Double], expected: DenseVector[Double]): (Array[DenseMatrix[Double]],
			Array[DenseVector[Double]]) = {
			val zs = mlp.feedForwardWithStateNoActivations(inputs)
			val as = MLPState(zs.nodes.head +: zs.nodes.indices.tail.map(idx => mlp.layers(idx - 1).activation(zs
				.nodes(idx)))
				.toArray)
			val outputErrors =
				if(mlp.layers.last.activation == Activation.SOFTMAX) {
					mlp.layers.last.activation.derivativeMat(zs.last) * mlp.loss.derivative(as.last, expected)
				} else {
					mlp.loss.derivative(as.last, expected) * mlp.layers.last.activation.derivative(zs.last)
				}

			def calcPrevErrors(errorsFront: DenseVector[Double], idx: Int): Array[DenseVector[Double]] = {
				if(idx <= 0) {
					Array.empty
				} else {
					val errors =
						if(mlp.layers(idx).activation == Activation.SOFTMAX) {
							mlp.layers(idx).activation.derivativeMat(zs(idx)) * (mlp.layers(idx).weights * errorsFront)
						} else {
							mlp.layers(idx).activation.derivative(zs(idx)) * (mlp.layers(idx).weights * errorsFront)
						}
					calcPrevErrors(errors, idx - 1) :+ errors
				}
			}

			val allErrors = (calcPrevErrors(outputErrors, mlp.layers.length - 1) :+ outputErrors).map(_.map(_ *
				learningRate))
			(
				mlp.layers.indices.map(idx => DenseMatrix.tabulate(mlp.layers(idx).weights.rows, mlp.layers(idx)
					.weights.cols) {
					case (in, out) => as(idx)(in) * allErrors(idx)(out)
				}).toArray,
				allErrors
			)
		}

		val batch = (inputs.take(batchSize).map(input => DenseVector(input)), expected.take(batchSize).map
		(expected => DenseVector(expected)))
		if(inputs.length <= batchSize) {
			backprop(batch._1, batch._2)
		} else {
			SGD.train(backprop(batch._1, batch._2), inputs.drop(batchSize), expected.drop(batchSize), learningRate,
				batchSize)
		}
	}

}
