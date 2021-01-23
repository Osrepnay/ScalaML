package io.github.osrepnay.scalaml.mlp

import breeze.linalg.DenseVector

/** A multilayer perceptron.
 *
 * @constructor Create a new untrained MLP
 * @param layers An array of layers in the network
 * @param loss   The loss function to use
 */

case class MLP(layers: Array[Layer], loss: Loss) {

	/** Feed forward an input through the network.
	 *
	 * @param inputs The inputs to feed through
	 * @return The outputs
	 */

	def feedForward(inputs: DenseVector[Double]): DenseVector[Double] = {
		if(layers.length == 1) {
			layers.head.feed(inputs)
		} else {
			MLP(layers.tail, loss).feedForward(layers.head.feed(inputs))
		}
	}

	/** Feed forward an input through the network.
	 *
	 * @param inputs The inputs to feed through
	 * @return The outputs
	 */

	def feedForward(inputs: Array[Double]): Array[Double] = {
		feedForward(DenseVector(inputs)).toArray
	}

	/** Feed forward an input through the network and calculate the loss.
	 *
	 * @param inputs   The inputs to feed through
	 * @param expected The expected outputs
	 * @return A tuple of the outputs and the loss
	 */

	def feedForward(inputs: DenseVector[Double], expected: DenseVector[Double]): (DenseVector[Double], Double) = {
		if(layers.length == 1) {
			val output = layers.head.feed(inputs)
			(output, loss(output, expected))
		} else {
			MLP(layers.tail, loss).feedForward(layers.head.feed(inputs), expected)
		}
	}

	/** Feed forward an input through the network and calculate the loss.
	 *
	 * @param inputs   The inputs to feed through
	 * @param expected The expected outputs
	 * @return A tuple of the outputs and the loss
	 */

	def feedForward(inputs: Array[Double], expected: Array[Double]): (Array[Double], Double) = {
		val result = feedForward(DenseVector(inputs), DenseVector(expected))
		(result._1.toArray, result._2)
	}

	/** Feed forward an input through the network.
	 *
	 * @param inputs The inputs to feed through
	 * @return The state of the network
	 */

	def feedForwardWithState(inputs: DenseVector[Double]): MLPState = {
		if(layers.length == 1) {
			MLPState(Array(inputs, layers.head.feed(inputs)))
		} else {
			val nextLayers = MLP(layers.tail, loss).feedForwardWithState(layers.head.feed
			(inputs))
			nextLayers.prependLayer(inputs)
		}
	}

	/** Feed forward an input through the network.
	 *
	 * @param inputs The inputs to feed through
	 * @return The state of the network without activations
	 */

	def feedForwardWithStateNoActivations(inputs: DenseVector[Double], first: Boolean = true): MLPState = {
		if(layers.length == 1) {
			val newInputs = layers.head.activation(inputs)
			MLPState(Array(inputs, layers.head.feedNoActivation(newInputs)))
		} else {
			if(first) {
				val nextLayers = MLP(layers.tail, loss)
					.feedForwardWithStateNoActivations(layers.head.feedNoActivation(inputs), first = false)
				nextLayers.prependLayer(inputs)
			} else {
				val newInputs = layers.head.activation(inputs)
				val nextLayers = MLP(layers.tail, loss)
					.feedForwardWithStateNoActivations(layers.head.feedNoActivation(newInputs), first = false)
				nextLayers.prependLayer(inputs)
			}
		}
	}

}
