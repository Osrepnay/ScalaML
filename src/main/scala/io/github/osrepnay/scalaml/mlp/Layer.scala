package io.github.osrepnay.scalaml.mlp

import breeze.linalg.{DenseMatrix, DenseVector}

import scala.util.Random

/** A single layer of a network.
 *
 * @constructor Create a new layer
 * @param weights    The weights in the layer
 * @param biases     The biases in the layer
 * @param activation The activation function to use in the layer
 */

case class Layer(weights: DenseMatrix[Double], biases: DenseVector[Double], activation: Activation) {

	/** Feed an input through the layer.
	 *
	 * @param inputs The inputs to feed
	 * @return The outputs
	 */

	def feed(inputs: DenseVector[Double]): DenseVector[Double] = {
		activation(feedNoActivation(inputs))
	}

	/** Feed an input through the network without activations.
	 *
	 * @param inputs The inputs to feed
	 * @return The outputs
	 */

	def feedNoActivation(inputs: DenseVector[Double]): DenseVector[Double] = {
		(weights.t * inputs) + biases
	}

}

/** Factory for randomized layers. */

object Layer {

	/** Creates a new randomized layer using a Gaussian distribution.
	 *
	 * @param inputNodes       The number of inputs to the layer
	 * @param outputNodes      The number of outputs to the layer
	 * @param activation       The activation function for the layer
	 * @param weightMultiplier The value to multiply the random weights and biases by
	 * @return A new randomized layer
	 */

	def apply(inputNodes: Int, outputNodes: Int, activation: Activation, weightMultiplier: Double): Layer = {
		val rand = new Random()
		Layer(DenseMatrix.tabulate(inputNodes, outputNodes) {case (_, _) => rand.nextGaussian() * weightMultiplier},
			DenseVector.tabulate(outputNodes) {_ => rand.nextGaussian() * weightMultiplier}, activation)
	}

}
