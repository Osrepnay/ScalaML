package io.github.osrepnay.scalaml.mlp

import breeze.linalg.{DenseMatrix, DenseVector}

/** An activation function and its derivative. */

case class Activation(function: DenseVector[Double] => DenseVector[Double], derivative: DenseVector[Double]
	=> DenseVector[Double], derivativeMat: DenseVector[Double] => DenseMatrix[Double]) {

	/** Apply the activation function.
	 *
	 * @param nums The numbers to apply the activation function on
	 * @return The result after applying the activation function
	 */

	def apply(nums: DenseVector[Double]): DenseVector[Double] = function(nums)

}

/** The predefined activation functions. */

object Activation {

	/** The ReLU (Rectified Linear Unit) activation function. */

	final val RELU: Activation = Activation(breeze.numerics.relu(_), _.map(num => if(num > 0) 1d else 0d), _ =>
		DenseMatrix.zeros(0, 0))

	/** The Leaky ReLU activation function. Same as normal ReLu, except has a small gradient instead of 0. */

	final val LEAKY_RELU: Activation = Activation(_.map(num => if(num > 0) num else num / 0.01), _.map(num => if(num >
		0) 1d else 0.01), _ => DenseMatrix.zeros(0, 0))

	/** The sigmoid activation function. */

	final val SIGMOID: Activation = Activation(breeze.numerics.sigmoid(_), nums => {
		val sigmoided = breeze.numerics.sigmoid(nums)
		sigmoided *:* (1d - sigmoided)
	}, _ => DenseMatrix.zeros(0, 0))

	/** The tanh activation function. Like sigmoid, but centered at 0. */

	final val TANH: Activation = Activation(breeze.numerics.tanh(_), nums => 1d - breeze.numerics.pow(breeze.numerics
		.tanh(nums), 2), _ => DenseMatrix.zeros(0, 0))

	/** The softmax activation function. Usually used at the last layer of a network. */

	final val SOFTMAX: Activation = Activation(nums => {
		val sum = breeze.linalg.sum(breeze.numerics.exp(nums))
		nums.map(math.exp(_) / sum)
	}, nums => DenseVector.zeros(0), nums => {
		val softmaxed = {
			val sum = breeze.linalg.sum(breeze.numerics.exp(nums))
			nums.map(math.exp(_) / sum)
		}
		DenseMatrix.tabulate(softmaxed.length, softmaxed.length) {
			case (i, j) =>
				if(i == j) {
					softmaxed(i) * (1d - softmaxed(i))
				} else {
					-softmaxed(i) * softmaxed(j)
				}
		}
	})

}
