package io.github.osrepnay.scalaml.mlp

import breeze.linalg.DenseVector

/** A loss function and its derivative. */

case class Loss (loss: (DenseVector[Double], DenseVector[Double]) => Double, derivative: (DenseVector[Double],
	DenseVector[Double]) => DenseVector[Double]) {

	/** Apply the loss function.
	 *
	 * @param prediction The prediction
	 * @param expected   The expected prediction
	 * @return The loss of the prediction
	 */

	def apply(prediction: DenseVector[Double], expected: DenseVector[Double]): Double = loss(prediction, expected)

}

/** The predefined loss functions. */

object Loss {

	/** The mean squared error loss function. */

	final val MSE: Loss = Loss(
		(prediction, expected) => breeze.linalg.sum(breeze.numerics.pow(prediction - expected, 2)) / (prediction
			.length * 2),
		(prediction, expected) => prediction - expected
	)

	/** The cross-entropy loss function. */

	final val CEL: Loss = Loss(
		(prediction, expected) => -breeze.linalg.sum((expected *:* breeze.numerics.log(prediction)) +:+ ((1d -
			expected) * breeze.numerics.log(1d - prediction))),
		(prediction, expected) => ((1d - expected) /:/ (1d - prediction)) -:- (expected /:/ prediction)
	)

}
