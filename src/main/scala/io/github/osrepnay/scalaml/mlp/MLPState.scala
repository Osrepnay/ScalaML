package io.github.osrepnay.scalaml.mlp

import breeze.linalg.DenseVector


/** The values of each node in a network.
 *
 * @param nodes The nodes in the network
 */

case class MLPState(nodes: Array[DenseVector[Double]]) {

	/** Get the nodes in a layer
	 *
	 * @param layer The layer to get
	 * @return The nodes in the layer
	 */

	def apply(layer: Int): DenseVector[Double] = {
		nodes(layer)
	}

	/** Creates a new MLPState containing a new layer at the end
	 *
	 * @param layer The layer to append
	 * @return A new MLPState with the layer appended
	 */

	def appendLayer(layer: DenseVector[Double]): MLPState = {
		MLPState(nodes :+ layer)
	}

	/** Creates a new MLPState containing a new layer at the beginning
	 *
	 * @param layer The layer to prepend
	 * @return A new MLPState with the layer prepended
	 */

	def prependLayer(layer: DenseVector[Double]): MLPState = {
		MLPState(layer +: nodes)
	}

	/** Get the layer at the start of the network
	 *
	 * @return The layer at the start
	 */

	def head: DenseVector[Double] = nodes.head

	/** Get the layer at the end of the network
	 *
	 * @return The layer at the end
	 */

	def last: DenseVector[Double] = nodes.last

}