package io.github.osrepnay.scalaml.example

import io.github.osrepnay.scalaml.mlp.train.SGD
import io.github.osrepnay.scalaml.mlp.{Activation, Layer, Loss, MLP}

import scala.io.Source

object ExampleMLP {

	def main(args: Array[String]): Unit = {
		val bufferedSourceTr = Source.fromURL("https://pjreddie.com/media/files/mnist_train.csv")
		val trainDataIterator = bufferedSourceTr.getLines().take(10000).map {line =>
			val splitLine = line.split(",")
			(Array.fill(10)(0d).updated(splitLine.head.toInt, 1d), splitLine.tail.map(_.toDouble / 255))
		}
		val trainDataDuplicate = trainDataIterator.duplicate
		val trainData = (trainDataDuplicate._1.map(_._1).toArray, trainDataDuplicate._2.map(_._2).toArray)
		println("Done processing training data.")
		val mlp = MLP(
			Layer(784, 40, Activation.SIGMOID, 0.1) +:
				Layer(40, 20, Activation.SIGMOID, 0.1) +:
				Layer(20, 10, Activation.SIGMOID, 0.1) +: Array.empty,
			Loss.CEL
		)
		val startTime = System.currentTimeMillis()
		val trainedNetwork = SGD.train(mlp, trainData._2, trainData._1, 0.05, 50, 25)
		println("Done training network.")
		println(s"Time to train: ${System.currentTimeMillis() - startTime} ms")
		val bufferedSourceTe = Source.fromURL("https://pjreddie.com/media/files/mnist_train.csv")
		val testDataIterator = bufferedSourceTe.getLines().map {line =>
			val splitLine = line.split(",")
			(Array.fill(10)(0d).updated(splitLine.head.toInt, 1d), splitLine.tail.map(_.toDouble / 255))
		}
		val testDataDuplicate = testDataIterator.duplicate
		val testData = (testDataDuplicate._1.map(_._1).toArray, testDataDuplicate._2.map(_._2).toArray)
		println("Done processing training data.")
		val accuracy = (testData._1 zip testData._2).count {test =>
			val result = trainedNetwork.feedForward(test._2)
			result.indexOf(result.max) == test._1.indexOf(1)
		} / testData._1.length.toDouble * 100
		println(s"The network predicts correctly with $accuracy% accuracy.")
	}

}
