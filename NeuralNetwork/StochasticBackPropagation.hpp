#ifndef STOCHASTIC_BACK_PROPAGATION_HPP
#define STOCHASTIC_BACK_PROPAGATION_HPP

#pragma once

#include "MemLeakDetection.hpp"
#include "SBPInterface.hpp"
#include "SBPParams.hpp"
#include "RandomNumberGenerator.hpp"

#include "boost_numeric_ublas_vector_include.hpp"
#include "boost_numeric_ublas_matrix_include.hpp"
#include "boost\numeric\ublas\io.hpp"

#include "boost\shared_ptr.hpp"

class StochasticBackPropagation
{
public:
	static double doSBP(SBPParams const & sbpp, boost::shared_ptr<SBPInterface> sbpObj, unsigned int seedVal);
private:
	static double calcError(SBPParams const & sbpp, boost::shared_ptr<SBPInterface> sbpObj);
	static DoubleVec calcDeltaK(DoubleVec expectedOut, DoubleVec actualOut, DoubleVec actK);
	static DoubleVec calcDeltaJ(EdgeMatrix emPrev, DoubleVec actDx, DoubleVec deltaPrev);
	static EdgeMatrix calcDeltaKJ(DoubleVec lrDeltaK, DoubleVec act, unsigned int sizeJ, unsigned int sizeK);
};

double StochasticBackPropagation::doSBP(SBPParams const & sbpp, boost::shared_ptr<SBPInterface> sbpObj, unsigned int seedVal)
{
	using namespace boost::numeric::ublas;

	RandomNumberGenerator rng(seedVal);

	double errorRate = 999999.0;

	// for each epoch
	for (unsigned int epoch = 0; epoch < sbpp.getNumEpochs(); ++epoch)
	{
		EdgeMatrixVec momentumEmVec;
		DoubleVecVec momentumDvVec;
		bool firstIteration = true;

		sbpObj->reInitialize();

		// for each iteration
		for (unsigned int iteration = 0; iteration < sbpp.getNumIterations(); ++iteration)
		{
			// get random training tuple for this iteration
			int ttId = rng.getNextInt(0, sbpp.getNumTrainingTuples()-1);
			TrainingTuple tt = sbpp.getTrainingTuple(ttId);

			// feed forward
			DoubleVec actualOutput = sbpObj->feedForward(tt.getInput());

			EdgeMatrixVec emVec;
			DoubleVecVec dvVec;

			// get signed int so --l doesn't wrap to 2^32
			int numLayersOfEdges = static_cast<int>(sbpObj->getNumLayers() - 1);

			DoubleVec delta;
			DoubleVec deltaPrev;
			unsigned int sizeI, sizeJ, sizeK = 0;
			DoubleVec net, act, actDx;

			// for each layer of edges
			for (int l = numLayersOfEdges-1; l >= 0; --l)
			{
				if (l == (numLayersOfEdges-1))
				{
					// calc deltaK
					net = sbpObj->getNetFromLastActivation(l + 1);
					actDx = sbpObj->calcActivationDx(net);
					delta = calcDeltaK(tt.getOutput(), actualOutput, actDx);

					// calc deltaKJ
					DoubleVec lrDeltaK = sbpp.getLR() * delta;
					EdgeMatrix em = sbpObj->getEdgeMatrix(l);
					sizeJ = em.size1();
					sizeK = em.size2();
					net = sbpObj->getNetFromLastActivation(l);
					act = sbpObj->calcActivation(net);
					EdgeMatrix deltaKJ = calcDeltaKJ(lrDeltaK, act, sizeJ, sizeK);
					emVec.push_back(deltaKJ);

					// calc deltaKBias
					dvVec.push_back(lrDeltaK);

					deltaPrev = delta;
				}
				else
				{
					// calc deltaJ
					EdgeMatrix emPrev = sbpObj->getEdgeMatrix(l + 1);
					net = sbpObj->getNetFromLastActivation(l + 1);
					actDx = sbpObj->calcActivationDx(net);
					delta = calcDeltaJ(emPrev, actDx, deltaPrev);

					// calc deltaJI
					DoubleVec lrDeltaJ = sbpp.getLR() * delta;
					EdgeMatrix em = sbpObj->getEdgeMatrix(l);
					sizeI = em.size1();
					sizeJ = em.size2();
					EdgeMatrix deltaJI;
					if (l > 0)
					{
						net = sbpObj->getNetFromLastActivation(l);
						act = sbpObj->calcActivation(net);
						deltaJI = calcDeltaKJ(lrDeltaJ, act, sizeI, sizeJ);
					}
					else
					{
						net = sbpObj->getNetFromLastActivation(l);
						deltaJI = calcDeltaKJ(lrDeltaJ, net, sizeI, sizeJ);
					}
					emVec.push_back(deltaJI);

					// calc deltaJBias
					dvVec.push_back(lrDeltaJ);

					deltaPrev = delta;
				}
			}

			// apply momentum
			if (!firstIteration)
			{
				unsigned int sizeEmVec = emVec.size();
				for (unsigned int i = 0; i < sizeEmVec; ++i)
				{
					emVec[i] = sbpp.getMC() * emVec[i] + (1.0 - sbpp.getMC()) * momentumEmVec[i];
					dvVec[i] = sbpp.getMC() * dvVec[i] + (1.0 - sbpp.getMC()) * momentumDvVec[i];
				}
				firstIteration = false;
			}

			// apply weight updates
			sbpObj->applyEdgeUpdates(emVec, dvVec);

			sbpObj->applyWeightDecay(sbpp.getDC());

			momentumEmVec = emVec;
			momentumDvVec = dvVec;
		}

		// calc error
		errorRate = calcError(sbpp, sbpObj);

		if (errorRate < sbpp.getErrorThreshold())
		{
			// sufficiently trained, break
			break;
		}
	}
	return errorRate;
}

double StochasticBackPropagation::calcError(SBPParams const & sbpp, boost::shared_ptr<SBPInterface> sbpObj)
{
	unsigned int numTrainingTuples = sbpp.getNumTrainingTuples();
	double sum = 0.0;
	for (unsigned int i = 0; i < numTrainingTuples; ++i)
	{
		TrainingTuple tt = sbpp.getTrainingTuple(i);
		DoubleVec actualOutput = sbpObj->feedForward(tt.getInput());
		DoubleVec diff = (tt.getOutput() - actualOutput);
		unsigned int numElements = diff.size();
		for (unsigned int j = 0; j < numElements; ++j)
		{
			sum += 0.5 * std::pow(diff(j), 2.0);
		}
	}
	return sum;
}

DoubleVec StochasticBackPropagation::calcDeltaK(DoubleVec expectedOut, DoubleVec actualOut, DoubleVec actDx)
{
	DoubleVec delta = expectedOut - actualOut;
	for (unsigned int k = 0; k < delta.size(); ++k)
	{
		delta(k) = delta(k) * actDx(k);
	}
	return delta;
}

DoubleVec StochasticBackPropagation::calcDeltaJ(EdgeMatrix emPrev, DoubleVec actDx, DoubleVec deltaPrev)
{
	unsigned int sizeJ = emPrev.size1();
	unsigned int sizeK = emPrev.size2();
	DoubleVec delta(actDx.size());
	for (unsigned int j = 0; j < sizeJ; ++j)
	{
		delta(j) = actDx(j);
		double sum = 0.0;
		for (unsigned int k = 0; k < sizeK; ++k)
		{
			sum += emPrev(j, k) * deltaPrev(k);
		}
		delta(j) *= sum;
	}
	return delta;
}

EdgeMatrix StochasticBackPropagation::calcDeltaKJ(DoubleVec lrDeltaK, DoubleVec act, unsigned int sizeJ, unsigned int sizeK)
{
	EdgeMatrix deltaKJ(sizeJ, sizeK);
	for (unsigned int k = 0; k < sizeK; ++k)
	{
		for (unsigned int j = 0; j < sizeJ; ++j)
		{
			deltaKJ(j, k) = lrDeltaK(k) * act(j);
		}
	}
	return deltaKJ;
}

#endif