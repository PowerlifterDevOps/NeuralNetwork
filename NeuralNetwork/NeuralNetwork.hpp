#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#pragma once

#include "MemLeakDetection.hpp"
#include "SBPInterface.hpp"
#include "NeuralNetworkParameters.hpp"

#include <math.h>

class NeuralNetwork : public SBPInterface
{
public:
	friend void Swap(NeuralNetwork & first, NeuralNetwork & second)
	{
		using std::swap;
		swap(first.nnParams, second.nnParams);
		swap(first.edgeMatrices, second.edgeMatrices);
		swap(first.biasVectors, second.biasVectors);
		swap(first.netsFromLastFeedForward, second.netsFromLastFeedForward);
		swap(first.bias, second.bias);
	}

	NeuralNetwork();
	NeuralNetwork(NeuralNetwork const & copyMe);
	NeuralNetwork(NeuralNetwork && moveMe);

	~NeuralNetwork();

	NeuralNetwork & operator=(NeuralNetwork rhs);

	void initNeuralNetwork(NeuralNetworkParameters newParams);
	void initEdgeMatrix(EdgeMatrix & em) const;
	void initEdgeVector(DoubleVec & dv) const;
	DoubleVec feedForward(DoubleVec input);
	DoubleVec calcActivation(DoubleVec input) const;
	DoubleVec calcActivationDx(DoubleVec input) const;
	DoubleVec getNetFromLastActivation(unsigned int layerId) const;
	unsigned int getNumLayers() const;
	void applyEdgeUpdates(EdgeMatrixVec emVec, DoubleVecVec dvVec);
	void applyWeightDecay(double decayCoefficient);
	void reInitialize();

	EdgeMatrix getEdgeMatrix(unsigned int layerId) const;
	DoubleVec getBiasVector(unsigned int layerId) const;

	bool setEdgeMatrix(unsigned int layerId, EdgeMatrix newMatrix);
	bool setBiasMatrix(unsigned int layerId, DoubleVec newVector);
private:
	NeuralNetworkParameters nnParams;
	EdgeMatrixVec edgeMatrices;
	DoubleVecVec biasVectors;
	DoubleVecVec netsFromLastFeedForward;
	double bias;

	double calcActivation(double input) const;
	double calcActivationDx(double input) const;
};

NeuralNetwork::NeuralNetwork()
: bias(1.0)
{
}

NeuralNetwork::NeuralNetwork(NeuralNetwork const & copyMe)
: nnParams(copyMe.nnParams)
, bias(copyMe.bias)
{
	edgeMatrices.clear();
	biasVectors.clear();
	for (unsigned int i = 0; i < nnParams.getNumLayers() - 1; ++i)
	{
		edgeMatrices.push_back(copyMe.getEdgeMatrix(i));
		biasVectors.push_back(copyMe.getBiasVector(i));
	}

	netsFromLastFeedForward.clear();
	for (unsigned int i = 1; i < nnParams.getNumLayers(); ++i)
	{
		DoubleVec temp;
		temp.resize(nnParams.getNumNeurons(i));
		netsFromLastFeedForward.push_back(temp);
	}
}

NeuralNetwork::NeuralNetwork(NeuralNetwork && moveMe)
{
	Swap(*this, moveMe);
}

NeuralNetwork::~NeuralNetwork()
{
	edgeMatrices.clear();
	biasVectors.clear();
	netsFromLastFeedForward.clear();
}

NeuralNetwork & NeuralNetwork::operator=(NeuralNetwork rhs)
{
	Swap(*this, rhs);
	return *this;
}

// network has a bias neuron with value of 1.0 connected to all hidden and output neurons
void NeuralNetwork::initNeuralNetwork(NeuralNetworkParameters newParams)
{
	nnParams = newParams;
	unsigned int numEdgeMatrices = nnParams.getNumLayers() - 1; // one less layer of edges than neurons
	for (unsigned int i = 0; i < numEdgeMatrices; ++i)
	{
		EdgeMatrix foo(nnParams.getNumNeurons(i), nnParams.getNumNeurons(i+1));
		initEdgeMatrix(foo);
		edgeMatrices.push_back(foo);

		DoubleVec bar(nnParams.getNumNeurons(i + 1));
		initEdgeVector(bar);
		biasVectors.push_back(bar);
	}
}

void NeuralNetwork::initEdgeMatrix(EdgeMatrix & em) const
{
	unsigned int numEdges = em.size1();
	double min = -1 / sqrt(numEdges);
	double max = 1 / sqrt(numEdges);
	for (unsigned int i = 0; i < em.size1(); ++i)
	{
		for (unsigned int j = 0; j < em.size2(); ++j)
		{
			double temp = nnParams.getRNG().getNextDouble(min, max);
			em(i, j) = temp;
		}
	}
}

void NeuralNetwork::initEdgeVector(DoubleVec & dv) const
{
	unsigned int numEdges = dv.size();
	double min = -1 / sqrt(numEdges);
	double max = 1 / sqrt(numEdges);
	for (unsigned int i = 0; i < dv.size(); ++i)
	{
		double temp = nnParams.getRNG().getNextDouble(min, max);
		dv(i) = temp;
	}
}

DoubleVec NeuralNetwork::feedForward(DoubleVec input)
{
	using namespace boost::numeric::ublas;

	netsFromLastFeedForward.clear();
	netsFromLastFeedForward.push_back(input);

	DoubleVec temp = prod(input, edgeMatrices[0]) + (bias * biasVectors[0]);
	netsFromLastFeedForward.push_back(temp);

	for (unsigned int i = 1; i < nnParams.getNumLayers() - 1; ++i)
	{
		temp = calcActivation(temp);
		temp = prod(temp, edgeMatrices[i]) + (bias * biasVectors[i]);
		netsFromLastFeedForward.push_back(temp);
	}

	temp = calcActivation(temp);
	return temp;
}

DoubleVec NeuralNetwork::calcActivation(DoubleVec input) const
{
	DoubleVec retVal;
	retVal.resize(input.size());
	for (unsigned int i = 0; i < input.size(); ++i)
	{
		retVal(i) = calcActivation(input(i));
	}
	return retVal;
}

DoubleVec NeuralNetwork::calcActivationDx(DoubleVec input) const
{
	DoubleVec retVal;
	retVal.resize(input.size());
	for (unsigned int i = 0; i < input.size(); ++i)
	{
		retVal(i) = calcActivationDx(input(i));
	}
	return retVal;
}

DoubleVec NeuralNetwork::getNetFromLastActivation(unsigned int layerId) const
{
	if (layerId >= 0 && layerId < netsFromLastFeedForward.size())
	{
		return netsFromLastFeedForward[layerId];
	}
	ZeroVector retVal;
	return retVal;
}

unsigned int NeuralNetwork::getNumLayers() const
{
	return nnParams.getNumLayers();
}

void NeuralNetwork::applyEdgeUpdates(EdgeMatrixVec emVec, DoubleVecVec dvVec)
{
	unsigned int sizeEmVec = emVec.size();
	unsigned int sizeDvVec = dvVec.size();
	if (sizeEmVec != sizeDvVec)
	{
		int x = 0;
		++x;
	}
	for (unsigned int l = 0; l < sizeEmVec; ++l)
	{
		edgeMatrices[sizeEmVec -1 - l] += emVec[l];
		biasVectors[sizeEmVec -1 - l] += dvVec[l];
		int x = 0;
		++x;
	}
}

void NeuralNetwork::applyWeightDecay(double decayCoefficient)
{
	unsigned int numEM = edgeMatrices.size();
	for (unsigned int i = 0; i < numEM; ++i)
	{
		unsigned int sizeIn = edgeMatrices[i].size1();
		unsigned int sizeOut = edgeMatrices[i].size2();
		for (unsigned int j = 0; j < sizeIn; ++j)
		{
			for (unsigned int k = 0; k < sizeOut; ++k)
			{
				edgeMatrices[i](j, k) *= decayCoefficient;
			}
		}
		unsigned int sizeBias = biasVectors[i].size();
		for (unsigned int j = 0; j < sizeBias; ++j)
		{
			biasVectors[i](j) *= decayCoefficient;
		}
	}
}

void NeuralNetwork::reInitialize()
{
	initNeuralNetwork(nnParams);
}

EdgeMatrix NeuralNetwork::getEdgeMatrix(unsigned int layerId) const
{
	if (layerId >= 0 && layerId < edgeMatrices.size())
	{
		return edgeMatrices[layerId];
	}
	ZeroMatrix retVal;
	return retVal;
}

DoubleVec NeuralNetwork::getBiasVector(unsigned int layerId) const
{
	if (layerId >= 0 && layerId < biasVectors.size())
	{
		return biasVectors[layerId];
	}
	ZeroVector retVal;
	return retVal;
}

bool NeuralNetwork::setEdgeMatrix(unsigned int layerId, EdgeMatrix newMatrix)
{
	if (layerId >= 0 && layerId < edgeMatrices.size())
	{
		edgeMatrices[layerId] = newMatrix;
		return true;
	}
	return false;
}

bool NeuralNetwork::setBiasMatrix(unsigned int layerId, DoubleVec newVector)
{
	if (layerId >= 0 && layerId < biasVectors.size())
	{
		biasVectors[layerId] = newVector;
		return true;
	}
	return false;
}

double NeuralNetwork::calcActivation(double input) const
{
	return nnParams.getA() * std::tanh(nnParams.getB()*input);
}

double NeuralNetwork::calcActivationDx(double input) const
{
	return 1 - pow(tanh(input), 2.0);
}

#endif