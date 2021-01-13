#ifndef NEURAL_NETWORK_PARAMETERS_HPP
#define NEURAL_NETWORK_PARAMETERS_HPP

#pragma once

#include "MemLeakDetection.hpp"
#include "RandomNumberGenerator.hpp"

#include "boost_numeric_ublas_vector_include.hpp"

class NeuralNetworkParameters
{
public:
	friend void Swap(NeuralNetworkParameters & first, NeuralNetworkParameters & second)
	{
		using std::swap;
		swap(first.neuronsPerLayer, second.neuronsPerLayer);
		swap(first.coeffA, second.coeffA);
		swap(first.coeffB, second.coeffB);
		swap(first.rng, second.rng);
	}

	NeuralNetworkParameters();
	NeuralNetworkParameters(NeuralNetworkParameters const & copyMe);
	NeuralNetworkParameters(NeuralNetworkParameters && moveMe);

	~NeuralNetworkParameters();

	NeuralNetworkParameters & operator=(NeuralNetworkParameters rhs);

	unsigned int getNumLayers() const;
	unsigned int getNumNeurons(unsigned int layerId) const;
	double getA() const;
	double getB() const;
	RandomNumberGenerator getRNG() const;
	 
	bool setNumLayers(unsigned int newNumLayers);
	bool setNumNeurons(unsigned int layerId, unsigned int numNeurons);
	bool setA(double newA);
	bool setB(double newB);
	bool setRNG(RandomNumberGenerator newRNG);
private:
	IntVec neuronsPerLayer;
	double coeffA;
	double coeffB;
	RandomNumberGenerator rng;
};

NeuralNetworkParameters::NeuralNetworkParameters()
: coeffA(1.716)
, coeffB(0.667)
{
}

NeuralNetworkParameters::NeuralNetworkParameters(NeuralNetworkParameters const & copyMe)
: coeffA(copyMe.coeffA)
, coeffB(copyMe.coeffB)
{
	neuronsPerLayer.clear();
	neuronsPerLayer.resize(copyMe.getNumLayers());
	for (unsigned int i = 0; i < copyMe.getNumLayers(); ++i)
	{
		neuronsPerLayer[i] = copyMe.getNumNeurons(i);
	}
}

NeuralNetworkParameters::NeuralNetworkParameters(NeuralNetworkParameters && moveMe)
{
	Swap(*this, moveMe);
}

NeuralNetworkParameters::~NeuralNetworkParameters()
{
	neuronsPerLayer.clear();
}

NeuralNetworkParameters & NeuralNetworkParameters::operator=(NeuralNetworkParameters rhs)
{
	Swap(*this, rhs);
	return *this;
}

unsigned int NeuralNetworkParameters::getNumLayers() const
{
	return neuronsPerLayer.size();
}

unsigned int NeuralNetworkParameters::getNumNeurons(unsigned int layerId) const
{
	if (layerId >= 0 && layerId < neuronsPerLayer.size())
	{
		return neuronsPerLayer[layerId];
	}
	return 0;
}

double NeuralNetworkParameters::getA() const
{
	return coeffA;
}

double NeuralNetworkParameters::getB() const
{
	return coeffB;
}

RandomNumberGenerator NeuralNetworkParameters::getRNG() const
{
	return rng;
}

bool NeuralNetworkParameters::setNumLayers(unsigned int newNumLayers)
{
	// minimum 3 layers
	if (newNumLayers > 2)
	{
		neuronsPerLayer.clear();
		neuronsPerLayer.resize(newNumLayers);
		for (unsigned int i = 0; i < newNumLayers; ++i)
		{
			neuronsPerLayer[i] = 0;
		}
		return true;
	}
	return false;
}

bool NeuralNetworkParameters::setNumNeurons(unsigned int layerId, unsigned int numNeurons)
{
	// minimum 1 neuron per layer
	if (layerId >= 0 && layerId < neuronsPerLayer.size() && numNeurons > 0)
	{
		neuronsPerLayer[layerId] = numNeurons;
		return true;
	}
	return false;
}

bool NeuralNetworkParameters::setA(double newA)
{
	if (newA > 0.0)
	{
		coeffA = newA;
		return true;
	}
	return false;
}

bool NeuralNetworkParameters::setB(double newB)
{
	if (newB > 0.0)
	{
		coeffB = newB;
		return true;
	}
	return false;
}

bool NeuralNetworkParameters::setRNG(RandomNumberGenerator newRNG)
{
	rng = newRNG;
	return true;
}

#endif