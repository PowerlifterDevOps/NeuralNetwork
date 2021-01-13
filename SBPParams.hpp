#ifndef SBP_PARAMS_HPP
#define SBP_PARAMS_HPP

#pragma once

#include "MemLeakDetection.hpp"
#include "TrainingTuple.hpp"

#include "boost_numeric_ublas_vector_include.hpp"

#include <string>
#include <fstream>
#include <iostream>

class SBPParams
{
public:
	friend void Swap(SBPParams & first, SBPParams & second)
	{
		using std::swap;
		swap(first.learningRate, second.learningRate);
		swap(first.momentumCoefficient, second.momentumCoefficient);
		swap(first.decayCoefficient, second.decayCoefficient);
		swap(first.errorThreshold, second.errorThreshold);
		swap(first.numIterations, second.numIterations);
		swap(first.numEpochs, second.numEpochs);
		swap(first.ttVec, second.ttVec);
	}

	SBPParams();
	SBPParams(SBPParams const & copyMe);
	SBPParams(SBPParams && moveMe);

	~SBPParams();

	SBPParams & operator=(SBPParams rhs);

	void LoadTrainingTuplesFromFile(std::string fileName);

	double getLR() const;
	double getMC() const;
	double getDC() const;
	double getErrorThreshold() const;
	unsigned int getNumIterations() const;
	unsigned int getNumEpochs() const;
	unsigned int getNumTrainingTuples() const;
	TrainingTuple getTrainingTuple(unsigned int tupleId) const;

	bool setLR(double newLR);
	bool setMC(double newMC);
	bool setDC(double newDC);
	bool setErrorThreshold(double newErrorThreshold);
	bool setNumIterations(unsigned int newNumIterations);
	bool setNumEpochs(unsigned int newNumEpochs);
	bool setTrainingTuple(unsigned int tupleId, TrainingTuple newTuple);
private:
	double learningRate;
	double momentumCoefficient;
	double decayCoefficient;
	double errorThreshold;
	unsigned int numIterations;
	unsigned int numEpochs;
	std::vector<TrainingTuple> ttVec;

	unsigned int countTokensInString(char * inStr) const;
	void massageTuples();
};

SBPParams::SBPParams()
: learningRate(0.1)
, momentumCoefficient(1.0) // no momentum
, decayCoefficient(1.0) // no decay
, errorThreshold(0.1)
, numIterations(1000)
, numEpochs(1)
{
}

SBPParams::SBPParams(SBPParams const & copyMe)
: learningRate(copyMe.learningRate)
, momentumCoefficient(copyMe.momentumCoefficient) // no momentum
, decayCoefficient(copyMe.decayCoefficient) // no decay
, errorThreshold(copyMe.errorThreshold)
, numIterations(copyMe.numIterations)
, numEpochs(copyMe.numEpochs)
{
	ttVec.clear();
	ttVec.reserve(copyMe.getNumTrainingTuples());
	for (unsigned int i = 0; i < copyMe.getNumTrainingTuples(); ++i)
	{
		ttVec.push_back(copyMe.getTrainingTuple(i));
	}
}

SBPParams::SBPParams(SBPParams && moveMe)
{
	Swap(*this, moveMe);
}

SBPParams::~SBPParams()
{
}

SBPParams & SBPParams::operator=(SBPParams rhs)
{
	Swap(*this, rhs);
	return *this;
}

void SBPParams::LoadTrainingTuplesFromFile(std::string fileName)
{
	std::fstream aFile;
	aFile.open(fileName, std::ios_base::in);

	// check if file is open
	if (aFile.is_open())
	{
		unsigned int sizeReadBuffer = 32768;
		char readBuffer[32768];
		char* token1 = NULL;
		char* context1 = NULL;
		char* token2 = NULL;
		char* context2 = NULL;
		char delim1[] = "|";
		char delim2[] = ",";

		// get first tuple
		aFile.getline(readBuffer, sizeReadBuffer);
		while (!aFile.eof())
		{
			bool input = false;
			bool calcSizeInput = true;
			bool calcSizeOutput = true;
			int vecSize = 0;

			TrainingTuple newTuple;
			// get input portion
			token1 = strtok_s(readBuffer, delim1, &context1);
			if (calcSizeInput)
			{
				vecSize = countTokensInString(token1);
				calcSizeInput = false;
			}
			input = true;
			while (token1 != NULL)
			{
				DoubleVec tokenVec(vecSize);
				tokenVec.resize(vecSize);
				// get first input token
				token2 = strtok_s(token1, delim2, &context2);
				unsigned int ctr = 0;
				while (token2 != NULL)
				{
					double token2Dbl = atof(token2);
					tokenVec[ctr++] = token2Dbl;
					token2 = strtok_s(NULL, delim2, &context2);
				}
				if (input)
				{
					newTuple.setInputVec(tokenVec);
				}
				else
				{
					newTuple.setOutputVec(tokenVec);
				}
				// get output portion
				token1 = strtok_s(NULL, delim1, &context1);
				if (calcSizeOutput)
				{
					vecSize = countTokensInString(token1);
					calcSizeOutput = false;
				}
				input = false;
			}

			// get next tuple
			ttVec.push_back(newTuple);
			aFile.getline(readBuffer, sizeReadBuffer);
		}
	}
	aFile.close();

	massageTuples();
}

double SBPParams::getLR() const
{
	return learningRate;
}

double SBPParams::getMC() const
{
	return momentumCoefficient;
}

double SBPParams::getDC() const
{
	return decayCoefficient;
}

double SBPParams::getErrorThreshold() const
{
	return errorThreshold;
}

unsigned int SBPParams::getNumIterations() const
{
	return numIterations;
}

unsigned int SBPParams::getNumEpochs() const
{
	return numEpochs;
}

unsigned int SBPParams::getNumTrainingTuples() const
{
	return ttVec.size();
}

TrainingTuple SBPParams::getTrainingTuple(unsigned int tupleId) const
{
	if (tupleId >= 0 && tupleId < ttVec.size())
	{
		return ttVec[tupleId];
	}
	TrainingTuple retVal;
	return retVal;
}

bool SBPParams::setLR(double newLR)
{
	if (newLR > 0.0 && newLR < 1.0)
	{
		learningRate = newLR;
		return true;
	}
	return false;
}

bool SBPParams::setMC(double newMC)
{
	if (newMC >= 0.0 && newMC <= 1.0)
	{
		momentumCoefficient = newMC;
		return true;
	}
	return false;
}

bool SBPParams::setDC(double newDC)
{
	if (newDC >= 0.0 && newDC <= 1.0)
	{
		decayCoefficient = newDC;
		return true;
	}
	return false;
}

bool SBPParams::setErrorThreshold(double newErrorThreshold)
{
	if (newErrorThreshold >= 0.0)
	{
		errorThreshold = newErrorThreshold;
	}
	return false;
}

bool SBPParams::setNumIterations(unsigned int newNumIterations)
{
	if (newNumIterations > 0)
	{
		numIterations = newNumIterations;
		return true;
	}
	return false;
}

bool SBPParams::setNumEpochs(unsigned int newNumEpochs)
{
	if (newNumEpochs > 0)
	{
		numEpochs = newNumEpochs;
		return true;
	}
	return false;
}

bool SBPParams::setTrainingTuple(unsigned int tupleId, TrainingTuple newTuple)
{
	if (tupleId >= 0 && tupleId < ttVec.size())
	{
		ttVec[tupleId] = newTuple;
		return true;
	}
	return false;
}

unsigned int SBPParams::countTokensInString(char * inStr) const
{
	unsigned int ctr = 0;
	char * ptr = strchr(inStr, ',');
	while (ptr != NULL)
	{
		++ctr;
		ptr = strchr(ptr + 1, ',');
	}
	return ++ctr;
}

void SBPParams::massageTuples()
{
	unsigned int sizeInput = ttVec[0].getInput().size();
	DoubleVec minVec(sizeInput);
	DoubleVec maxVec(sizeInput);
	for (unsigned int i = 0; i < sizeInput; ++i)
	{
		minVec(i) = 999999.99;
		maxVec(i) = -999999.99;
	}

	unsigned int numTuples = ttVec.size();
	for (unsigned int i = 0; i < numTuples; ++i)
	{
		for(unsigned int j = 0; j < sizeInput; ++j)
		{
			if (ttVec[i].getInput()(j) < minVec(j))
			{
				minVec(j) = ttVec[i].getInput()(j);
			}
			if (ttVec[i].getInput()(j) > maxVec(j))
			{
				maxVec(j) = ttVec[i].getInput()(j);
			}
			if (minVec(j) == maxVec(j))
			{
				minVec(j) = -1.0;
				maxVec(j) = 1.0;
			}
		}
	}

	for (unsigned int i = 0; i < numTuples; ++i)
	{
		DoubleVec newInput = ttVec[i].getInput();
		for (unsigned int j = 0; j < sizeInput; ++j)
		{
			double newVal = -1.0 + 2.0*((newInput(j) - minVec(j)) / (maxVec(j) - minVec(j)));
			newInput(j) = newVal;
		}
		ttVec[i].setInputVec(newInput);
	}
}

#endif