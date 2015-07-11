#ifndef TRAINING_TUPLE_HPP
#define TRAINING_TUPLE_HPP

#pragma once

#include "MemLeakDetection.hpp"

#include "boost_numeric_ublas_vector_include.hpp"

class TrainingTuple
{
public:
	friend void Swap(TrainingTuple & first, TrainingTuple & second)
	{
		using std::swap;
		swap(first.inputVec, second.inputVec);
		swap(first.outputVec, second.outputVec);
	}

	TrainingTuple();
	TrainingTuple(TrainingTuple const & copyMe);
	TrainingTuple(TrainingTuple && moveMe);

	~TrainingTuple();

	TrainingTuple & operator=(TrainingTuple rhs);

	DoubleVec getInput() const;
	DoubleVec getOutput() const;

	void setInputVec(DoubleVec newInput);
	void setOutputVec(DoubleVec newOutput);
private:
	DoubleVec inputVec;
	DoubleVec outputVec;
};

TrainingTuple::TrainingTuple()
{
}

TrainingTuple::TrainingTuple(TrainingTuple const & copyMe)
: inputVec(copyMe.inputVec)
, outputVec(copyMe.outputVec)
{
}

TrainingTuple::TrainingTuple(TrainingTuple && moveMe)
{
	Swap(*this, moveMe);
}

TrainingTuple::~TrainingTuple()
{
}

TrainingTuple & TrainingTuple::operator=(TrainingTuple rhs)
{
	Swap(*this, rhs);
	return *this;
}

DoubleVec TrainingTuple::getInput() const
{
	return inputVec;
}

DoubleVec TrainingTuple::getOutput() const
{
	return outputVec;
}

void TrainingTuple::setInputVec(DoubleVec newInputVec)
{
	inputVec = newInputVec;
}

void TrainingTuple::setOutputVec(DoubleVec newOutput)
{
	outputVec = newOutput;
}

#endif