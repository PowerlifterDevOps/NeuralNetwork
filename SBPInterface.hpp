#ifndef SBP_INTERFACE_HPP
#define SBP_INTERFACE_HPP

#pragma once

#include "boost_numeric_ublas_vector_include.hpp"
#include "boost_numeric_ublas_matrix_include.hpp"

class SBPInterface
{
public:
	virtual DoubleVec feedForward(DoubleVec input) = 0;
	virtual DoubleVec calcActivation(DoubleVec input) const = 0;
	virtual DoubleVec calcActivationDx(DoubleVec input) const = 0;
	virtual DoubleVec getNetFromLastActivation(unsigned int layerId) const = 0;
	virtual unsigned int getNumLayers() const = 0;
	virtual EdgeMatrix getEdgeMatrix(unsigned int layerId) const = 0;
	virtual DoubleVec getBiasVector(unsigned int layerId) const = 0;
	virtual void applyEdgeUpdates(EdgeMatrixVec emVec, DoubleVecVec dvVec) = 0;
	virtual void applyWeightDecay(double decayCoefficient) = 0;
	virtual void reInitialize() = 0;
private:
};

#endif