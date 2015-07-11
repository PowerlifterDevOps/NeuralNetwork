#ifndef BOOST_NUMERIC_UBLAS_VECTOR_INCLUDE_HPP
#define BOOST_NUMERIC_UBLAS_VECTOR_INCLUDE_HPP

#pragma once

#pragma warning(push)
#pragma warning(disable:4996)
#pragma warning(disable:4127)
#pragma warning(disable:4702)
#include "boost\numeric\ublas\vector.hpp"
#pragma warning(pop)

#include <vector>

typedef boost::numeric::ublas::vector<unsigned int> IntVec;
typedef boost::numeric::ublas::vector<double> DoubleVec;
typedef boost::numeric::ublas::zero_vector<double> ZeroVector;
typedef std::vector<DoubleVec> DoubleVecVec;

#endif