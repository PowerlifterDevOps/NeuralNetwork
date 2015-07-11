#ifndef BOOST_NUMERIC_UBLAS_MATRIX_INCLUDE_HPP
#define BOOST_NUMERIC_UBLAS_MATRIX_INCLUDE_HPP

#pragma once

#pragma warning(push)
#pragma warning(disable:4702)
#include "boost\numeric\ublas\matrix.hpp"
#pragma warning(pop)

#include <vector>

typedef boost::numeric::ublas::matrix<double> EdgeMatrix;
typedef boost::numeric::ublas::zero_matrix<double> ZeroMatrix;
typedef boost::numeric::ublas::identity_matrix<double> IdentityMatrix;
typedef std::vector<EdgeMatrix> EdgeMatrixVec;

#endif