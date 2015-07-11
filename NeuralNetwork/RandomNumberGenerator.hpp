#ifndef RANDOM_NUMBER_GENERATOR_HPP
#define RANDOM_NUMBER_GENERATOR_HPP

#pragma once

#include "MemLeakDetection.hpp"

#include "boost_random_mersenne_twister_include.hpp"

#include "boost\shared_ptr.hpp"
#include "boost\random\uniform_real_distribution.hpp"
#include "boost\random\uniform_int_distribution.hpp"

typedef boost::shared_ptr<boost::random::mt19937> RngPtr;

class RandomNumberGenerator
{
public:
	friend void Swap(RandomNumberGenerator & first, RandomNumberGenerator & second)
	{
		using std::swap;
		swap(first.rng, second.rng);
	}

	RandomNumberGenerator();
	RandomNumberGenerator(unsigned int seedVal);
	RandomNumberGenerator(RandomNumberGenerator const & copyMe);
	RandomNumberGenerator(RandomNumberGenerator && moveMe);

	~RandomNumberGenerator();

	RandomNumberGenerator & operator=(RandomNumberGenerator rhs);

	double getNextDouble(double minVal, double maxVal);
	int getNextInt(int minVal, int maxVal);
private:
	static const unsigned int defaultRNGSeed = 17;
	static const unsigned int iSize = 1048576;
	RngPtr rng;

	bool initRNGSpace();
};

RandomNumberGenerator::RandomNumberGenerator()
{
	rng = RngPtr(new boost::random::mt19937(defaultRNGSeed));
	initRNGSpace();
}

RandomNumberGenerator::RandomNumberGenerator(unsigned int seedVal)
{
	rng = RngPtr(new boost::random::mt19937(seedVal));
	initRNGSpace();
}

RandomNumberGenerator::RandomNumberGenerator(RandomNumberGenerator const & copyMe)
: rng(copyMe.rng)
{
}

RandomNumberGenerator::RandomNumberGenerator(RandomNumberGenerator && moveMe)
{
	Swap(*this, moveMe);
}

RandomNumberGenerator::~RandomNumberGenerator()
{
}

RandomNumberGenerator & RandomNumberGenerator::operator=(RandomNumberGenerator rhs)
{
	Swap(*this, rhs);
	return *this;
}

double RandomNumberGenerator::getNextDouble(double minVal, double maxVal)
{
	boost::random::uniform_real_distribution<double> realDist(minVal, maxVal);
	double retVal = realDist(*rng);
	return retVal;
}

int RandomNumberGenerator::getNextInt(int minVal, int maxVal)
{
	boost::random::uniform_int_distribution<int> intDist(minVal, maxVal);
	int retVal = intDist(*rng);
	return retVal;
}

bool RandomNumberGenerator::initRNGSpace()
{
	boost::random::uniform_real_distribution<double> realDist(-100.0, 100.0);
	for (unsigned int i = 0; i < iSize; ++i)
	{
		realDist(*rng);
	}
	return true;
}

#endif