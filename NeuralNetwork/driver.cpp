#include "MemLeakDetection.hpp"
#include "NeuralNetworkParameters.hpp"
#include "NeuralNetwork.hpp"
#include "StochasticBackPropagation.hpp"
#include "TrainingTuple.hpp"

#include "boost\numeric\ublas\io.hpp"
#include "boost\shared_ptr.hpp"

#include <fstream>

#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

int main()
{
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_CHECK_ALWAYS_DF | _CRTDBG_LEAK_CHECK_DF);

	NeuralNetworkParameters nnp;
	nnp.setA(1.716);
	nnp.setB(0.667);
	nnp.setNumLayers(4);
	nnp.setNumNeurons(0, 324);	
	nnp.setNumNeurons(1, 24);
	nnp.setNumNeurons(2, 24);
	nnp.setNumNeurons(3, 6);
	RandomNumberGenerator rng(GetTickCount());
	nnp.setRNG(rng);
	
	boost::shared_ptr<NeuralNetwork> nnPtr(new NeuralNetwork());
	nnPtr->initNeuralNetwork(nnp);

	SBPParams sbpp;
	sbpp.setDC(1.0);
	sbpp.setLR(0.0123);
	sbpp.setMC(0.95);
	sbpp.setErrorThreshold(1);
	sbpp.setNumEpochs(1);
	sbpp.setNumIterations(600000);
	sbpp.LoadTrainingTuplesFromFile("..\\NeuralNetwork\\TrainingData\\RC100_7TrainingData.txt");
	double errorRate = StochasticBackPropagation::doSBP(sbpp, nnPtr, GetTickCount());

	std::fstream outFile;
	outFile.open("..\\NeuralNetwork\\derp.txt", std::ios_base::trunc | std::ios_base::out);

	std::cout << "Error Rate <" << errorRate << ">" << std::endl;
	outFile << "Error Rate <" << errorRate << ">" << std::endl;

	outFile << "after training" << std::endl;
	for (unsigned int i = 0; i < sbpp.getNumTrainingTuples(); ++i)
	{
		DoubleVec actualOut = nnPtr->feedForward(sbpp.getTrainingTuple(i).getInput());
		outFile << "ff <" << sbpp.getTrainingTuple(i).getInput() << "> == <" << actualOut << std::endl;
	}

	outFile.close();
	
	return 0;
}