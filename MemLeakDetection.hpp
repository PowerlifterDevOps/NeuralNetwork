#ifndef MEM_LEAK_DETECTION_HPP
#define MEM_LEAK_DETECTION_HPP

#pragma once

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)

#ifdef _DEBUG
	#define new DEBUG_NEW
#endif

#endif