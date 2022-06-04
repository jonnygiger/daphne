/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_RUNTIME_LOCAL_CONTEXT_DAPHNECONTEXT_H
#define SRC_RUNTIME_LOCAL_CONTEXT_DAPHNECONTEXT_H

#pragma once

#include <api/cli/DaphneUserConfig.h>

#include <vector>
#include <iostream>
#include <memory>
#include <chrono>

#include "IContext.h"

#ifdef USE_CUDA
    #include "CUDAContext.h"
#endif

// This macro is intended to be used in kernel function signatures, such that
// we can change the ubiquitous DaphneContext parameter in a single place, if
// required.
#define DCTX(varname) DaphneContext * varname

typedef std::chrono::duration<std::chrono::_V2::steady_clock::rep, std::chrono::_V2::steady_clock::period> DurationType;

struct Entry {
    std::chrono::time_point<std::chrono::_V2::steady_clock> Start;
    DurationType Duration;
    int Worker;
    int Round;
    int LocalTaskID;
};

/**
 * @brief This class carries all kinds of run-time context information.
 * 
 * An instance of this class is passed to every kernel at run-time. It allows
 * the kernel to retrieve information about the run-time environment.
 */
struct DaphneContext {
    // Feel free to extend this class with any kind of run-time information
    // that might be relevant to some kernel. Each kernel can extract the
    // information it requires and does not need to worry about information it
    // does not require.
    // If you need to add a bunch of related information items, please consider
    // creating an individual struct/class for them and adding a single member
    // of that type here, in order to separate concerns and allow a  high-level
    // overview of the context information.


    std::vector<std::unique_ptr<IContext>> cuda_contexts;
    int MTCounter = 0;
    
    std::vector<std::vector<Entry>> timeTraceEntries;
    /*
    for(int i=0; i<32; i++) {
        std::vector<Entry> tmp;
	timeTraceEntries.push_back(tmp);
    }
    */

    /**
     * @brief The user configuration (including information passed via CLI
     * arguments etc.).
     *
     * Modifying the configuration is intensionally allowed, since it enables
     * changing the configuration at run-time via DaphneDSL.
     */
    DaphneUserConfig& config;

    explicit DaphneContext(DaphneUserConfig& config) : config(config) {
        //
        std::cout << "DaphneContext Start." << std::endl;
	for(int i=0; i<20; i++) {
	    timeTraceEntries.push_back(std::vector<Entry>(0));
	}
    }

    ~DaphneContext() {
        for (auto& ctx : cuda_contexts) {
            ctx->destroy();
        }
	//std::cout << timeTraceEntries[0][0].LocalTaskID << std::endl;
	std::cout << "Size of main vector=" << timeTraceEntries.size() << std::endl;
	std::cout << "Size of first subvector=" << timeTraceEntries[0].size() << std::endl;
	for (size_t i=0; i<timeTraceEntries.size(); i++) {
	    for (size_t j=0; j<timeTraceEntries[i].size(); j++) {
		auto durationForPrinting = std::chrono::duration_cast<std::chrono::microseconds>(timeTraceEntries[i][j].Duration).count();
	        std::cout << durationForPrinting << "," << timeTraceEntries[i][j].Worker << "," << timeTraceEntries[i][j].Round << "," << timeTraceEntries[i][j].LocalTaskID << " ";
	    }
	    std::cout << std::endl;
	}
	std::cout << "DaphneContext End." << std::endl;
        cuda_contexts.clear();
    }

#ifdef USE_CUDA
    // ToDo: in a multi device setting this should use a find call instead of a direct [] access
    [[nodiscard]] CUDAContext* getCUDAContext(int dev_id) const {
        return dynamic_cast<CUDAContext*>(cuda_contexts[dev_id].get());
    }
#endif

    [[nodiscard]] bool useCUDA() const { return !cuda_contexts.empty(); }
    
    [[maybe_unused]] [[nodiscard]] DaphneUserConfig getUserConfig() const { return config; }
};

#endif //SRC_RUNTIME_LOCAL_CONTEXT_DAPHNECONTEXT_H
