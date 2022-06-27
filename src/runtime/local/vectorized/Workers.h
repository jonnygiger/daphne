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

#pragma once

#include <thread>
#include <sched.h>
#include <numeric>

class Worker {
protected:
    std::unique_ptr<std::thread> t;

    // Worker only used as derived class, which starts the thread after the class has been constructed (order matters).
    Worker() : t() {}
public:
    // Worker is move only due to std::thread. Therefore, we delete the copy constructor and the assignment operator.
    Worker(const Worker&) = delete;
    Worker& operator=(const Worker&) = delete;

    // move constructor
    Worker(Worker&& obj)  noexcept : t(std::move(obj.t)) {}

    // move assignment operator
    Worker& operator=(Worker&& obj)  noexcept {
        if(t->joinable())
            t->join();
        t = std::move(obj.t);
        return *this;
    }

    virtual ~Worker() {
        if(t->joinable())
            t->join();
    };

    void join() {
        t->join();
    }
    virtual void run() = 0;
    static bool isEOF(Task* t) {
        return dynamic_cast<EOFTask*>(t);
    }
};

class WorkerCPU : public Worker {
    std::vector<TaskQueue*> _q;
    std::vector<int> _physical_ids;
    std::vector<int> _unique_threads;
    std::array<bool, 256> eofWorkers;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numQueues;
    int _queueMode;
    int _stealLogic;
    bool _pinWorkers;
    bool _foreman;
public:
    // this constructor is to be used in practice
    WorkerCPU(std::vector<TaskQueue*> deques, std::vector<int> physical_ids, std::vector<int> unique_threads, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0, int queueMode = 0, int stealLogic = 0, bool pinWorkers = 0, bool foreman = 0) : Worker(), _q(deques), _physical_ids(physical_ids), _unique_threads(unique_threads),
            _verbose(verbose), _fid(fid), _batchSize(batchSize), _threadID(threadID), _numQueues(numQueues), _queueMode(queueMode), _stealLogic(stealLogic), _pinWorkers(pinWorkers), _foreman(foreman) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPU::run, this);
    }
    
    ~WorkerCPU() override = default;

    void run() override {
        if (_pinWorkers) {
            // pin worker to CPU core
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(_threadID, &cpuset);
            sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        }

        int currentDomain = _physical_ids[_threadID];
        int targetQueue = currentDomain;
        int startingQueue = targetQueue;
        
        Task* t = _q[currentDomain]->dequeueTask();

        while( !isEOF(t) ) {
            //execute self-contained task
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
            t->execute(_fid, _batchSize);
            delete t;
            //get next tasks (blocking)
            t = _q[currentDomain]->dequeueTask();
        }

        // All tasks from own queue have completed. If this is a foreman worker, stealing half from other queue

        if( _numQueues > 1 && _foreman) {
            
            targetQueue = (targetQueue+1)%_numQueues;
            
            while ( targetQueue != startingQueue ) {
                std::vector<Task*> tmp;
                _q[targetQueue]->dequeueHalf(tmp);
                _q[currentDomain]->enqueueBatch(tmp);
                t = _q[currentDomain]->dequeueTask();
                if( isEOF(t) ) {
                    targetQueue = (targetQueue+1)%_numQueues;
                } else {
                    t->execute(_fid, _batchSize);
                    delete t;
                }
            }
        }
        
        // No more tasks available anywhere
        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

////entry point for std:thread
//static void runWorker(Worker* worker) {
//    worker->run();
//}
