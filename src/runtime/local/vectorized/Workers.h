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
    static bool isUnavailableTask(Task* t) {
        return dynamic_cast<UnavailableTask*>(t);
    }
};

class WorkerCPU : public Worker {
    TaskQueue* _q;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
public:
    // this constructor is to be used in practice
    WorkerCPU(TaskQueue* tq, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100) : Worker(), _q(tq),
            _verbose(verbose), _fid(fid), _batchSize(batchSize) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPU::run, this);
    }
    
    ~WorkerCPU() override = default;

    void run() override {
        Task* t = _q->dequeueTask();

        while( !isEOF(t) ) {
            //execute self-contained task
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
            t->execute(_fid, _batchSize);
            delete t;
            //get next tasks (blocking)
            t = _q->dequeueTask();
        }
        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

class WorkerCPUPerCPU : public Worker {
    std::vector<TaskQueue*> _q;
    std::vector<int> _numaDomains;
    std::array<bool, 256> eofWorkers;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numaID;
    int _numQueues;
    int _queueMode;
    int _stealLogic;
public:
    // this constructor is to be used in practice
    WorkerCPUPerCPU(std::vector<TaskQueue*> deques, std::vector<int> numaDomains, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0, int queueMode = 0, int stealLogic = 0) : Worker(), _q(deques), _numaDomains(numaDomains),
            _verbose(verbose), _fid(fid), _batchSize(batchSize), _threadID(threadID), _numQueues(numQueues), _queueMode(queueMode), _stealLogic(stealLogic) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPUPerCPU::run, this);
    }
    
    ~WorkerCPUPerCPU() override = default;

    void run() override {
        // pin worker to CPU core
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(_threadID, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        
        int targetQueue = _threadID;
        int currentDomain = _numaDomains[_threadID];
        
        Task* t = _q[targetQueue]->dequeueTask();

        while( !isEOF(t) ) {
            //execute self-contained task
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
            t->execute(_fid, _batchSize);
            delete t;
            //get next tasks (blocking)
            t = _q[targetQueue]->dequeueTask();
        }
        
        if( _stealLogic == 0) {
            // Stealing in sequential order
            
            targetQueue = (targetQueue+1)%_numQueues;
            
            while (targetQueue != _threadID) {
                t = _q[targetQueue]->dequeueTask();
                if( isEOF(t) ) {
                    targetQueue = (targetQueue+1)%_numQueues;
                } else {
                    t->execute(_fid, _batchSize);
                    delete t;
                }
            }
        } else if ( _stealLogic == 1) {
            // Stealing in sequential order from same domain first
        
            targetQueue = (targetQueue+1)%_numQueues;

            while (targetQueue != _threadID) {
                if ( _numaDomains[targetQueue] == currentDomain ){
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        targetQueue = (targetQueue+1)%_numQueues;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                } else {
                    targetQueue = (targetQueue+1)%_numQueues;
                }
            }
            
            // No more tasks on this domain, now switching to other domain
            
            targetQueue = (targetQueue+1)%_numQueues;
            
            while (targetQueue != _threadID) {
                if ( _numaDomains[targetQueue] != currentDomain ){
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        targetQueue = (targetQueue+1)%_numQueues;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                } else {
                    targetQueue = (targetQueue+1)%_numQueues;
                }
            }
        } else if( _stealLogic == 2) {
            // stealing from random workers until all workers EOF
            
            eofWorkers.fill(false);
            while( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < _numQueues ) {
                targetQueue = rand() % _numQueues;
                if( eofWorkers[targetQueue] == false ) {
                    t = _q[targetQueue]->dequeueTask();
                    //std::cout << "Execute task stolen from: " << targetQueue << std::endl;
                    if( isEOF(t) ) {
                        eofWorkers[targetQueue] = true;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                }
            }
            
        } else if ( _stealLogic == 3) {
            // stealing from random workers from same socket first
            int queuesThisDomain = 0;
            eofWorkers.fill(false);
            
            for( int i=0; i<_numQueues; i++ ) {
                if( _numaDomains[i] == currentDomain ) {
                    queuesThisDomain++;
                }
            }
            
            while( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < queuesThisDomain ) {
                targetQueue = rand() % _numQueues;
                if( _numaDomains[targetQueue] == currentDomain ) {
                    if( eofWorkers[targetQueue] == false ) {
                        t = _q[targetQueue]->dequeueTask();
                        if( isEOF(t) ) {
                            eofWorkers[targetQueue] = true;
                        } else {
                            t->execute(_fid, _batchSize);
                            delete t;
                        }
                    }
                }
            }
            
            // all workers on same domain are EOF, now also allowing stealing from other domain
            // This could also be done by keeping a list of EOF workers on the other domain
            
            while ( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < _numQueues ) {
                targetQueue = rand() % _numQueues;
                // no need to check if they are on the other domain, because otherwise they would be EOF anyway
                if( eofWorkers[targetQueue] == false ) {
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        eofWorkers[targetQueue] = true;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                }
            }
        }
        
        // No more tasks available anywhere
        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

class WorkerCPUPerGroup : public Worker {
    std::vector<TaskQueue*> _q;
    std::vector<int> _numaDomains;
    std::array<bool, 256> eofWorkers;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numaID;
    int _numQueues;
    int _queueMode;
    int _stealLogic;
public:
    // this constructor is to be used in practice
    WorkerCPUPerGroup(std::vector<TaskQueue*> deques, std::vector<int> numaDomains, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0, int queueMode = 0, int stealLogic = 0) : Worker(), _q(deques), _numaDomains(numaDomains),
            _verbose(verbose), _fid(fid), _batchSize(batchSize), _threadID(threadID), _numQueues(numQueues), _queueMode(queueMode), _stealLogic(stealLogic) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&WorkerCPUPerGroup::run, this);
    }
    
    ~WorkerCPUPerGroup() override = default;

    void run() override {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(_threadID, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        int currentDomain = _numaDomains[_threadID];
        int targetQueue = currentDomain;
        
        Task* t = _q[targetQueue]->dequeueTask();

        while( !isEOF(t) ) {
            //execute self-contained task
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
            t->execute(_fid, _batchSize);
            delete t;
            //get next tasks (blocking)
            t = _q[targetQueue]->dequeueTask();
        }
        
        // No more tasks on own queue, now switching to other queues
        // Can be improved by assigning a "Foreman" for each socket
        // responsible for task stealing
        
        targetQueue = (targetQueue+1)%_numQueues;

        while(targetQueue != currentDomain) {
            t = _q[targetQueue]->dequeueTask();
            if( isEOF(t) ) {
                targetQueue = (targetQueue+1)%_numQueues;
            } else {
                t->execute(_fid, _batchSize);
                delete t;
            }
        }

        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

class NonBlockingWorkerCPU : public Worker {
    TaskQueue* _q;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
public:
    // this constructor is to be used in practice
    NonBlockingWorkerCPU(TaskQueue* tq, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100) : Worker(), _q(tq),
            _verbose(verbose), _fid(fid), _batchSize(batchSize) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&NonBlockingWorkerCPU::run, this);
    }
    
    ~NonBlockingWorkerCPU() override = default;

    void run() override {
        Task* t = _q->dequeueTask();

        while( !isEOF(t) ) {
			//execute self-contained task
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
			t->execute(_fid, _batchSize);
			delete t;
			
			// The first task has been executed, dequeueing the next
			t = _q->dequeueTask();
			while( isUnavailableTask(t) ) {
				// dequeue was unsuccessful, a placeholder was returned, try again (spin loop)
				t = _q->dequeueTask();
			}
			// dequeue was successful, start over with the outer while loop
        }
		// EOF was reached on the central queue, there are no tasks left to start
        
        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

class NonBlockingWorkerCPUPerCPU : public Worker {
    std::vector<TaskQueue*> _q;
    std::vector<int> _numaDomains;
    std::array<bool, 256> eofWorkers;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numaID;
    int _numQueues;
    int _queueMode;
    int _stealLogic;
public:
    // this constructor is to be used in practice
    NonBlockingWorkerCPUPerCPU(std::vector<TaskQueue*> deques, std::vector<int> numaDomains, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0, int queueMode = 0, int stealLogic = 0) : Worker(), _q(deques), _numaDomains(numaDomains),
            _verbose(verbose), _fid(fid), _batchSize(batchSize), _threadID(threadID), _numQueues(numQueues), _queueMode(queueMode), _stealLogic(stealLogic) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&NonBlockingWorkerCPUPerCPU::run, this);
    }
    
    ~NonBlockingWorkerCPUPerCPU() override = default;

    void run() override {
        // pin worker to CPU core
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(_threadID, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        
        int targetQueue = _threadID;
        int currentDomain = _numaDomains[_threadID];
        
        Task* t = _q[targetQueue]->dequeueTask();

        while( !isEOF(t) ) {
			//execute self-contained task
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
			t->execute(_fid, _batchSize);
			delete t;
			
			// The first task has been executed, dequeueing the next
			t = _q[targetQueue]->dequeueTask();
			while( isUnavailableTask(t) ) {
				// dequeue was unsuccessful, a placeholder was returned, try again (spin loop)
				t = _q[targetQueue]->dequeueTask();
			}
			// dequeue was successful, start over with the outer while loop
        }
		// No more tasks left on this queue, now stealing from other queues
        
        if( _stealLogic == 0) {
            // Stealing in sequential order
            
            bool eofQueues[_numQueues];
            int numEofQueues = 0;
            for(int i = 0; i < n; i++){
                eofQueues[i] = false;
            }
            eofQueues[_threadID] = true;
            targetQueue = (targetQueue+1)%_numQueues;
            
            while( numEofQueues < _numQueues ) {
                if( eofQueues[targetQueue] == false ) {
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        eofQueues[targetQueue] = true;
                        numEofQueues++;
                    } else if( !isUnavailableTask(t) ) {
                            t->execute(_fid, _batchSize);
                            delete t;
                    }
                    targetQueue = (targetQueue+1)%_numQueues;
                }
            }
        } else if ( _stealLogic == 1) {
            // Stealing in sequential order from same domain first
            
            int numQueuesThisDomain = 0;
            for( int i=0; i<_numQueues; i++)
            {
                if( _numaDomains[i] == currentDomain ) {
                    numQueuesThisDomain++;
                }
            }
            bool eofQueues[_numQueues];
            eofQueues[_threadID] = true;
            int numEofQueuesSameDomain = 1;
            int numQueuesOtherDomain = 0;
            for( int i=0; i<_numQueues; i++)
            {
                if( _numaDomains[i] != currentDomain ) {
                    numQueuesOtherDomain++;
                }
            }
            int numEofQueuesOtherDomain = 0;
            
            targetQueue = (targetQueue+1)%_numQueues;

            // steal tasks from same domain
            while (numEofQueuesSameDomain < numQueuesThisDomain) {
                if( _numaDomains[targetQueue] == currentDomain ){
                    if( eofQueues[targetQueue] == false ) {
                        t = _q[targetQueue]->dequeueTask();
                        if( isEOF(t) ) {
                            eofQueues[targetQueue] = true;
                            numEofQueuesSameDomain++;
                        } else if( !isUnavailableTask(t) ) {
                    }
                    targetQueue = (targetQueue+1)%_numQueues;
                        
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        targetQueue = (targetQueue+1)%_numQueues;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                } else {
                    targetQueue = (targetQueue+1)%_numQueues;
                }
            }
            
            // No more tasks on this domain, now switching to other domain
            
            targetQueue = (targetQueue+1)%_numQueues;
            
            while (targetQueue != _threadID) {
                if ( _numaDomains[targetQueue] != currentDomain ){
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        targetQueue = (targetQueue+1)%_numQueues;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                } else {
                    targetQueue = (targetQueue+1)%_numQueues;
                }
            }
        } else if( _stealLogic == 2) {
            // stealing from random workers until all workers EOF
            
            eofWorkers.fill(false);
            while( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < _numQueues ) {
                targetQueue = rand() % _numQueues;
                if( eofWorkers[targetQueue] == false ) {
                    t = _q[targetQueue]->dequeueTask();
                    //std::cout << "Execute task stolen from: " << targetQueue << std::endl;
                    if( isEOF(t) ) {
                        eofWorkers[targetQueue] = true;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                }
            }
            
        } else if ( _stealLogic == 3) {
            // stealing from random workers from same socket first
            int queuesThisDomain = 0;
            eofWorkers.fill(false);
            
            for( int i=0; i<_numQueues; i++ ) {
                if( _numaDomains[i] == currentDomain ) {
                    queuesThisDomain++;
                }
            }
            
            while( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < queuesThisDomain ) {
                targetQueue = rand() % _numQueues;
                if( _numaDomains[targetQueue] == currentDomain ) {
                    if( eofWorkers[targetQueue] == false ) {
                        t = _q[targetQueue]->dequeueTask();
                        if( isEOF(t) ) {
                            eofWorkers[targetQueue] = true;
                        } else {
                            t->execute(_fid, _batchSize);
                            delete t;
                        }
                    }
                }
            }
            
            // all workers on same domain are EOF, now also allowing stealing from other domain
            // This could also be done by keeping a list of EOF workers on the other domain
            
            while ( std::accumulate(eofWorkers.begin(), eofWorkers.end(), 0) < _numQueues ) {
                targetQueue = rand() % _numQueues;
                // no need to check if they are on the other domain, because otherwise they would be EOF anyway
                if( eofWorkers[targetQueue] == false ) {
                    t = _q[targetQueue]->dequeueTask();
                    if( isEOF(t) ) {
                        eofWorkers[targetQueue] = true;
                    } else {
                        t->execute(_fid, _batchSize);
                        delete t;
                    }
                }
            }
        }
        
        // No more tasks available anywhere
        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

class NonBlockingWorkerCPUPerGroup : public Worker {
    std::vector<TaskQueue*> _q;
    std::vector<int> _numaDomains;
    std::array<bool, 256> eofWorkers;
    bool _verbose;
    uint32_t _fid;
    uint32_t _batchSize;
    int _threadID;
    int _numaID;
    int _numQueues;
    int _queueMode;
    int _stealLogic;
public:
    // this constructor is to be used in practice
    NonBlockingWorkerCPUPerGroup(std::vector<TaskQueue*> deques, std::vector<int> numaDomains, bool verbose, uint32_t fid = 0, uint32_t batchSize = 100, int threadID = 0, int numQueues = 0, int queueMode = 0, int stealLogic = 0) : Worker(), _q(deques), _numaDomains(numaDomains),
            _verbose(verbose), _fid(fid), _batchSize(batchSize), _threadID(threadID), _numQueues(numQueues), _queueMode(queueMode), _stealLogic(stealLogic) {
        // at last, start the thread
        t = std::make_unique<std::thread>(&NonBlockingWorkerCPUPerGroup::run, this);
    }
    
    ~NonBlockingWorkerCPUPerGroup() override = default;

    void run() override {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(_threadID, &cpuset);
        sched_setaffinity(0, sizeof(cpu_set_t), &cpuset);
        int currentDomain = _numaDomains[_threadID];
        int targetQueue = currentDomain;
        
        Task* t = _q[targetQueue]->dequeueTask();

        while( !isEOF(t) ) {
            //execute self-contained task
            if( _verbose )
                std::cerr << "WorkerCPU: executing task." << std::endl;
            t->execute(_fid, _batchSize);
            delete t;
            //get next tasks (blocking)
            t = _q[targetQueue]->dequeueTask();
        }
        
        // No more tasks on own queue, now switching to other queues
        // Can be improved by assigning a "Foreman" for each socket
        // responsible for task stealing
        
        targetQueue = (targetQueue+1)%_numQueues;

        while(targetQueue != currentDomain) {
            t = _q[targetQueue]->dequeueTask();
            if( isEOF(t) ) {
                targetQueue = (targetQueue+1)%_numQueues;
            } else {
                t->execute(_fid, _batchSize);
                delete t;
            }
        }

        if( _verbose )
            std::cerr << "WorkerCPU: received EOF, finalized." << std::endl;
    }
};

////entry point for std:thread
//static void runWorker(Worker* worker) {
//    worker->run();
//}
