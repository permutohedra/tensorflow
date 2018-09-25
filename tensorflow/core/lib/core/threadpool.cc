/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/lib/core/threadpool.h"

#define EIGEN_USE_THREADS
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/platform/context.h"
#include "tensorflow/core/platform/denormal.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/setround.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace thread {

struct EigenEnvironment {
  typedef Thread EnvThread;
  struct TaskImpl {
    std::function<void()> f;
    Context context;
    uint64 trace_id;
  };
  struct Task {
    std::unique_ptr<TaskImpl> f;
  };

  Env* const env_;
  const ThreadOptions thread_options_;
  const string name_;

  EigenEnvironment(Env* env, const ThreadOptions& thread_options,
                   const string& name)
      : env_(env), thread_options_(thread_options), name_(name) {}

  EnvThread* CreateThread(std::function<void()> f) {
    return env_->StartThread(thread_options_, name_, [=]() {
      // Set the processor flag to flush denormals to zero.
      port::ScopedFlushDenormal flush;
      // Set the processor rounding mode to ROUND TO NEAREST.
      port::ScopedSetRound round(FE_TONEAREST);
      f();
    });
  }

  Task CreateTask(std::function<void()> f) {
    uint64 id = 0;
    if (tracing::EventCollector::IsEnabled()) {
      id = tracing::GetUniqueArg();
      tracing::RecordEvent(tracing::EventCategory::kScheduleClosure, id);
    }
    return Task{
        std::unique_ptr<TaskImpl>(new TaskImpl{
            std::move(f),
            Context(ContextKind::kThread),
            id,
        }),
    };
  }

  void ExecuteTask(const Task& t) {
    WithContext wc(t.f->context);
    tracing::ScopedRegion region(tracing::EventCategory::kRunClosure,
                                 t.f->trace_id);
    t.f->f();
  }
};

struct ThreadPool::Impl : Eigen::ThreadPoolTempl<EigenEnvironment> {
  Impl(Env* env, const ThreadOptions& thread_options, const string& name,
       int num_threads, bool low_latency_hint)
      : Eigen::ThreadPoolTempl<EigenEnvironment>(
            num_threads, low_latency_hint,
            EigenEnvironment(env, thread_options, name)) {}

  void ParallelFor(int64 total, int64 cost_per_unit,
                   std::function<void(int64, int64)> fn) {
    CHECK_GE(total, 0);
    CHECK_EQ(total, (int64)(Eigen::Index)total);
    Eigen::ThreadPoolDevice device(this, this->NumThreads());
    device.parallelFor(
        total, Eigen::TensorOpCost(0, 0, cost_per_unit),
        [&fn](Eigen::Index first, Eigen::Index last) { fn(first, last); });
  }
};

ThreadPool::ThreadPool(Env* env, const string& name, int num_threads)
    : ThreadPool(env, ThreadOptions(), name, num_threads, true) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads)
    : ThreadPool(env, thread_options, name, num_threads, true) {}

ThreadPool::ThreadPool(Env* env, const ThreadOptions& thread_options,
                       const string& name, int num_threads,
                       bool low_latency_hint)
    : env_(env),
      thread_options_(thread_options),
      name_("tf_" + name),
      num_threads_(num_threads),
      low_latency_hint_(low_latency_hint) {
  CHECK_GE(num_threads_, 1);
  impl_.reset(new ThreadPool::Impl(env_, thread_options_, name_,
                                   num_threads_, low_latency_hint_));

  std::lock_guard<std::mutex> lock(all_thread_pools_mu_);
  all_thread_pools_.push_front(this);
  all_thread_pools_it_ = all_thread_pools_.begin();
}

ThreadPool::~ThreadPool() {
  std::lock_guard<std::mutex> lock(all_thread_pools_mu_);
  all_thread_pools_.erase(all_thread_pools_it_);
}

void ThreadPool::Schedule(std::function<void()> fn) {
  CHECK(fn != nullptr);
  CHECK(impl_);
  impl_->Schedule(std::move(fn));
}

void ThreadPool::ParallelFor(int64 total, int64 cost_per_unit,
                             std::function<void(int64, int64)> fn) {
  CHECK(impl_);
  impl_->ParallelFor(total, cost_per_unit, std::move(fn));
}

void ThreadPool::ParallelForWithWorkerId(
    int64 total, int64 cost_per_unit,
    const std::function<void(int64, int64, int)>& fn) {
  impl_->ParallelFor(total, cost_per_unit,
                     [this, &fn](int64 start, int64 limit) {
                       // ParallelFor may use the current thread to do some
                       // work synchronously. When calling CurrentThreadId()
                       // from outside of the thread pool, we get -1, so we can
                       // shift every id up by 1.
                       int id = CurrentThreadId() + 1;
                       fn(start, limit, id);
                     });
}

void ThreadPool::PauseAllThreads() {
  std::lock_guard<std::mutex> lock(all_thread_pools_mu_);
  for (ThreadPool* tp : all_thread_pools_) {
    // This will immediately set impl_ to nullptr and then wait for any
    // outstanding work items to complete. This means that it is an error to
    // call Schedule() after PauseAllThreads() is called, even from within
    // another work item.
    tp->impl_.reset();
  }
}

void ThreadPool::ResumeAllThreads() {
  std::lock_guard<std::mutex> lock(all_thread_pools_mu_);
  for (ThreadPool* tp : all_thread_pools_) {
    tp->impl_.reset(new ThreadPool::Impl(
          tp->env_, tp->thread_options_, tp->name_,
	  tp->num_threads_, tp->low_latency_hint_));
  }
}

std::list<ThreadPool*> ThreadPool::all_thread_pools_;
std::mutex ThreadPool::all_thread_pools_mu_;

int ThreadPool::CurrentThreadId() const { return impl_->CurrentThreadId(); }

}  // namespace thread
}  // namespace tensorflow
