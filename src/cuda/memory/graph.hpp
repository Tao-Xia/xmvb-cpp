#pragma once

#include <utility>

#include "cuda/memory/context.hpp"

namespace xmvb::cuda {

/** Owns one CUDA executable graph; all captured device buffers remain caller-owned. */
class Graph {
 public:
  template<class Emit>
  Graph(Context& context, Emit&& emit) : context_(&context) {
    cudaGraph_t graph=nullptr;
    check_cuda(cudaStreamBeginCapture(context.stream(),cudaStreamCaptureModeThreadLocal),
        "begin action graph capture");
    try {
      std::forward<Emit>(emit)();
    } catch (...) {
      cudaStreamEndCapture(context.stream(),&graph);
      if (graph) cudaGraphDestroy(graph);
      throw;
    }
    check_cuda(cudaStreamEndCapture(context.stream(),&graph),"end action graph capture");
    try {
      check_cuda(cudaGraphGetNodes(graph,nullptr,&nodes_),"query action graph nodes");
      check_cuda(cudaGraphInstantiateWithFlags(&execution_,graph,0),"instantiate action graph");
      check_cuda(cudaGraphUpload(execution_,context.stream()),"upload action graph");
    } catch (...) {
      if (execution_) cudaGraphExecDestroy(execution_);
      cudaGraphDestroy(graph);
      throw;
    }
    cudaGraphDestroy(graph);
    context.record_graph_capture();
  }
  Graph(const Graph&)=delete;
  Graph& operator=(const Graph&)=delete;
  ~Graph() { if (execution_) cudaGraphExecDestroy(execution_); }
  void launch() const {
    check_cuda(cudaGraphLaunch(execution_,context_->stream()),"launch action graph");
    context_->record_graph_launch();
  }
  std::size_t nodes() const noexcept { return nodes_; }

 private:
  Context* context_;
  cudaGraphExec_t execution_=nullptr;
  std::size_t nodes_=0;
};

} // namespace xmvb::cuda
