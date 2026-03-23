#pragma once
#include <cstdint>
#include <torch/nn/modules/linear.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>



class ModelImpl : public torch::nn::Module
{
public:
    
    ModelImpl(int input_dim, int hidden_dim);
    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor x, int64_t n, torch::Tensor tril_idx);
    void load_parameters(const std::string &path);
private:
    torch::nn::Sequential net;
    torch::nn::Linear L_proj{nullptr}, H_proj{nullptr};

};
TORCH_MODULE(Model);



class Infer
{
public:
    Infer(const int nstr, const Model &model);;
    void forward(double *hvb, double *svb, double *hvb_1det, double *svb_1det);

private:
    int64_t nstr_;
    Model model_;
    torch::Tensor tril_idx_, row_idx_, col_idx_;
};


Model GetDeepVBHModel();
Infer GetInfer(int nstr, const Model& model);