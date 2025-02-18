#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <torch/torch.h>
#include <iostream>
#include <ATen/Functions.h>

namespace py = pybind11;

// Función para centrar los datos
torch::Tensor centrar_datos(const torch::Tensor& X) {
    auto media = torch::mean(X, 0);
    return X - media;
}

// Función para blanquear los datos
torch::Tensor blanquear_datos(const torch::Tensor& X) {
    auto cov = torch::mm(X.t(), X) / X.size(0);
    //auto eigen = torch::symeig(cov, /*eigenvectors=*/true);
    //auto D = std::get<0>(eigen);
    //auto E = std::get<1>(eigen);

    auto eigen = at::linalg_eigh(cov, "L");
    auto D = std::get<0>(eigen);
    auto E = std::get<1>(eigen);
    
    auto D_inv_sqrt = torch::diag(torch::pow(D, -0.5));
    return torch::mm(X, torch::mm(E, D_inv_sqrt));
}

// Función para realizar ICA utilizando el algoritmo de aproximación de Newton
std::tuple<torch::Tensor, torch::Tensor> ica(const torch::Tensor& X, int num_componentes, int max_iter = 200, double tol = 1e-4) {
    auto X_centrado = centrar_datos(X);
    auto X_blanqueado = blanquear_datos(X_centrado);

    int n_features = X_blanqueado.size(1);
    auto W = torch::eye(n_features);

    for (int i = 0; i < max_iter; ++i) {
        auto W_old = W.clone();
        auto WX = torch::mm(X_blanqueado, W.t());
        auto g = torch::tanh(WX);
        auto g_prime = 1 - g.pow(2);
        auto W_new = torch::mm(g.t(), X_blanqueado) / X_blanqueado.size(0) - torch::diag(torch::mean(g_prime, 0));
        W = W_new;

        // Decorrelación
        auto svd_result = torch::svd(W);
        auto U = std::get<0>(svd_result);
        auto V = std::get<2>(svd_result);
        W = torch::mm(U, V.t());

        if (torch::max(torch::abs(W - W_old)).item<double>() < tol) {
            break;
        }
    }

    auto S = torch::mm(X_blanqueado, W.t());
    return std::make_tuple(S, W);
}

PYBIND11_MODULE(ica_module, m) {
    m.def("ica", &ica, "Análisis de Componentes Independientes",
          py::arg("X"), py::arg("num_componentes"), py::arg("max_iter") = 200, py::arg("tol") = 1e-4);
}
