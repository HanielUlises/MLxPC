#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <random>
#include <filesystem>
#include <fstream>
#include <iomanip>

// Feedforward Neural Network for multi-class classification
struct MultiClassFNN : torch::nn::Module {
    MultiClassFNN(int64_t input_dim, int64_t hidden_dim, int64_t num_classes) {
        // Linear layers
        fc1 = register_module("fc1", torch::nn::Linear(input_dim, hidden_dim));
        bn1 = register_module("bn1", torch::nn::BatchNorm1d(hidden_dim));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_dim, hidden_dim));
        bn2 = register_module("bn2", torch::nn::BatchNorm1d(hidden_dim));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_dim, hidden_dim / 2));
        bn3 = register_module("bn3", torch::nn::BatchNorm1d(hidden_dim / 2));
        fc4 = register_module("fc4", torch::nn::Linear(hidden_dim / 2, num_classes));

        // Dropout
        dropout = torch::nn::Dropout(0.3);
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(bn1(fc1->forward(x)));
        x = dropout(x);
        x = torch::relu(bn2(fc2->forward(x)));
        x = dropout(x);
        x = torch::relu(bn3(fc3->forward(x)));
        x = fc4->forward(x); // No activation here, as CrossEntropyLoss expects raw logits
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr}, fc4{nullptr};
    torch::nn::BatchNorm1d bn1{nullptr}, bn2{nullptr}, bn3{nullptr};
    torch::nn::Dropout dropout{nullptr};
};

// Generate synthetic dataset for multi-class classification
std::pair<torch::Tensor, torch::Tensor> synthetic_data_generation(int64_t num_samples, int64_t input_dim, int64_t num_classes) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0, 1.0);

    torch::Tensor data = torch::empty({num_samples, input_dim});
    torch::Tensor labels = torch::empty({num_samples}, torch::kLong);

    for (int64_t i = 0; i < num_samples; ++i) {
        int64_t cls = i % num_classes;
        labels[i] = cls;
        for (int64_t j = 0; j < input_dim; ++j) {
            data[i][j] = dist(gen) + (cls * 0.5); // Shift data slightly per class
        }
    }

    return {data, labels};
}

// Training function
void train(
    MultiClassFNN& model,
    torch::Tensor data,
    torch::Tensor labels,
    int64_t num_classes,
    int64_t batch_size,
    int64_t num_epochs,
    double learning_rate,
    torch::Device device
) {
    model.to(device);
    data = data.to(device);
    labels = labels.to(device);

    // Optimizer and loss function
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(learning_rate));
    auto loss_fn = torch::nn::CrossEntropyLoss();

    int64_t num_samples = data.size(0);
    std::cout << std::fixed << std::setprecision(4);

    for (int64_t epoch = 1; epoch <= num_epochs; ++epoch) {
        model.train();
        double epoch_loss = 0.0;
        int64_t correct = 0;

        // Shuffle indices
        auto indices = torch::randperm(num_samples, torch::kLong).to(device);
        for (int64_t i = 0; i < num_samples; i += batch_size) {
            auto batch_indices = indices.slice(0, i, std::min(i + batch_size, num_samples));
            auto batch_data = data.index_select(0, batch_indices);
            auto batch_labels = labels.index_select(0, batch_indices);

            optimizer.zero_grad();
            auto output = model.forward(batch_data);
            auto loss = loss_fn(output, batch_labels);
            loss.backward();
            optimizer.step();

            epoch_loss += loss.item<double>() * batch_data.size(0);
            auto predicted = output.argmax(1);
            correct += predicted.eq(batch_labels).sum().item<int64_t>();
        }

        epoch_loss /= num_samples;
        double accuracy = static_cast<double>(correct) / num_samples * 100.0;
        std::cout << "Epoch [" << epoch << "/" << num_epochs << "], Loss: " << epoch_loss
                  << ", Accuracy: " << accuracy << "%\n";
    }
}

// Evaluation function
void evaluate(MultiClassFNN& model, torch::Tensor data, torch::Tensor labels, torch::Device device) {
    model.eval();
    data = data.to(device);
    labels = labels.to(device);

    auto loss_fn = torch::nn::CrossEntropyLoss();
    torch::NoGradGuard no_grad;

    auto output = model.forward(data);
    auto loss = loss_fn(output, labels);
    auto predicted = output.argmax(1);
    auto correct = predicted.eq(labels).sum().item<int64_t>();
    double accuracy = static_cast<double>(correct) / data.size(0) * 100.0;

    std::cout << "Evaluation Loss: " << loss.item<double>()
              << ", Accuracy: " << accuracy << "%\n";
}

void save_model(MultiClassFNN& model, const std::string& path) {
    torch::save(model, path);
    std::cout << "Model saved to " << path << "\n";
}

void load_model(MultiClassFNN& model, const std::string& path) {
    torch::load(model, path);
    std::cout << "Model loaded from " << path << "\n";
}

int main() {
    try {
        const int64_t input_dim = 20;
        const int64_t hidden_dim = 256;
        const int64_t num_classes = 5;
        const int64_t num_samples = 10000;
        const int64_t batch_size = 64;
        const int64_t num_epochs = 20;
        const double learning_rate = 0.001;
        const std::string model_path = "multiclass_fnn.pt";

        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n";

        auto [train_data, train_labels] = synthetic_data_generation(num_samples, input_dim, num_classes);
        auto [test_data, test_labels] = synthetic_data_generation(num_samples / 5, input_dim, num_classes);

        MultiClassFNN model(input_dim, hidden_dim, num_classes);
        std::cout << "Model initialized with " << input_dim << " inputs, " << num_classes << " classes.\n";

        train(model, train_data, train_labels, num_classes, batch_size, num_epochs, learning_rate, device);

        evaluate(model, test_data, test_labels, device);

        save_model(model, model_path);

        MultiClassFNN loaded_model(input_dim, hidden_dim, num_classes);
        load_model(loaded_model, model_path);
        evaluate(loaded_model, test_data, test_labels, device);
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}