// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "onnxruntime_cxx_api.h"
#include <iostream>
#include <mutex>

namespace isaaclab
{

class Algorithms
{
public:
    virtual std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs) = 0;

    std::vector<float> get_action()
    {
        std::lock_guard<std::mutex> lock(act_mtx_);
        return action;
    }
    
    std::vector<float> action;
protected:
    std::mutex act_mtx_;
};

class OrtRunner : public Algorithms
{
public:
    OrtRunner(std::string model_path)
    {
        // Init Model
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "onnx_model");
        session_options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);

        session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);

        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            Ort::TypeInfo input_type = session->GetInputTypeInfo(i);
            input_shapes.push_back(input_type.GetTensorTypeAndShapeInfo().GetShape());
            auto input_name = session->GetInputNameAllocated(i, allocator);
            input_names.push_back(input_name.release());
        }

        for (const auto& shape : input_shapes) {
            size_t size = 1;
            for (const auto& dim : shape) {
                size *= dim;
            }
            input_sizes.push_back(size);
        }

        // Collect all outputs (actions + optional GRU h_out)
        for (size_t i = 0; i < session->GetOutputCount(); ++i) {
            auto output_name = session->GetOutputNameAllocated(i, allocator);
            output_names.push_back(output_name.release());
        }

        // First output is always actions
        Ort::TypeInfo output_type = session->GetOutputTypeInfo(0);
        output_shape = output_type.GetTensorTypeAndShapeInfo().GetShape();
        action.resize(output_shape[1]);

        // Detect GRU: model has "h_in" input and "h_out" output
        for (const auto& name : input_names) {
            if (std::string(name) == "h_in") {
                is_recurrent_ = true;
                break;
            }
        }

        if (is_recurrent_) {
            // Find h_in shape and allocate zero hidden state
            for (size_t i = 0; i < input_names.size(); ++i) {
                if (std::string(input_names[i]) == "h_in") {
                    h_shape_ = input_shapes[i];   // e.g. [1, 1, 256]
                    break;
                }
            }
            size_t h_size = 1;
            for (auto d : h_shape_) h_size *= d;
            hidden_state_.assign(h_size, 0.0f);
            std::cout << "[OrtRunner] GRU policy detected — hidden size=" << h_size << std::endl;
        }
    }

    // Reset GRU hidden state (call on episode reset)
    void reset_hidden()
    {
        if (is_recurrent_) std::fill(hidden_state_.begin(), hidden_state_.end(), 0.0f);
    }

    std::vector<float> act(std::unordered_map<std::string, std::vector<float>> obs)
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        // Inject GRU hidden state into obs map so the loop below picks it up
        if (is_recurrent_) {
            obs["h_in"] = hidden_state_;
        }

        // Validate all ONNX inputs are present
        for (const auto& name : input_names) {
            if (obs.find(name) == obs.end()) {
                throw std::runtime_error("Input name " + std::string(name) + " not found in observations.");
            }
        }

        // Build input tensors (order must match input_names)
        std::vector<Ort::Value> input_tensors;
        for (size_t i = 0; i < input_names.size(); ++i) {
            const std::string name_str(input_names[i]);
            auto& input_data = obs.at(name_str);
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_data.data(), input_sizes[i],
                input_shapes[i].data(), input_shapes[i].size());
            input_tensors.push_back(std::move(input_tensor));
        }

        // Run — fetch all outputs so h_out is available
        auto output_tensors = session->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size());

        // Copy actions (first output)
        {
            std::lock_guard<std::mutex> lock(act_mtx_);
            auto floatarr = output_tensors[0].GetTensorMutableData<float>();
            std::memcpy(action.data(), floatarr, output_shape[1] * sizeof(float));
        }

        // Update GRU hidden state from h_out (second output)
        if (is_recurrent_ && output_tensors.size() > 1) {
            auto h_ptr = output_tensors[1].GetTensorMutableData<float>();
            std::memcpy(hidden_state_.data(), h_ptr, hidden_state_.size() * sizeof(float));
        }

        return action;
    }

private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<std::vector<int64_t>> input_shapes;
    std::vector<int64_t> input_sizes;
    std::vector<int64_t> output_shape;

    // GRU hidden state
    bool is_recurrent_ = false;
    std::vector<int64_t> h_shape_;
    std::vector<float> hidden_state_;
};
};