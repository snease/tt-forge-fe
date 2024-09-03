// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <type_traits>

#include "utils/assert.hpp"

namespace tt 
{

namespace graphlib 
{
    class Graph;
}

enum class GraphType : std::uint8_t
{
    Forward = 0,
    Backward = 1,
    Loss = 2,
    Optimizer = 3,
    GraphTypeCount = 4,
};

template <typename T>
constexpr std::underlying_type_t<T> to_underlying(T e) noexcept
{
    return static_cast<std::underlying_type_t<T>>(e);
}

constexpr std::uint8_t GRAPH_TYPE_COUNT = to_underlying(GraphType::GraphTypeCount);
using StaticGraphArray = std::array<graphlib::Graph*, GRAPH_TYPE_COUNT>;

class ForgeModule
{
public:
    ForgeModule(std::string name, graphlib::Graph* forward_graph) : name_(name), graphs_{nullptr}
    {
        TT_ASSERT(forward_graph != nullptr);
        graphs_[to_underlying(GraphType::Forward)] = forward_graph;
    }

    void set_graph(GraphType type, graphlib::Graph* graph)
    {
        TT_ASSERT(graph != nullptr);
        graphs_[to_underlying(type)] = graph;
    }

    graphlib::Graph* graph(GraphType type) const
    {
        TT_ASSERT(graphs_[to_underlying(type)] != nullptr);
        return graphs_[to_underlying(type)];
    }

    std::vector<graphlib::Graph*> graphs()
    {
        std::vector<graphlib::Graph*> tmp;
        tmp.reserve(graphs_.size());
        for (auto graph : graphs_) {
            if (graph != nullptr) {
                tmp.push_back(graph);
            }
        }
        return tmp;
    }

    std::string name() const
    {
        return name_;
    }

private:
    std::string name_;
    StaticGraphArray graphs_;
};

} // namespace tt
