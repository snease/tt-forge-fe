// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "exec_graphs.hpp"
#include <utils/assert.hpp>
#include <utils/logger.hpp>

#include "graph_lib/defines.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "passes/amp.hpp"
#include "passes/pre_placer_buda_passes.hpp"

using Graph = tt::graphlib::Graph;

namespace tt::passes
{
    std::vector<tt::graphlib::Graph*> create_execution_graphs(tt::graphlib::Graph * graph)
    {
        Graph *fwd_graph = new Graph(tt::graphlib::IRLevel::IR_TT_FORGE, graph->name() + "_fwd");

        auto topo = graphlib::topological_sort(*graph);

        for (auto input_node : graph->ordered_module_inputs())
        {
            if (input_node->get_epoch_type() != graphlib::NodeEpochType::Forward)
            {
                continue;
            }

            log_info("Cloning input node {} for fwd exec graph", input_node->name());
            auto cloned_input = input_node->clone(input_node->name());
            fwd_graph->add_node(std::move(cloned_input), 0 /*subgraph_id=*/);
        }

        for (auto param : graph->nodes([](const Node* node) { return node->node_type() == NodeType::kInput && node->as<graphlib::InputNode>()->is_parameter(); }))
        {
            log_info("Cloning input node {} for fwd exec graph", param->name());
            auto cloned_input = param->clone(param->name());
            fwd_graph->add_node(std::move(cloned_input), 0 /*subgraph_id=*/);
        }

        for (auto node : topo)
        {
            if (node->node_type() != graphlib::NodeType::kPyOp
                || node->get_epoch_type() != graphlib::NodeEpochType::Forward)
            {
                continue;
            }

            auto cloned_node = node->clone(node->name());
            auto fwd_node = fwd_graph->add_node(std::move(cloned_node), 0 /*subgraph_id=*/);

            for (auto operand : graph->operands(node))
            {
                graphlib::Node *cloned_operand = nullptr;

                if (operand->node_type() == graphlib::NodeType::kInput)
                {
                    TT_ASSERT(fwd_graph->has_node_with_name(operand->name()), "Input node " + operand->name() + " in the new fwd graph should have already been created!");
                    cloned_operand = fwd_graph->get_node_by_name(operand->name());
                }

                if (operand->node_type() == graphlib::NodeType::kPyOp)
                {
                    if (!fwd_graph->has_node_with_name(operand->name()))
                    {
                        cloned_operand = fwd_graph->add_node(operand->clone(operand->name()), 0 /*subgraph_id=*/);
                    }

                    cloned_operand = fwd_graph->get_node_by_name(operand->name());
                }

                TT_ASSERT(cloned_operand != nullptr, "Operand node should have been created by now!");

                TT_ASSERT(graph->get_edges(operand, node).size() == 1, "Expected only one edge between operand and node");
                auto original_edge = graph->get_edges(operand, node)[0];
                graph->add_edge(cloned_operand, fwd_node, original_edge.producer_output_port_id, original_edge.consumer_input_port_id, original_edge.edge_type);
            }

            Node* intermediate_output_node = nullptr;
            for (auto user_edge : graph->user_edges(node, [](const Edge& edge) { return edge.edge_type == EdgeType::kData; }))
            {
                if (graph->node_by_id(user_edge.consumer_node_id)->get_epoch_type() == NodeEpochType::Backward)
                {
                    if (intermediate_output_node == nullptr)
                    {
                        auto intermediate_output = graphlib::create_node<graphlib::OutputNode>(node->name() + "_intermediate_output");
                        intermediate_output->mark_intermediate();

                        intermediate_output_node = fwd_graph->add_node(std::move(intermediate_output), 0 /*subgraph_id=*/);
                    }

                    fwd_graph->add_edge(fwd_node, intermediate_output_node);
                }
            }
        }

        log_info("Created fwd graph: {}", fwd_graph->name());
        for (auto node : fwd_graph->nodes())
        {
             log_info("Node: {}", node->name());
        }

        Graph *bwd_graph = new Graph(tt::graphlib::IRLevel::IR_TT_FORGE, graph->name() + "_bwd");

        return {fwd_graph, bwd_graph};
    }

} // namespace tt::passes
