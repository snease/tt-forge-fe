// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "exec_graphs.hpp"
#include <memory>
#include <utils/assert.hpp>
#include <utils/logger.hpp>

#include "forge_module.hpp"
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
    bool is_activation_node(const tt::graphlib::Node *node)
    {
        if (node->node_type() != graphlib::NodeType::kInput)
        {
            return false;
        }

        return node->as<graphlib::InputNode>()->is_activation();
    }

    bool is_parameter_node(const tt::graphlib::Node *node)
    {
        if (node->node_type() != graphlib::NodeType::kInput)
        {
            return false;
        }

        return node->as<graphlib::InputNode>()->is_parameter();
    }

    bool is_regular_output_node(const tt::graphlib::Node *node)
    {
        return node->node_type() == graphlib::NodeType::kOutput && !node->as<graphlib::OutputNode>()->is_intermediate();
    }

    bool is_intermediate_output_node(const tt::graphlib::Node *node)
    {
        return node->node_type() == graphlib::NodeType::kOutput && node->as<graphlib::OutputNode>()->is_intermediate();
    }

    bool is_loss_input_node(const tt::graphlib::Node *node)
    {
        return node->node_type() == graphlib::NodeType::kInput && node->as<graphlib::InputNode>()->is_loss();
    }

    bool is_loss_output_node(const tt::graphlib::Node *node)
    {
        return node->node_type() == graphlib::NodeType::kOutput && node->as<graphlib::OutputNode>()->is_loss_output();
    }

    std::vector<graphlib::NodeId> map_to_ids(const std::vector<tt::graphlib::Node *> &nodes)
    {
        std::vector<graphlib::NodeId> ids;
        ids.reserve(nodes.size());
        for (auto node : nodes)
        {
            ids.push_back(node->id());
        }

        return ids;
    }

    void create_execution_graphs(tt::ForgeModule& module)
    {
        auto graph = module.graph(GraphType::Forward);

        auto fwd_graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_TT_FORGE, "forward");

        for (auto node : graph->nodes_by_type(graphlib::NodeType::kPyOp))
        {
            if (node->as<graphlib::PyOpNode>()->op_type().op == "nop")
            {
                graphlib::bypass_node(graph, node, true /* remove_node */);
            }
        }

        auto topo = graphlib::topological_sort(*graph);

        // for (auto input_node : graph->ordered_module_inputs())
        // {
        //     if (input_node->get_epoch_type() != graphlib::NodeEpochType::Forward)
        //     {
        //         continue;
        //     }
        //
        //     log_info("Cloning input node {} for fwd exec graph", input_node->name());
        //     auto cloned_input = input_node->clone(input_node->name());
        //     fwd_graph->add_node(std::move(cloned_input), 0 /*subgraph_id=*/);
        // }

        for (auto param : graph->nodes([](const Node* node) { return node->node_type() == NodeType::kInput && node->get_epoch_type() == graphlib::NodeEpochType::Forward; }))
        {
            log_info("Cloning input node {} for fwd exec graph", param->name());
            auto input_node = param->as<graphlib::InputNode>();
            auto cloned_input = input_node->clone(param->name());
            std::unique_ptr<graphlib::InputNode> input_node_ptr(cloned_input.release()->as<graphlib::InputNode>());
            fwd_graph->add_node(std::move(input_node_ptr), 0 /*subgraph_id=*/);
        }

        std::unordered_map<std::string, std::string> node_to_intermediate_map;

        for (auto node : topo)
        {
            if (node->node_type() == graphlib::NodeType::kInput 
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
                    TT_ASSERT(fwd_graph->has_node_with_name(operand->name()), "Operand node " + operand->name() + " in the new fwd graph should have already been created!");
                    if (!fwd_graph->has_node_with_name(operand->name()))
                    {
                        cloned_operand = fwd_graph->add_node(operand->clone(operand->name()), 0 /*subgraph_id=*/);
                    }

                    cloned_operand = fwd_graph->get_node_by_name(operand->name());
                }

                TT_ASSERT(cloned_operand != nullptr, "Operand node should have been created by now!");

                TT_ASSERT(graph->get_edges(operand, node).size() == 1, "Expected only one edge between operand and node");
                auto original_edge = graph->get_edges(operand, node)[0];
                fwd_graph->add_edge(cloned_operand, fwd_node, original_edge.producer_output_port_id, original_edge.consumer_input_port_id, original_edge.edge_type);
            }

            Node* intermediate_output_node = nullptr;
            for (auto user_edge : graph->user_edges(node, [](const Edge& edge) { return edge.edge_type == EdgeType::kData; }))
            {
                if (graph->node_by_id(user_edge.consumer_node_id)->get_epoch_type() == NodeEpochType::Backward)
                {
                    if (intermediate_output_node == nullptr)
                    {
                        auto intermediate_output = graphlib::create_node<graphlib::OutputNode>(node->name() + "_intermediate");
                        intermediate_output->mark_intermediate();
                        intermediate_output->set_shape(node->shape());
                        intermediate_output->set_output_df(node->output_df());

                        intermediate_output_node = fwd_graph->add_node(std::move(intermediate_output), 0 /*subgraph_id=*/);
                        node_to_intermediate_map[node->name()] = intermediate_output_node->name();

                        fwd_graph->add_edge(fwd_node, intermediate_output_node);
                    }
                }
            }
        }

        auto activation_nodes = map_to_ids(fwd_graph->nodes(is_activation_node));
        auto parameter_nodes = map_to_ids(fwd_graph->nodes(is_parameter_node));

        auto regular_output_nodes = map_to_ids(fwd_graph->nodes(is_regular_output_node));
        auto intermediate_output_nodes = map_to_ids(fwd_graph->nodes(is_intermediate_output_node));
        regular_output_nodes.insert(regular_output_nodes.end(), intermediate_output_nodes.begin(), intermediate_output_nodes.end());

        if (!graph->training())
        {
            module.set_graph(GraphType::Forward, fwd_graph.release());
            return;
        }

        auto bwd_graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_TT_FORGE, "backward");

        for (auto input_node : graph->nodes([](const Node* node) { return node->node_type() == NodeType::kInput; }))
        {
            if (input_node->get_epoch_type() == graphlib::NodeEpochType::Forward)
            {
                bool has_bwd_users = false;
                for (auto user : graph->data_users(input_node))
                {
                    if (user->get_epoch_type() == graphlib::NodeEpochType::Backward)
                    {
                        has_bwd_users = true;
                        break;
                    }
                }

                if (!has_bwd_users)
                {
                    continue;
                }
            }

            log_info("Cloning input node {} for bwd exec graph", input_node->name());
            auto cloned_input = input_node->clone(input_node->name());
            std::unique_ptr<graphlib::InputNode> input_node_ptr(cloned_input.release()->as<graphlib::InputNode>());
            bwd_graph->add_node(std::move(input_node_ptr), 0 /*subgraph_id=*/);
        }

        for (auto fwd_out_node : fwd_graph->nodes_by_type(graphlib::NodeType::kOutput))
        {
            if (fwd_out_node->as<graphlib::OutputNode>()->is_intermediate())
            {
                log_info("Creating intermediate input {} for bwd graph.", fwd_out_node->name());
                auto intermediate_input = graphlib::create_node<graphlib::InputNode>(fwd_out_node->name(), graphlib::InputNodeType::Activation, false);
                intermediate_input->set_shape(fwd_out_node->shape());
                intermediate_input->set_output_df(fwd_out_node->output_df());
                bwd_graph->add_node(std::move(intermediate_input), 0 /*subgraph_id=*/);
            }
        }

        log_info("Going through topo");
        for (auto node : topo)
        {
            if (node->node_type() != graphlib::NodeType::kPyOp
                || node->get_epoch_type() != graphlib::NodeEpochType::Backward)
            {
                continue;
            }

            auto cloned_node = node->clone(node->name());
            auto bwd_node = bwd_graph->add_node(std::move(cloned_node), 0 /*subgraph_id=*/);

            for (auto operand : graph->data_operands(node))
            {
                graphlib::Node *cloned_operand = nullptr;

                if (operand->node_type() == graphlib::NodeType::kInput)
                {
                    TT_ASSERT(bwd_graph->has_node_with_name(operand->name()), "Input node " + operand->name() + " in the new bwd graph should have already been created!");
                    cloned_operand = bwd_graph->get_node_by_name(operand->name());
                }

                if (operand->node_type() == graphlib::NodeType::kPyOp)
                {
                    if (operand->get_epoch_type() == graphlib::NodeEpochType::Forward)
                    {
                        TT_ASSERT(node_to_intermediate_map.find(operand->name()) != node_to_intermediate_map.end(), "Intermediate output node for " + operand->name() + " not found!");
                        cloned_operand = bwd_graph->get_node_by_name(node_to_intermediate_map[operand->name()]);
                    }
                    else 
                    {

                        TT_ASSERT(bwd_graph->has_node_with_name(operand->name()), "Operand node " + operand->name() + " in the new bwd graph should have already been created!");
                        cloned_operand = bwd_graph->get_node_by_name(operand->name());
                    }
                }

                TT_ASSERT(cloned_operand != nullptr, "Operand node should have been created by now!");

                // TT_ASSERT(graph->edges(operand, node).size() == 1, "Expected only one edge between operand and node");
                log_info("Adding edge between {} and {}", operand->name(), bwd_node->name());
                for (auto edge: graph->user_edges(operand))
                {
                    log_info("Edge: {} -> {}", graph->node_by_id(edge.producer_node_id)->name(), graph->node_by_id(edge.consumer_node_id)->name());
                }
                TT_ASSERT(graph->get_edges(operand, node).size() > 0, "Expected at least one edge between operand and node");
                auto original_edge = graph->get_edges(operand, node)[0];
                bwd_graph->add_edge(cloned_operand, bwd_node, original_edge.producer_output_port_id, original_edge.consumer_input_port_id, graphlib::EdgeType::kData);
                log_info("Added!");
            }
        }

        log_info("Creating grad accumulator nodes for bwd graph");
        for (auto queue_node: graph->nodes_by_type(graphlib::NodeType::kQueue))
        {
            if (!queue_node->as<graphlib::QueueNode>()->is_grad_accumulator())
            {
                continue;
            }

            TT_ASSERT(graph->operand_data_edges(queue_node).size() == 1, "Expected only one operand edge for grad accumulator queue node");
            auto operand = graph->data_operands(queue_node)[0];
            auto output_node = graphlib::create_node<graphlib::OutputNode>(queue_node->name() + "_grad_accumulator");
            output_node->set_shape(queue_node->shape());
            output_node->set_output_df(queue_node->output_df());
            auto grad_out = bwd_graph->add_node(std::move(output_node), 0 /*subgraph_id=*/);
            
            auto cloned_operand = bwd_graph->get_node_by_name(operand->name());
            bwd_graph->add_edge(cloned_operand, grad_out, 0, 0, EdgeType::kData);
        }

        activation_nodes = map_to_ids(bwd_graph->nodes(is_activation_node));
        auto loss_nodes = map_to_ids(bwd_graph->nodes(is_loss_input_node));
        parameter_nodes = map_to_ids(bwd_graph->nodes(is_parameter_node));
        activation_nodes.insert(activation_nodes.end(), loss_nodes.begin(), loss_nodes.end());

        regular_output_nodes = map_to_ids(bwd_graph->nodes(is_regular_output_node));
        intermediate_output_nodes = map_to_ids(bwd_graph->nodes(is_intermediate_output_node));

        fwd_graph->dump("split_exec_graphs_fwd");
        bwd_graph->dump("split_exec_graphs_bwd");

        module.set_graph(GraphType::Forward, fwd_graph.release());
        module.set_graph(GraphType::Backward, bwd_graph.release());
    }

} // namespace tt::passes
