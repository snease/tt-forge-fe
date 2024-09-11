// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "exec_graphs.hpp"
#include <memory>
#include <queue>
#include <utils/assert.hpp>
#include <utils/logger.hpp>

#include "forge_graph_module.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"

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

void clone_and_add_to_graph(const Graph *graph, Graph *new_graph, const graphlib::Node *node)
{
    auto cloned_node = node->clone(node->name());
    new_graph->add_node(std::move(cloned_node), 0 /*subgraph_id=*/);
}

bool needs_intermediate_output(const Graph *graph, const graphlib::Node *node)
{
    if (node->node_type() != graphlib::NodeType::kPyOp)
    {
        return false;
    }

    bool has_bwd_edge = false;
    bool has_output_node_already = false;
    for (auto user_edge : graph->user_data_edges(node))
    {
        if (graph->node_by_id(user_edge.consumer_node_id)->get_epoch_type() == graphlib::NodeEpochType::Backward)
        {
            has_bwd_edge = true;
        }

        if (graph->node_by_id(user_edge.consumer_node_id)->node_type() == graphlib::NodeType::kOutput)
        {
            has_output_node_already = true;
        }
    }
    
    return has_bwd_edge && !has_output_node_already;
}

std::unique_ptr<Graph> split_forward(const Graph* graph, const std::vector<graphlib::Node*> &topo)
{
    auto fwd_graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_TT_FORGE, "forward");
    fwd_graph->set_training(graph->training());

    for (auto node : topo)
    {
        if (node->get_epoch_type() != graphlib::NodeEpochType::Forward)
        {
            continue;
        }

        clone_and_add_to_graph(graph, fwd_graph.get(), node);

        if (needs_intermediate_output(graph, node))
        {
            auto intermediate_output = graphlib::create_node<graphlib::OutputNode>(node->name() + "_intermediate");
            intermediate_output->mark_intermediate();
            intermediate_output->set_shape(node->shape());
            intermediate_output->set_output_df(node->output_df());

            auto intermediate_output_node = fwd_graph->add_node(std::move(intermediate_output), 0 /*subgraph_id=*/);
            fwd_graph->add_edge(fwd_graph->get_node_by_name(node->name()), intermediate_output_node);
        }

        for (auto operand : graph->operands(node))
        {
            fwd_graph->add_edge(fwd_graph->get_node_by_name(operand->name()), fwd_graph->get_node_by_name(node->name()));
        }
    }

    std::vector<graphlib::NodeId> fwd_module_inputs;
    for (auto input : graph->ordered_module_inputs())
    {
        if (input->is_forward())
        {
            fwd_module_inputs.push_back(fwd_graph->get_node_by_name(input->name())->id());
        }
    }

    fwd_graph->register_module_inputs(fwd_module_inputs);

    for (auto output : graph->ordered_module_outputs())
    {
        log_info("Adding node {} to outputs", output->name());
        fwd_graph->register_module_outputs({fwd_graph->get_node_by_name(output->name())->id()}, true /* append */);
    }

    std::vector<graphlib::NodeId> fwd_module_intermediates;
    for (auto output : fwd_graph->nodes_by_type(graphlib::NodeType::kOutput))
    {
        if (graph->has_node_with_name(output->name()))
        {
            continue;
        }

        if (output->as<graphlib::OutputNode>()->is_intermediate())
        {
            log_info("Adding node {} to intermediate outputs", output->name());
            fwd_module_intermediates.push_back(output->id());
        }
        else
        {
            log_info("Adding node {} to outputs", output->name());
            fwd_graph->register_module_outputs({output->id()}, true /* append */);
        }
    }

    fwd_graph->register_module_intermediates(fwd_module_intermediates);
    fwd_graph->register_module_outputs(fwd_module_intermediates, true /* append */);

    return fwd_graph;
}

std::unique_ptr<Graph> split_backward(const Graph *graph, const Graph *fwd_graph, const std::vector<graphlib::Node*> &topo)
{
    auto bwd_graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_TT_FORGE, "backward");

    for (auto node : topo)
    {
        if (node->get_epoch_type() != graphlib::NodeEpochType::Backward)
        {
            continue;
        }

        clone_and_add_to_graph(graph, bwd_graph.get(), node);

        for (auto operand : graph->data_operands(node))
        {
            if (bwd_graph->has_node_with_name(operand->name()))
            {
                bwd_graph->add_edge(bwd_graph->get_node_by_name(operand->name()), bwd_graph->get_node_by_name(node->name()));
                continue;
            }

            if (operand->node_type() == graphlib::NodeType::kInput)
            {
                clone_and_add_to_graph(graph, bwd_graph.get(), operand);
                bwd_graph->add_edge(bwd_graph->get_node_by_name(operand->name()), bwd_graph->get_node_by_name(node->name()));
                continue;
            }

            if (operand->get_epoch_type() == graphlib::NodeEpochType::Forward)
            {
                auto fwd_operand = fwd_graph->get_node_by_name(operand->name());
                auto users = fwd_graph->data_users(fwd_operand);

                for (auto user : users)
                {
                    if (user->node_type() == graphlib::NodeType::kOutput)
                    {
                        // Find the intermediate output node in the fwd graph
                        if (bwd_graph->has_node_with_name(user->name()))
                        {
                            bwd_graph->add_edge(bwd_graph->get_node_by_name(user->name()), bwd_graph->get_node_by_name(node->name()));
                            continue;
                        }

                        auto intermediate_input_node = graphlib::create_node<graphlib::InputNode>(user->name(), graphlib::InputNodeType::Activation, false);
                        intermediate_input_node->set_shape(user->shape());
                        intermediate_input_node->set_output_df(user->output_df());

                        bwd_graph->add_node(std::move(intermediate_input_node), 0 /*subgraph_id=*/);
                        bwd_graph->add_edge(bwd_graph->get_node_by_name(user->name()), bwd_graph->get_node_by_name(node->name()));
                    }
                }
            }
        }

        if (node->node_type() == graphlib::NodeType::kQueue)
        {
            auto queue_node = node->as<graphlib::QueueNode>();
            if (queue_node->is_grad_accumulator())
            {
                TT_ASSERT(graph->operand_data_edges(queue_node).size() == 1, "Expected only one operand edge for grad accumulator queue node");
                auto operand = graph->data_operands(queue_node)[0];
                auto output_node = graphlib::create_node<graphlib::OutputNode>(queue_node->name() + "_grad_accumulator");
                output_node->set_shape(queue_node->shape());
                log_info("Setting shape of {} output node to {}", output_node->name(), queue_node->shape());
                output_node->set_output_df(queue_node->output_df());
                auto grad_out = bwd_graph->add_node(std::move(output_node), 0 /*subgraph_id=*/);

                auto cloned_operand = bwd_graph->get_node_by_name(operand->name());
                bwd_graph->add_edge(cloned_operand, grad_out, 0, 0, graphlib::EdgeType::kData);
            }
        }
    }

    std::vector<graphlib::NodeId> bwd_module_inputs;
    for (auto input : graph->ordered_module_inputs())
    {
        if (input->is_backward())
        {
            bwd_module_inputs.push_back(bwd_graph->get_node_by_name(input->name())->id());
        }
    }

    for (auto input: fwd_graph->ordered_module_outputs())
    {
        if (bwd_graph->has_node_with_name(input->name()))
        {
            bwd_module_inputs.push_back(bwd_graph->get_node_by_name(input->name())->id());
        }

    }

    for (auto bwd_input : bwd_graph->nodes_by_type(graphlib::NodeType::kInput))
    {
        if (is_loss_input_node(bwd_input) || graphlib::is_constant_input(bwd_input) || is_parameter_node(bwd_input))
        {
            continue;
        }

        if (std::find(bwd_module_inputs.begin(), bwd_module_inputs.end(), bwd_input->id()) != bwd_module_inputs.end())
        {
            continue;
        }

        bwd_module_inputs.push_back(bwd_input->id());
    }

    std::vector<graphlib::NodeId> bwd_module_intermediates;
    for (auto intermediate : fwd_graph->ordered_module_intermediates())
    {
        TT_ASSERT(bwd_graph->has_node_with_name(intermediate->name()), "Intermediate node not found in bwd graph");
        bwd_module_intermediates.push_back(bwd_graph->get_node_by_name(intermediate->name())->id());
    }

    bwd_graph->register_module_intermediates(bwd_module_intermediates);
    bwd_graph->register_module_inputs(bwd_module_inputs);

    std::vector<graphlib::NodeId> bwd_module_outputs;
    for (auto output : bwd_graph->nodes_by_type(graphlib::NodeType::kOutput))
    {
        bwd_module_outputs.push_back(output->id());
    }

    bwd_graph->register_module_outputs(bwd_module_outputs);
    return bwd_graph;
}

void create_execution_graphs(tt::ForgeGraphModule& module)
{
    auto graph = module.get_graph(GraphType::Forward);

    for (auto node : graph->nodes_by_type(graphlib::NodeType::kPyOp))
    {
        if (node->as<graphlib::PyOpNode>()->op_type().op == "nop")
        {
            graphlib::bypass_node(graph, node, true /* remove_node */);
        }
    }

    auto topo = graphlib::topological_sort(*graph);

    auto fwd_graph = split_forward(graph, topo);
    auto bwd_graph = split_backward(graph, fwd_graph.get(), topo);

    fwd_graph->dump("split_exec_graphs_fwd");
    bwd_graph->dump("split_exec_graphs_bwd");

    module.set_graph(GraphType::Forward, fwd_graph.release());
    module.set_graph(GraphType::Backward, bwd_graph.release());
}

} // namespace tt::passes
