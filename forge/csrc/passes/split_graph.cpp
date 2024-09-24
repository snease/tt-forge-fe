// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "split_graph.hpp"

#include <memory>
#include <utils/assert.hpp>
#include <utils/logger.hpp>

#include "forge_graph_module.hpp"
#include "graph_lib/defines.hpp"
#include "graph_lib/graph.hpp"
#include "graph_lib/node.hpp"
#include "graph_lib/node_types.hpp"
#include "graph_lib/utils.hpp"
#include "reportify/reportify.hpp"

using Graph = tt::graphlib::Graph;

namespace tt::passes
{

bool is_parameter_node(const tt::graphlib::Node *node)
{
    return node->node_type() == graphlib::NodeType::kInput && node->as<graphlib::InputNode>()->is_parameter();
}

bool is_loss_input_node(const tt::graphlib::Node *node)
{
    return node->node_type() == graphlib::NodeType::kInput && node->as<graphlib::InputNode>()->is_loss();
}

void clone_and_add(const graphlib::Node *node, Graph *new_graph)
{
    auto cloned_node = node->clone(node->name());
    new_graph->add_node(std::move(cloned_node), 0 /*subgraph_id=*/);
}

// 1. Clone a node from the source graph
// 2. Add it to the dest graph
// 3. Replicate all operand edges from the source graph to the dest graph
void clone_and_connect_operands(const graphlib::Graph* src_graph, const graphlib::Node* src_node, Graph* dst_graph)
{
    auto cloned_node = src_node->clone(src_node->name());
    dst_graph->add_node(std::move(cloned_node), 0 /*subgraph_id=*/);

    auto& src_node_name = src_node->name();

    for (auto operand : src_graph->operands(src_node))
    {
        auto dst_operand = dst_graph->get_node_by_name(operand->name());
        dst_graph->add_edge(dst_operand, dst_graph->get_node_by_name(src_node_name));
    }
}

bool needs_intermediate_output(const Graph *graph, const graphlib::Node *node)
{
    if (!node->is_forward() || node->node_type() != graphlib::NodeType::kPyOp)
    {
        return false;
    }

    auto user_edges = graph->user_data_edges(node);

    // Intermediate node is needed if an op in forward graph has a data user in the backward graph.
    bool has_bwd_user = std::any_of(user_edges.begin(), user_edges.end(), [graph](const graphlib::Edge &edge) {
        return graph->node_by_id(edge.consumer_node_id)->is_backward();
    });

    // Don't add duplicate intermediate output nodes.
    bool has_output_node_already = std::any_of(user_edges.begin(), user_edges.end(), [graph](const graphlib::Edge &edge) {
        return graph->node_by_id(edge.consumer_node_id)->node_type() == graphlib::NodeType::kOutput;
    });

    return has_bwd_user && !has_output_node_already;
}

// Create a forward graph by extracting all forward nodes from the original graph.
// The forward graph also needs to have intermediate output nodes for ops that have data users in the backward graph.
std::unique_ptr<Graph> split_forward(const Graph* graph, const std::vector<graphlib::Node*> &topo)
{
    auto fwd_graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_TT_FORGE, "forward");
    fwd_graph->set_training(graph->training());

    // Create a forward graph by cloning all forward nodes from the original graph.
    // Also, establish the same data edges between forward nodes as in the original graph.
    for (auto node : topo)
    {
        if (!node->is_forward())
        {
            continue;
        }

        clone_and_connect_operands(graph, node, fwd_graph.get());
    }

    // Add all the module inputs to the forward graph (keeping the same order as in the original graph).
    std::vector<graphlib::NodeId> fwd_module_inputs;
    for (auto input : graph->ordered_module_inputs())
    {
        if (input->is_forward())
        {
            fwd_module_inputs.push_back(fwd_graph->get_node_by_name(input->name())->id());
        }
    }

    // Since we are splitting the graph, we need to add intermediate output nodes for all ops which will have
    // data users outside of this graph.
    for (auto node : graph->nodes())
    {
        if (needs_intermediate_output(graph, node))
        {
            auto intermediate_output = graphlib::create_node<graphlib::OutputNode>(node->name() + "_intermediate");
            intermediate_output->mark_intermediate();
            intermediate_output->set_shape(node->shape());
            intermediate_output->set_output_df(node->output_df());

            auto intermediate_output_node = fwd_graph->add_node(std::move(intermediate_output), 0 /*subgraph_id=*/);
            fwd_graph->add_edge(fwd_graph->get_node_by_name(node->name()), intermediate_output_node);
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

    fwd_graph->register_module_outputs(fwd_module_intermediates, true /* append */);

    return fwd_graph;
}

std::unique_ptr<Graph> split_backward(const Graph *graph, const Graph *fwd_graph, const std::vector<graphlib::Node*> &topo)
{
    auto bwd_graph = std::make_unique<Graph>(tt::graphlib::IRLevel::IR_TT_FORGE, "backward");
    bwd_graph->set_training(graph->training());

    // Adding all the intermediate outputs from the forward graph as inputs to the backward graph.
    // Order is important, so we add them in the order they were added to the forward graph.
    for (auto intermediate_output : fwd_graph->ordered_intermediates())
    {
        log_info("Adding intermediate output {} as input to backward graph", intermediate_output->name());
        auto intermediate_output_node = graphlib::create_node<graphlib::InputNode>(intermediate_output->name(), graphlib::InputNodeType::Activation, false);
        intermediate_output_node->set_shape(intermediate_output->shape());
        intermediate_output_node->set_output_df(intermediate_output->output_df());

        bwd_graph->add_node(std::move(intermediate_output_node), 0 /*subgraph_id=*/);
    }

    for (auto node : topo)
    {
        if (!node->is_backward())
        {
            continue;
        }

        clone_and_add(node, bwd_graph.get());

        for (auto operand : graph->data_operands(node))
        {
            if (bwd_graph->has_node_with_name(operand->name()))
            {
                bwd_graph->add_edge(bwd_graph->get_node_by_name(operand->name()), bwd_graph->get_node_by_name(node->name()));
                continue;
            }

            if (operand->node_type() == graphlib::NodeType::kInput)
            {
                clone_and_add(operand, bwd_graph.get());
                bwd_graph->add_edge(bwd_graph->get_node_by_name(operand->name()), bwd_graph->get_node_by_name(node->name()));
                continue;
            }

            if (operand->is_forward())
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

    for (auto queue : bwd_graph->nodes_by_type(graphlib::NodeType::kQueue)) {
        bwd_graph->remove_node(queue);
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

    bwd_graph->register_module_inputs(bwd_module_inputs);

    std::vector<graphlib::NodeId> bwd_module_outputs;
    for (auto output : bwd_graph->nodes_by_type(graphlib::NodeType::kOutput))
    {
        bwd_module_outputs.push_back(output->id());
    }

    bwd_graph->register_module_outputs(bwd_module_outputs);
    return bwd_graph;
}

ForgeGraphModule split_graph(graphlib::Graph* graph)
{
    auto topo = graphlib::topological_sort(*graph);

    auto fwd_graph = split_forward(graph, topo);
    reportify::dump_graph(graph->name(), "split_fwd", fwd_graph.get());

    ForgeGraphModule module(graph->name(), fwd_graph.release());

    if (!graph->training()) {
        // We're not in training mode, so we don't need to split the graph further.
        return module;
    }

    auto bwd_graph = split_backward(graph, module.get_graph(GraphType::Forward), topo);
    reportify::dump_graph(graph->name(), "split_bwd", bwd_graph.get());

    module.set_graph(GraphType::Backward, bwd_graph.release());
    return module;
}

} // namespace tt::passes