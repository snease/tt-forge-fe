// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "passes/pre_lowering_passes.hpp"

#include "graph_lib/utils.hpp"
#include "passes/commute_utils.hpp"
#include "python_bindings_common.hpp"
#include "reportify/reportify.hpp"

namespace tt
{

using NodeType = graphlib::NodeType;
using Edge = graphlib::Edge;
using EdgeType = graphlib::EdgeType;

void convert_broadcast_ops_to_tms(Graph *graph)
{
    std::vector<Node *> broadcast_ops = graph->nodes(
        [](Node *node) -> bool
        {
            graphlib::OpNode *op = dynamic_cast<graphlib::OpNode *>(node);
            return op and op->op_name() == "broadcast";
        });

    for (Node *node : broadcast_ops)
    {
        graphlib::OpNode *op = node->as<graphlib::OpNode>();
        graphlib::OpType op_type = op->op_type();
        constexpr bool remove_node = true;
        graphlib::bypass_node(
            graph,
            node,
            remove_node,
            [graph, op_type](Edge new_edge, Edge)
            {
                auto attr = graph->get_edge_attributes(new_edge);
                attr->prepend_tm(op_type);
            });
    }
}

void place_inter_subgraph_queues(graphlib::Graph *graph)
{
    for (Node *n : graph->nodes_by_type(NodeType::kOutput))
    {
        std::vector<Node *> consumers = graph->data_users(n);
        if (consumers.size() == 0)
            continue;
        std::vector<Node *> producers = graph->data_operands(n);
        TT_ASSERT(producers.size() == 1);

        std::cout << "removing node: " << n->name() << std::endl;
        graph->remove_node(n);
        for (Node *consumer : consumers)
        {
            std::cout << "adding edge from: " << producers[0]->name() << " to: " << consumer->name() << std::endl;
            graph->add_edge(producers[0], consumer);
        }
    }
}

static void insert_tile_broadcasts(
    Graph *graph, graphlib::Edge edge, std::vector<int> ignore_dims = {}, bool try_consteval = true)
{
    std::vector<graphlib::OpNode *> tile_broadcasts;
    auto attrs = graph->get_edge_attributes(edge);
    std::vector<graphlib::OpType> tms = attrs->get_tms();
    auto node = graph->node_by_id(edge.producer_node_id);
    auto shape = node->shape();
    auto user_node = graph->node_by_id(edge.consumer_node_id);
    auto user_shape = user_node->shape();
    auto bcast_shape = user_shape.size() > shape.size() ? shape.as_rank(user_shape.size()) : shape;
    bool producer_is_constant =
        (node->node_type() == NodeType::kInput) && (node->as<graphlib::InputNode>()->is_constant());
    auto users = graph->user_data_edges(node);
    std::size_t edge_index = std::find(users.begin(), users.end(), edge) - users.begin();

    auto tags = user_node->as<graphlib::TaggedNode>()->get_tags();
    std::vector<graphlib::OpType> target_tms = tms;
    Edge target_edge = edge;  // edge on which to insert scalar broadcast
    for (std::size_t tm_index = 0; tm_index < tms.size(); tm_index++)
    {
        auto tm = tms[tm_index];
        TT_ASSERT(tm.op == "broadcast", "Only bcast tms allowed on edges before lowering");
        if (tm.op == "broadcast")
        {
            int dim = shape.negative_index(std::get<int>(tm.attr[0]));
            int size = std::get<int>(tm.attr[1]);
            if (size == 1)
                continue;

            if (std::find(ignore_dims.begin(), ignore_dims.end(), dim) != ignore_dims.end())
                continue;

            if (dim < -2)
                continue;  // only R/C

            if (bcast_shape[dim] != 1)
                continue;

            graphlib::OpType op_type("tile_broadcast");
            int brcst_size = (size > 32) ? 32 : size;

            // If we allow conseval for non tile-aligned inputs we end up with very large input tensors that waste DRAM
            try_consteval &= (user_shape[dim] % graphlib::Shape::FORGE_TILE_DIM) == 0;

            // If node is constant, and not divisible by TILE_DIM, we need to broadcast the whole thing
            if (producer_is_constant and user_shape[dim] % graphlib::Shape::FORGE_TILE_DIM != 0)
                brcst_size = size;
            op_type.attr = {dim, brcst_size};

            std::string name = node->name() + "_s_brcst_";
            if (dim < 0)
                name += "m";
            name += std::to_string(abs(dim));
            name += "_" + std::to_string(edge_index) + "_" + std::to_string(tm_index);
            int unique_id = 0;
            while (graph->has_node_with_name(name + (unique_id ? ("_" + std::to_string(unique_id)) : std::string(""))))
                unique_id++;
            name += unique_id ? ("_" + std::to_string(unique_id)) : std::string("");

            log_trace(LogGraphCompiler, "Insert tile broadcast: {} (after) {}", name, node->name());

            auto tile_broadcast = graph->add_node(
                std::make_unique<graphlib::PyOpNode>(name, op_type),
                graph->get_subgraph_id_for_node(edge.consumer_node_id));
            bcast_shape[dim] = brcst_size;
            tile_broadcast->set_shape(bcast_shape);
            tile_broadcast->as<graphlib::TaggedNode>()->add_tags(tags);
            tile_broadcasts.push_back(tile_broadcast);
            auto [new_in_edge, new_out_edge] = graphlib::insert_node_on_edge(graph, target_edge, tile_broadcast);
            tile_broadcast->set_output_df(node->output_df());

            // Set output df after adding to the graph because df will be initialized to consumer df during insertion
            tile_broadcast->set_output_df(node->output_df());

            // All TMs should always go to output
            graph->get_edge_attributes(new_in_edge)->set_tms({});

            // If user shape is not divisible by TILE_DIM, tile_broadcast lowering logic will handle
            // broadcast
            if (user_shape[dim] % graphlib::Shape::FORGE_TILE_DIM == 0)
            {
                graph->get_edge_attributes(new_out_edge)->set_tms(target_tms);
            }
            else
            {
                std::size_t offset = tm_index - (tms.size() - target_tms.size());
                TT_ASSERT(offset < target_tms.size());
                target_tms.erase(target_tms.begin() + offset);
                graph->get_edge_attributes(new_out_edge)->set_tms(target_tms);
            }

            // If there are more broadcasts, insert on this new edge
            target_edge = new_out_edge;
        }
    }

    if (try_consteval and not env_as<bool>("FORGE_DISABLE_TILE_BROADCAST_CONSTEVAL"))
    {
        for (auto op : tile_broadcasts)
        {
            graphlib::try_consteval_op(graph, op, true);
        }
    }
}

void replace_with_broadcasted_const(
    Graph *graph,
    graphlib::ConstantInputNode *constant,
    std::shared_ptr<void> broadcasted_tensor,
    graphlib::Shape target_shape,
    graphlib::PyOpNode *original_tile_bcast)
{
    auto broadcasted_const = graph->add_node(
        graphlib::create_node<graphlib::ConstantInputNode>(
            constant->name() + "_tile_bcast", broadcasted_tensor, target_shape),
        graph->get_subgraph_id_for_node(constant->id()));
    broadcasted_const->set_shape(target_shape);
    broadcasted_const->set_output_df(original_tile_bcast->output_df());
    graphlib::Edge edge(broadcasted_const->id(), 0, original_tile_bcast->id(), 0, graphlib::EdgeType::kData);
    graph->add_edge(edge);
    graph->remove_node(constant);
}

bool safe_to_hoist_past(const Graph *graph, const Node *operand)
{
    if (graph->user_data_edges(operand).size() > 1)
        return false;  // we don't want to deal with this now

    if (graph->operand_data_edges(operand).size() > 1)
        return false;  // not a unary op

    if (operand->node_type() != NodeType::kPyOp)
        return false;

    const std::string &op_type = operand->as<graphlib::PyOpNode>()->op_type().op;

    // Any unary op that doesn't change shape, and not transpose (which could keep the same shape)
    if (op_type == "transpose")
        return false;

    graphlib::Shape incoming_shape = graph->data_operands(operand)[0]->shape();
    graphlib::Shape my_shape = operand->shape();

    return (my_shape.as_vector() == incoming_shape.as_vector());
}

static bool commutable_reshape(graphlib::PyOpNode *reshape)
{
    return (
        (reshape->op_name() == "reshape") or (reshape->op_name() == "squeeze") or (reshape->op_name() == "unsqueeze"));
}

// Swap add and reshape as well as gelu and reshape if reshape only changes the dim size
static bool swap_reshape(Graph *graph, graphlib::PyOpNode *add, graphlib::PyOpNode *reshape, bool requant = false)
{
    if (reshape->is_matmul())
        return false;

    if (not commutable_reshape(reshape) and not requant)
        return false;

    if (env_as<bool>("FORGE_FUSE_MATMUL_GELU"))
    {
        TT_ASSERT(
            (add->op_type().op == "add") || (add->op_type().op == "gelu") || (add->op_type().op == "forge_requantize"));
    }
    else
    {
        TT_ASSERT((add->op_type().op == "add") || (add->op_type().op == "forge_requantize"));
    }

    if (graph->data_users(reshape).size() > 1)
        return false;  // reshape goes to more than just add

    // Check for validity of reshape
    if (reshape->op_type().op == "reshape")
    {
        auto input_shape = graph->data_operands(reshape)[0]->shape();
        auto output_shape = reshape->shape();

        if (input_shape.volume() != output_shape.volume())
            return false;  // can't swap, the reshape actually changes shape

        std::uint32_t index = 0;
        const std::uint32_t in_size = input_shape.size();
        const std::uint32_t out_size = output_shape.size();
        while ((index < in_size) && (index < out_size))
        {
            if (input_shape[in_size - 1 - index] != output_shape[out_size - 1 - index])
                return false;  // change of shape
            index++;
        }
    }

    log_trace(LogGraphCompiler, "Swapping {} and {}", add->name(), reshape->name());

    TT_ASSERT(graph->user_data_edges(reshape).size() == 1);
    graphlib::Edge edge = graph->user_data_edges(reshape).front();
    graphlib::swap(graph, edge);
    return true;
}

static bool has_fusable_upstream_matmul(graphlib::Graph *graph, graphlib::PyOpNode *op, bool requant = false)
{
    if (op == nullptr)
        return false;

    while (not(
        op->is_dense_matmul() || (op->is_depthwise_matmul() and not requant)))  // requant can't be fused to depthwise
    {
        if (not(commutable_reshape(op)))
        {
            if (not(requant and op->is_tm()))  // requant can be commuted through TM
                return false;
        }

        auto operands = graph->data_operands(op);
        TT_ASSERT(operands.size() == 1);
        op = dynamic_cast<graphlib::PyOpNode *>(operands.front());
        if (not op)
            return false;
    }

    // If matmul has more outputs than just to bias, we can't merge
    if (graph->user_data_edges(op).size() > 1)
        return false;

    // If matmul already has bias merged, we can't merge another one
    if (graph->operand_data_edges(op).size() > 2 and !requant)
        return false;

    return true;
}

void fuse_bias(Graph *graph)
{
    // Find matmul + bias, and merge bias into the matmul
    for (Node *node : graphlib::topological_sort(*graph))
    {
        // Look for bias
        if ((node->node_type() != graphlib::kPyOp) || (node->as<graphlib::PyOpNode>()->op_type().op != "add"))
            continue;

        graphlib::PyOpNode *op = node->as<graphlib::PyOpNode>();

        if (op->get_epoch_type() != graphlib::NodeEpochType::Forward)
            continue;

        auto operands = graph->data_operands(op);
        TT_ASSERT(operands.size() == 2);

        // Models coming out TVM frequently have a reshape between matmul and add, which only expands dims
        // Reshape and add need to be swaped in order for the matmul+add to be fused further down.
        if (not has_fusable_upstream_matmul(graph, dynamic_cast<graphlib::PyOpNode *>(operands[0])))
            continue;

        while (swap_reshape(graph, op, operands[0]->as<graphlib::PyOpNode>()))
        {
            operands = graph->data_operands(op);  // "reload" operands
        }

        bool tile_broadcasted = false;
        if (operands[0]->shape()[-1] != operands[1]->shape()[-1] and operands[1]->shape()[-1] == 1)
        {
            auto operand_edges = graph->operand_data_edges(op);
            TT_ASSERT(operand_edges.size() >= 2);
            insert_tile_broadcasts(graph, operand_edges[1], {-2});
            operands = graph->data_operands(op);
            tile_broadcasted = true;
        }

        // Currently, only row-broadcast bias is merged. So, all dims except for the last one should be 1.
        auto shape = operands[1]->shape().as_vector();
        bool correct_shape = (operands[0]->shape()[-1] == operands[1]->shape()[-1]) or tile_broadcasted;
        for (std::size_t i = 0; i < shape.size() - 1; i++)
            if (shape[i] != 1)
                correct_shape = false;

        if (!correct_shape)
            continue;

        // Fused matmul bias will do tile broadcast of the row, and if there's no broadcast on the bias node,
        // then we can't fuse.
        auto tms = graph->get_edge_attributes(graph->operand_data_edges(op)[1])->get_tms();
        bool broadcast = false;
        for (auto tm : tms)
            // Broadcast must be to tile dim, otherwise in-kernel broadcast will broadcast too far
            if ((tm.op == "broadcast") && (std::get<int>(tm.attr[1]) % graphlib::Shape::FORGE_TILE_DIM == 0))
            {
                broadcast = true;
                break;
            }

        if (!broadcast)
        {
            if (not(operands[1]->shape() == operands[0]->shape() and operands[1]->node_type() == graphlib::kInput and
                    operands[1]->as<graphlib::InputNode>()->is_constant()))
            {
                continue;
            }
        }

        // Ok to merge
        log_trace(LogGraphCompiler, "Merging {} and {}", operands[0]->name(), op->name());

        // Create a new bias edge to matmul
        Edge bias_input_edge = graph->operand_data_edges(op)[1];
        Edge new_bias_input_edge = Edge(
            bias_input_edge.producer_node_id,
            bias_input_edge.producer_output_port_id,
            operands[0]->id(),
            2,
            graphlib::EdgeType::kData);
        graph->add_edge(new_bias_input_edge);
        graph->copy_edge_attributes(bias_input_edge, new_bias_input_edge);

        // Get user edges for Add, and copy over to matmul
        auto user_edges = graph->user_edges(op);
        for (Edge edge : user_edges)
        {
            graphlib::PortId producer_output_port_id = 0;
            if (edge.edge_type == graphlib::EdgeType::kAutogradFwdToBwd)
            {
                producer_output_port_id = 2;
            }
            Edge new_edge = Edge(
                operands[0]->id(),
                producer_output_port_id,
                edge.consumer_node_id,
                edge.consumer_input_port_id,
                edge.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(edge, new_edge);
        }

        // The output of fused matmul should match the output of the original add
        operands[0]->as<graphlib::OpNode>()->set_golden_transforms(op->get_golden_transforms());
        operands[0]->as<graphlib::OpNode>()->set_golden_id(op->id());

        // Remove add
        graph->remove_node(op);
    }
}

void fuse_requantize(Graph *graph)
{
    // Find int8 matmul + requant, and merge requant into matmul
    for (Node *node : graphlib::topological_sort(*graph))
    {
        // Look for bias
        if ((node->node_type() != graphlib::kPyOp) ||
            (node->as<graphlib::PyOpNode>()->op_type().op != "forge_requantize"))
            continue;

        graphlib::PyOpNode *op = node->as<graphlib::PyOpNode>();

        if (op->get_epoch_type() != graphlib::NodeEpochType::Forward)
            continue;

        auto operands = graph->data_operands(op);
        TT_ASSERT(operands.size() == 2);

        // Models coming out TVM frequently have a reshape between matmul and add, which only expands dims
        // Reshape and add need to be swaped in order for the matmul+add to be fused further down.
        if (not has_fusable_upstream_matmul(graph, dynamic_cast<graphlib::PyOpNode *>(operands[0]), true /* requant */))
            continue;

        while (swap_reshape(graph, op, operands[0]->as<graphlib::PyOpNode>(), true /* requant */))
        {
            operands = graph->data_operands(op);  // "reload" operands
        }

        auto mm_tms = graph->get_edge_attributes(graph->operand_data_edges(op)[0])->get_tms();
        TT_ASSERT(mm_tms.size() == 0);

        // Ok to merge
        log_trace(LogGraphCompiler, "Merging {} and {}", operands[0]->name(), op->name());

        auto matmul = operands[0]->as<graphlib::PyOpNode>();
        // auto scale = operands[1]->as<graphlib::ConstantInputNode>();
        auto requant_attrs = op->op_type().attr;
        auto matmul_attrs = matmul->op_type().attr;

        auto scale_adge = graph->operand_data_edges(op)[1];
        Edge new_scale_edge = Edge(
            scale_adge.producer_node_id,
            scale_adge.producer_output_port_id,
            operands[0]->id(),
            3,
            graphlib::EdgeType::kData);  // Input0, input1, bias, scale
        graph->add_edge(new_scale_edge);
        graph->copy_edge_attributes(scale_adge, new_scale_edge);

        // copy over zp attrs
        matmul_attrs.push_back(requant_attrs[0]);  // Add requant zp to the back of matmul attr
        passes::update_matmul_attr(matmul, std::get<int>(requant_attrs[0]));
        auto matmul_forge_attr = matmul->op_type().forge_attrs;
        matmul_forge_attr["requant"] = "true";
        matmul->overwrite_forge_attrs(matmul_forge_attr);
        matmul->set_output_df(op->output_df());

        // Get user edges for requant, and copy over to matmul
        auto user_edges = graph->user_edges(op);
        for (Edge edge : user_edges)
        {
            graphlib::PortId producer_output_port_id = 0;
            Edge new_edge = Edge(
                operands[0]->id(),
                producer_output_port_id,
                edge.consumer_node_id,
                edge.consumer_input_port_id,
                edge.edge_type);
            graph->add_edge(new_edge);
            graph->copy_edge_attributes(edge, new_edge);
        }

        // Remove requant
        graph->remove_node(op);
    }
}

void fuse_gelu(Graph *graph)
{
    // Find matmul + gelu, and merge gelu into the matmul
    for (Node *node : graphlib::topological_sort(*graph))
    {
        // Look for gelu
        if ((node->node_type() != graphlib::kPyOp) || (node->as<graphlib::PyOpNode>()->op_type().op != "gelu"))
            continue;

        graphlib::PyOpNode *op = node->as<graphlib::PyOpNode>();

        if (op->get_epoch_type() != graphlib::NodeEpochType::Forward)
            continue;

        auto operands = graph->data_operands(op);
        TT_ASSERT(operands.size() == 1);

        if ((operands[0]->node_type() == graphlib::kPyOp) &&
            ((operands[0]->as<graphlib::PyOpNode>()->op_type().op == "reshape") ||
             (operands[0]->as<graphlib::PyOpNode>()->op_type().op == "squeeze") ||
             (operands[0]->as<graphlib::PyOpNode>()->op_type().op == "unsqueeze")))
        {
            swap_reshape(graph, op, operands[0]->as<graphlib::PyOpNode>());
            operands = graph->data_operands(op);  // "reload" operands
        }

        if (operands[0]->node_type() != graphlib::kPyOp)
            continue;
        auto opnd = operands[0]->as<graphlib::PyOpNode>();
        if (!(opnd->is_matmul()))
            continue;
        if (opnd->is_sparse_matmul())
            continue;

        // If matmul has more outputs than just to gelu, we can't merge
        if (graph->user_data_edges(operands[0]).size() > 1)
            continue;

        auto mmul = operands[0]->as<graphlib::PyOpNode>();
        auto matmul_attrs = mmul->op_type().forge_attrs;
        if (matmul_attrs.find("sfpu_op") != matmul_attrs.end())
            continue;
        matmul_attrs["sfpu_op"] = "gelu";
        mmul->overwrite_forge_attrs(matmul_attrs);

        // Ok to merge
        log_trace(LogGraphCompiler, "Merging {} and {}", operands[0]->name(), op->name());

        // Get user edges for gelu, and copy over to matmul
        operands[0]->as<graphlib::OpNode>()->set_golden_id(op->id());
        graphlib::bypass_node(
            graph, node, true, [graph](Edge new_edge, Edge edge) { graph->copy_edge_attributes(edge, new_edge); });
    }
}

}  // namespace tt
