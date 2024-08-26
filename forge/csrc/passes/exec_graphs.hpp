// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

namespace tt::graphlib
{
class Graph;
}

namespace tt::passes 
{
    // Create execution graphs from the TT-Forge graph.
    std::vector<tt::graphlib::Graph*> create_execution_graphs(tt::graphlib::Graph * graph);

} // namespace tt:passes

