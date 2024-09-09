// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "forge_module.hpp"

namespace tt::graphlib
{
class Graph;
}

namespace tt::passes 
{
    // Create execution graphs from the TT-Forge graph.
    void create_execution_graphs(tt::ForgeModule& graph);

} // namespace tt:passes

