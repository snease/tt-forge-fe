// SPDX-FileCopyrightText: © 2024 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "mlir_passes.hpp"

// Standard headers
#include <stdexcept>

// MLIR headers
#include "mlir/IR/BuiltinOps.h"

// TTMLIR headers
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"
#include "ttmlir/Dialect/TTNN/Pipelines/TTNNPipelines.h"
#include "ttmlir/Dialect/TTNN/Transforms/Passes.h"
#include "utils/logger.hpp"

namespace tt::passes
{
/// Public API for running MLIR passes and generating binary.
void run_mlir_passes(mlir::OwningOpRef<mlir::ModuleOp> &mlir_module)
{
    static bool _ = []()
    {
        // Register required passes
        mlir::tt::ttir::registerPasses();
        mlir::tt::ttnn::registerPasses();

        // Register pass pipelines
        // This will internally register the pipelines in the MLIR pipeline registry. Then,
        // the registry can be used to lookup the pipeline by its name and add it to the pass manager.
        mlir::tt::ttnn::registerTTNNPipelines();

        return true;
    }();
    (void)_;

    // Create a pass manager.
    mlir::PassManager pm(mlir_module.get()->getName());

    // Get the pipeline info for the wanted pipeline.
    const auto pipelineInfo = mlir::PassPipelineInfo::lookup("ttir-to-ttnn-backend-pipeline");

    // This error handler is necessary when adding the pipeline to the pass manager (via PassPipelineInfo).
    // It's supposed to be called when there's an error during parsing of the pipeline options.
    // However, I think it's wrongly implemented in the MLIR library, so it doesn't get called.
    mlir::function_ref<mlir::LogicalResult(const mlir::Twine &)> err_handler = [](const mlir::Twine &location)
    {
        log_error(LogMLIRCompiler, "Error during parsing pipeline options: {}", location.str());
        return mlir::failure();
    };

    // Create a string to store overrides for the override output layout option.
    std::string override_output_layout = "override-output-layout=";
    override_output_layout += "matmul_1=1x8:l1:width_sharded,";
    override_output_layout += "add_2=1x8:l1:width_sharded,";
    override_output_layout += "add_2_in_1_layout=1x8:l1:width_sharded,";
    override_output_layout += "relu_3=1x8:l1:width_sharded,";
    override_output_layout += "matmul_5=1x1:l1:width_sharded,";
    override_output_layout += "add_6=1x1:l1:width_sharded,";
    override_output_layout += "add_6_in_1_layout=1x1:l1:width_sharded,";
    override_output_layout += "softmax_7=1x1:l1:width_sharded";

    // Create a string to store overrides
    std::string overrides = "";
    overrides += "enable-optimizer=true ";
    overrides += "sharding-pass-enabled=true ";
    overrides += override_output_layout;

    std::cout << std::endl;
    std::cout << "Overrides: " << overrides << std::endl;
    std::cout << std::endl;

    // Pipeline options are empty for now.
    std::string options{overrides};

    auto result = pipelineInfo->addToPipeline(pm, options, err_handler);
    if (mlir::failed(result))
    {
        throw std::runtime_error("Failed to add the pipeline to the pass manager!");
    }

    // Run the pass manager.
    if (mlir::failed(pm.run(mlir_module.get())))
    {
        throw std::runtime_error("Failed to run MLIR compiler pass pipeline.");
    }

#ifdef DEBUG
    // Create a string to store the output
    std::string moduleStr;
    llvm::raw_string_ostream rso(moduleStr);

    // Print the MLIR module
    mlir::OpPrintingFlags printFlags;
    printFlags.enableDebugInfo();
    mlir_module.get()->print(rso, printFlags);

    rso.flush();

    log_trace(LogMLIRCompiler, "MLIR module after running passes:\n{}", moduleStr);
#endif
}
}  // namespace tt::passes
