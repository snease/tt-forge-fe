#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nlohmann/json.hpp"

#include "shape.hpp"
#include "python_bindings_common.hpp"


namespace py = pybind11;

namespace tt::graphlib
{
enum class RuntimeTensorTransformType
{
    NoTransform = 0,
    ReinterpretShape,
    Prestride,
    EmbeddingIndex,
    ConstantInput,
    Unpad,
    Concatenate,
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    tt::graphlib::RuntimeTensorTransformType,
    {
        {tt::graphlib::RuntimeTensorTransformType::NoTransform, "NoTransform"},
        {tt::graphlib::RuntimeTensorTransformType::ReinterpretShape, "ReinterpretShape"},
        {tt::graphlib::RuntimeTensorTransformType::Prestride, "Prestride"},
        {tt::graphlib::RuntimeTensorTransformType::EmbeddingIndex, "EmbeddingIndex"},
        {tt::graphlib::RuntimeTensorTransformType::ConstantInput, "ConstantInput"},
        {tt::graphlib::RuntimeTensorTransformType::Unpad, "Unpad"},
        {tt::graphlib::RuntimeTensorTransformType::Concatenate, "Concatenate"},
    });

class RuntimeTensorTransform
{
    // TODO: Refactor this properly
   public:
    RuntimeTensorTransformType type = RuntimeTensorTransformType::NoTransform;

    // ReinterpretShape
    graphlib::Shape original_shape;
    graphlib::Shape reinterpreted_shape;

    // Unpad
    graphlib::Shape unpadded_shape;

    RuntimeTensorTransform() = default;

    // Temporary fields for Prestride until refactor
    int stride_height;
    int stride_width;
    int kernel_height;
    int kernel_width;

    int concat_group;
    int concat_index;
    int concat_dim;

    RuntimeTensorTransform(
        RuntimeTensorTransformType type,
        graphlib::Shape original_shape,
        graphlib::Shape reinterpreted_shape,
        graphlib::Shape unpadded_shape,
        int stride_height,
        int stride_width,
        int kernel_height,
        int kernel_width,
        int concat_group,
        int concat_index,
        int concat_dim) :
        type(type),
        original_shape(original_shape),
        reinterpreted_shape(reinterpreted_shape),
        unpadded_shape(unpadded_shape),
        stride_height(stride_height),
        stride_width(stride_width),
        kernel_height(kernel_height),
        kernel_width(kernel_width),
        concat_group(concat_group),
        concat_index(concat_index),
        concat_dim(concat_dim)
    {
    }

    RuntimeTensorTransform(Shape original_shape, Shape reinterpreted_shape)
    {
        this->type = RuntimeTensorTransformType::ReinterpretShape;

        this->original_shape = original_shape;
        this->reinterpreted_shape = reinterpreted_shape;
    }
    RuntimeTensorTransform(Shape unpadded_shape)
    {
        this->type = RuntimeTensorTransformType::Unpad;

        this->unpadded_shape = unpadded_shape;
    }

    static RuntimeTensorTransform ConcatenateOnHost(int group, int index, int dim)
    {
        RuntimeTensorTransform transform;
        transform.type = RuntimeTensorTransformType::Concatenate;
        transform.concat_group = group;
        transform.concat_index = index;
        transform.concat_dim = dim;
        return transform;
    }

    static RuntimeTensorTransform EmbeddingIndex(Shape original_shape)
    {
        RuntimeTensorTransform transform;
        transform.type = RuntimeTensorTransformType::EmbeddingIndex;
        transform.original_shape = original_shape;
        return transform;
    }

    void swap_original_and_reinterpreted_shapes()
    {
        if (this->type == RuntimeTensorTransformType::ReinterpretShape)
        {
            std::swap(original_shape, reinterpreted_shape);
        }
    }

    void set_constant_input_tensor(py::object tensor)
    {
        this->type = RuntimeTensorTransformType::ConstantInput;
        this->constant_tensor = make_shared_py_object(tensor);
    }

    py::object get_constant_input_tensor() { return borrow_shared_py_object(this->constant_tensor); }

    NLOHMANN_DEFINE_TYPE_INTRUSIVE(
        RuntimeTensorTransform,
        type,
        original_shape,
        reinterpreted_shape,
        unpadded_shape,
        stride_height,
        stride_width,
        kernel_height,
        kernel_width,
        concat_group,
        concat_index,
        concat_dim);

   private:
    // Constant Input
    std::shared_ptr<void> constant_tensor;
};

} // namespace tt::graphlib
