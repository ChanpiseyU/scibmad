AUTODIFF_JULIA_CODE = """
        using DifferentiationInterface
        import ForwardDiff
        using DifferentiationInterface: gradient, jacobian, derivative
        const backend = AutoForwardDiff()

        function scibmad_call_with_arg(f, args, arg_index, x)
            return f((i == arg_index ? x : args[i] for i in eachindex(args))...)
        end

        function scibmad_derivative_arg(f, backend, args, arg_index, x)
            return derivative(y -> scibmad_call_with_arg(f, args, arg_index, y), backend, x)
        end

        function scibmad_gradient_arg(f, backend, args, arg_index, x)
            return gradient(y -> scibmad_call_with_arg(f, args, arg_index, y), backend, x)
        end

        function scibmad_jacobian_arg(f, backend, args, arg_index, x)
            return jacobian(y -> scibmad_call_with_arg(f, args, arg_index, y), backend, x)
        end

        function scibmad_jacobian_scalar_arg(f, backend, args, arg_index, x)
            return jacobian(y -> scibmad_call_with_arg(f, args, arg_index, y[1]), backend, [x])
        end
        """